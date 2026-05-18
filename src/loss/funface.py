
import math

import torch


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class FunFace(torch.nn.Module):

    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=0.01,
                 mixing_factor=0.1
                 ):
        super(FunFace, self).__init__()
        self.classnum = classnum
        self.kernel = torch.nn.Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.eps = 1.e-5
        self.h = h
        self.s = s
        self.mixing_factor = mixing_factor

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('batch_mean_norms', torch.ones(1)*(20))
        self.register_buffer('batch_std_norms', torch.ones(1)*100)
        self.register_buffer('batch_mean_cr', torch.ones(1)*(2))
        self.register_buffer('batch_std_cr', torch.ones(1)*10)

    def forward(self, embbedings, label):

        kernel_norm = l2_norm(self.kernel, axis=0)
        norms = torch.norm(embbedings, 2, 1, True)
        embbedings = torch.div(embbedings, norms)

        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # Feature Norm
        with torch.no_grad():
            mean_norm = safe_norms.mean().detach()
            std_norm = safe_norms.std().detach()
            self.batch_mean_norms = mean_norm * self.t_alpha + (1. - self.t_alpha) * self.batch_mean_norms
            self.batch_std_norms =  std_norm * self.t_alpha + (1. - self.t_alpha) * self.batch_std_norms

        margin_scaler_norm = (safe_norms - self.batch_mean_norms) / (self.batch_std_norms+self.eps) # 66% between -1, 1
        margin_scaler_norm = margin_scaler_norm * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler_norm = torch.clip(margin_scaler_norm, -1., 1.)

        # # Certainty-Ratio
        index = torch.where(label != -1)[0]
        with torch.no_grad():
            distmat=cosine[index,label.view(-1)].detach().clone()
            max_negative_cloned=cosine.detach().clone()
            max_negative_cloned[index,label.view(-1)]= -self.eps
            max_negative, _=max_negative_cloned.max(dim=1)

        nccc = torch.clip(max_negative[index, None], 0., 1.)
        pcc = torch.clip(distmat[index,None], 0., 1.)

        cr = (pcc / (nccc + self.eps))

        # Feature Norm
        with torch.no_grad():
            mean_cr = cr.mean().detach()
            std_cr = cr.std().detach()
            self.batch_mean_cr = mean_cr * self.t_alpha + (1. - self.t_alpha) * self.batch_mean_cr
            self.batch_std_cr =  std_cr * self.t_alpha + (1. - self.t_alpha) * self.batch_std_cr

        margin_scaler_cr = (cr - self.batch_mean_cr) / (self.batch_std_cr+self.eps) # 66% between -1, 1
        margin_scaler_cr = margin_scaler_cr * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler_cr = torch.clip(margin_scaler_cr, -1., 1.)

        joint_margin_scaler = self.mixing_factor * margin_scaler_cr + (1. - self.mixing_factor) * margin_scaler_norm 

        # # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * joint_margin_scaler * -1 
        m_arc = m_arc * g_angular

        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        #g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * joint_margin_scaler) 
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m, None, mean_norm, std_norm