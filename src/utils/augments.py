
import random

import torch
import torchvision
from torchvision.transforms import v2 as T


class AddNoise(torch.nn.Module):

    def __init__(self, min_noise=0.05, max_noise=0.2, noise_types=("uniform", "gaussian"), epsilon=1e-3):
        super().__init__()
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.noise_types = list(noise_types)
        self.epsilon = epsilon

    def forward(self, img):
        noise_type = random.choice(self.noise_types)
        if noise_type == "uniform":
            return torch.clamp((a:=random.uniform(self.min_noise, self.max_noise)) * torch.empty_like(img).uniform_(-1., 1.) +\
                               (1. - a) * img, -1. + self.epsilon, 1. - self.epsilon)
        elif noise_type == "gaussian":
            return torch.clamp((a:=random.uniform(self.min_noise, self.max_noise)) * torch.empty_like(img).normal_(0., 1.) +\
                               (1. - a) * img, -1. + self.epsilon, 1. - self.epsilon)


class Crop(torch.nn.Module):

    def __init__(self, use_ada, **kwargs):
        super().__init__()
        crop_kwargs = dict(kwargs)
        self.fill = crop_kwargs.pop("fill", -1)

        self.random_resized_crop = T.RandomResizedCrop(
            size=crop_kwargs.pop("size", (112, 112)),
            scale=crop_kwargs.pop("scale", (.2, 1.) if use_ada else (0.5, 1.0)),
            ratio=crop_kwargs.pop("ratio", (0.75, 1.3333333333333333)),
            **crop_kwargs,
        )

    def forward(self, img):
        new_img = torch.full_like(img, self.fill)
        i, j, h, w = self.random_resized_crop.get_params(
            img,
            self.random_resized_crop.scale,
            self.random_resized_crop.ratio)
        cropped = T.functional.crop(img, i, j, h, w)
        new_img[:,i:i+h,j:j+w] = cropped
        return new_img


class LowRes(torch.nn.Module):

    def __init__(self, use_ada, min_res=None, max_res=None, base_size=112, interpolation_types=None):
        super().__init__()
        self.min_res = base_size * .2 if min_res is None and use_ada else (16. if min_res is None else min_res)
        self.max_res = base_size * 1. if max_res is None and use_ada else (64. if max_res is None else max_res)

        self._interpolation_types = _resolve_interpolation_types(interpolation_types) if interpolation_types is not None else [
            torchvision.transforms.InterpolationMode.BILINEAR, 
            torchvision.transforms.InterpolationMode.NEAREST, 
            torchvision.transforms.InterpolationMode.NEAREST_EXACT, 
            torchvision.transforms.InterpolationMode.BILINEAR,
            torchvision.transforms.InterpolationMode.BICUBIC
            ]
        
        self.resize_transform = T.functional.resize

    def forward(self, img):
        res_ = int(random.uniform(self.min_res, self.max_res))
        inter_type = random.choice(self._interpolation_types)

        original_size = img.shape[1]

        img = self.resize_transform(img, [res_, res_], interpolation=inter_type)
        return self.resize_transform(img, [original_size, original_size], interpolation=inter_type)


def _resolve_interpolation_types(interpolation_types):
    resolved = []
    for interpolation_type in interpolation_types:
        if isinstance(interpolation_type, str):
            resolved.append(getattr(torchvision.transforms.InterpolationMode, interpolation_type.upper()))
        else:
            resolved.append(interpolation_type)
    return resolved


class Augmentor(torch.nn.Module):

    def __init__(
        self,
        enable,
        use_ada=False,
        probabilities=None,
        color_jitter_kwargs=None,
        affine_kwargs=None,
        grayscale_kwargs=None,
        noise_kwargs=None,
        erasing_kwargs=None,
        low_res_kwargs=None,
        crop_kwargs=None,
    ):
        super().__init__()

        self.enable = enable
        self.use_ada = use_ada

        default_probabilities = {
            "color_jitter": .2,
            "affine": .2,
            "grayscale": .05,
            "noise": .2,
            "erasing": .2,
            "low_res": .2,
            "crop": .2,
        }
        if self.use_ada:
            default_probabilities.update({
                "affine": 0.,
                "grayscale": 0.,
                "noise": 0.,
                "erasing": 0.,
            })
        probabilities = {**default_probabilities, **(probabilities or {})}

        color_jitter_kwargs = {
            "brightness": .5 if self.use_ada else .6,
            "contrast": .5 if self.use_ada else .4,
            "saturation": .5 if self.use_ada else .4,
            "hue": .0 if self.use_ada else .01,
            **(color_jitter_kwargs or {}),
        }
        affine_kwargs = {
            "degrees": (-5, 5),
            "translate": (.05, .05),
            "scale": (.98, 1.02),
            "fill": -1,
            **(affine_kwargs or {}),
        }
        grayscale_kwargs = {
            "num_output_channels": 3,
            **(grayscale_kwargs or {}),
        }
        noise_kwargs = noise_kwargs or {}
        erasing_kwargs = {
            "p": 1.,
            "scale": (0.01, 0.1),
            "value": -1,
            **(erasing_kwargs or {}),
        }
        low_res_kwargs = low_res_kwargs or {}
        crop_kwargs = crop_kwargs or {}

        self.color_jitter = T.Compose(
            [T.Lambda(lambda img: (img + 1.) / 2.),
             T.ColorJitter(
                **color_jitter_kwargs
              ),
            T.Normalize([.5, .5, .5], [.5, .5, .5])]
        ) 
        self.affine_transform = T.RandomAffine(**affine_kwargs)
        self.grayscale_transform = T.Grayscale(**grayscale_kwargs)
        self.noise_transform = AddNoise(**noise_kwargs)
        self.erasing_transform = T.RandomErasing(**erasing_kwargs)
        self.low_res_transform = LowRes(self.use_ada, **low_res_kwargs)
        self.crop_transform = Crop(self.use_ada, **crop_kwargs)
        self.all_transforms = (
            self.color_jitter,
            self.affine_transform,
            self.grayscale_transform,
            self.noise_transform,
            self.erasing_transform,
            self.low_res_transform,
            self.crop_transform,
        )
        self.probabilities = (
            probabilities["color_jitter"],
            probabilities["affine"],
            probabilities["grayscale"],
            probabilities["noise"],
            probabilities["erasing"],
            probabilities["low_res"],
            probabilities["crop"],
        )


    def forward(self, img):

        if not self.enable:
            return img

        rng_ = torch.rand((len(self.all_transforms),))
        for trans, prob, pprob in zip(self.all_transforms, self.probabilities, rng_):
            if pprob < prob:
                img = trans(img)

        return img
