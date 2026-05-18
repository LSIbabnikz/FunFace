
from pathlib import Path

import torch
from torchvision.transforms import v2 as T
import yaml

import braceexpand

from backbone.model import (
    IR_18,
    IR_34,
    IR_50,
    IR_101,
    IR_152,
    IR_200,
    IR_SE_50,
    IR_SE_101,
    IR_SE_152,
    IR_SE_200,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_BUILDERS = {
    "ir_18": IR_18,
    "ir_34": IR_34,
    "ir_50": IR_50,
    "ir_101": IR_101,
    "ir_152": IR_152,
    "ir_200": IR_200,
    "ir_se_50": IR_SE_50,
    "ir_se_101": IR_SE_101,
    "ir_se_152": IR_SE_152,
    "ir_se_200": IR_SE_200,
}


def resolve_path(path_like, must_exist=False):

    path = Path(path_like).expanduser()
    if not path.is_absolute():
        cwd_path = (Path.cwd() / path).resolve()
        repo_path = (REPO_ROOT / path).resolve()
        if cwd_path.exists():
            path = cwd_path
        elif repo_path.exists() or repo_path.parent.exists():
            path = repo_path
        else:
            path = cwd_path

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Resolved path does not exist: {path}")

    return path


def load_config(config_path, required_keys=()):

    config_path = resolve_path(config_path, must_exist=True)
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}

    missing_keys = [key for key in required_keys if key not in cfg]
    if missing_keys:
        raise KeyError(f"Missing config keys: {missing_keys}")

    return cfg


def configure_runtime(device):

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def resolve_device(cfg):

    requested_device = str(cfg.get("device", "auto")).lower()
    device_id = int(cfg.get("device_id", 0))

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_id}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for inference, but no CUDA device is available.")
        return torch.device(f"cuda:{device_id}")

    if requested_device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for inference, but no CUDA device is available.")
        return torch.device(requested_device)

    if requested_device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested for inference, but it is not available.")
        return torch.device("mps")

    if requested_device == "cpu":
        return torch.device("cpu")

    return torch.device(requested_device)


def resolve_precision(cfg, device):

    requested_precision = str(cfg.get("mixed_precision", "auto")).lower()

    if requested_precision in {"none", "fp32"}:
        return None, "fp32"

    if requested_precision == "auto":
        if device.type == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16, "bf16"
            return torch.float16, "fp16"
        if device.type == "cpu":
            return torch.bfloat16, "bf16"
        return None, "fp32"

    precision_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if requested_precision not in precision_map:
        raise ValueError(f"Unsupported mixed_precision value: {requested_precision}")

    precision_dtype = precision_map[requested_precision]
    if device.type == "cuda":
        if (
            precision_dtype == torch.bfloat16
            and hasattr(torch.cuda, "is_bf16_supported")
            and not torch.cuda.is_bf16_supported()
        ):
            print(" \t => Requested bf16 is unavailable on this CUDA device, falling back to fp16")
            return torch.float16, "fp16"
        return precision_dtype, requested_precision

    if device.type == "cpu" and precision_dtype == torch.bfloat16:
        return precision_dtype, requested_precision

    print(f" \t => Mixed precision is disabled for device type '{device.type}', using fp32 instead")
    return None, "fp32"


def build_input_transform(cfg=None, horizontal_flip_probability=0.0):

    if cfg is None:
        input_size = (112, 112)
    else:
        input_size = tuple(cfg.get("feature_extractor_input_size", [112, 112]))

    transforms = [
        T.Resize(input_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    if horizontal_flip_probability > 0:
        transforms.append(T.RandomHorizontalFlip(horizontal_flip_probability))

    return T.Compose(transforms)


def build_feature_extractor(cfg):

    feature_extractor_name = str(cfg["feature_extractor"]).lower()
    feature_extractor_input_size = tuple(cfg.get("feature_extractor_input_size", [112, 112]))

    if feature_extractor_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported feature extractor '{feature_extractor_name}'")

    feature_extractor = MODEL_BUILDERS[feature_extractor_name](feature_extractor_input_size)
    feature_extractor.eval()
    return feature_extractor


def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay


# #### USELESS BEGIN
# def seed_worker(worker_id: int, cfg_seed: int, rank):
#     worker_seed = torch.initial_seed() % 2**32
#     print(f' (Rank - {rank} with seed {cfg_seed}): Worker {worker_id} setting seed to {worker_seed}')
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
#
#
# def construct_image_trans(cfg=None):
#
#     input_trans = build_input_transform(cfg, horizontal_flip_probability=0.5)
#     val_input_trans = build_input_transform(cfg)
#
#     return input_trans, val_input_trans
# #### USELESS END


def expand_dataset_urls(urls, accelerator, cfg):

    shards = list(braceexpand.braceexpand(urls)) if isinstance(urls, str) else list(urls)

    rank = accelerator.process_index
    world = accelerator.num_processes
    shards = shards[rank::world]

    full_urls = [
        f"pipe:curl --connect-timeout 30 --retry 30 --retry-delay 3 -f -s -L {u}"
        for u in shards
    ]

    print(f" \t => ({accelerator.process_index}) Assigned URLs : {full_urls[:2]}...{full_urls[-2:]}")

    return full_urls
