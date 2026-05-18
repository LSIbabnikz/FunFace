

import argparse
import glob
import pickle
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from accelerate.utils import tqdm
from torch.utils.data import DataLoader, Dataset
from utils.utils import (
    build_feature_extractor as _build_feature_extractor,
    build_input_transform as _build_input_transform,
    configure_runtime as _configure_runtime,
    load_config as _load_config,
    resolve_device as _resolve_device,
    resolve_path as _resolve_path,
    resolve_precision as _resolve_precision,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "configs" / "inference_default.yaml"
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _load_checkpoint(weights_path):

    try:
        return torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location="cpu")


def _extract_feature_extractor_state_dict(checkpoint):

    if isinstance(checkpoint, dict):
        for key in ("feature_extractor", "backbone", "model", "state_dict", "model_state_dict"):
            nested_checkpoint = checkpoint.get(key)
            if isinstance(nested_checkpoint, dict):
                checkpoint = nested_checkpoint
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("The loaded checkpoint does not contain a usable state dict.")

    state_dict = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue

        normalized_key = key
        for prefix in ("module.", "feature_extractor.", "backbone.", "model."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        state_dict[normalized_key] = value

    if not state_dict:
        raise ValueError("No tensor weights were found in the loaded checkpoint.")

    return state_dict


def _load_feature_extractor_weights(feature_extractor, cfg):

    weights_path = _resolve_path(cfg["weights_path"], must_exist=True)
    checkpoint = _load_checkpoint(weights_path)
    state_dict = _extract_feature_extractor_state_dict(checkpoint)

    model_state_keys = set(feature_extractor.state_dict().keys())
    matched_state_dict = {key: value for key, value in state_dict.items() if key in model_state_keys}
    if not matched_state_dict:
        raise RuntimeError(f"No feature extractor weights from '{weights_path}' matched the model definition.")

    incompatible_keys = feature_extractor.load_state_dict(matched_state_dict, strict=False)
    if incompatible_keys.missing_keys:
        raise RuntimeError(
            "Missing feature extractor weights after loading checkpoint: "
            f"{incompatible_keys.missing_keys}"
        )

    return weights_path


def _resolve_input_images(cfg):

    raw_inputs = cfg["input_images"]
    if isinstance(raw_inputs, (str, Path)):
        raw_inputs = [raw_inputs]

    recursive = bool(cfg.get("recursive", True))
    image_extensions = tuple(
        extension.lower() for extension in cfg.get("image_extensions", DEFAULT_IMAGE_EXTENSIONS)
    )

    resolved_images = []
    for raw_input in raw_inputs:
        raw_input = str(raw_input)

        if any(token in raw_input for token in "*?[]"):
            glob_matches = glob.glob(raw_input, recursive=recursive)
            if not Path(raw_input).is_absolute():
                glob_matches.extend(glob.glob(str(REPO_ROOT / raw_input), recursive=recursive))
            candidate_paths = [Path(path) for path in glob_matches]
        else:
            candidate_path = _resolve_path(raw_input, must_exist=True)
            if candidate_path.is_dir():
                iterator = candidate_path.rglob("*") if recursive else candidate_path.glob("*")
                candidate_paths = [path for path in iterator if path.is_file()]
            else:
                candidate_paths = [candidate_path]

        for candidate_path in candidate_paths:
            if candidate_path.is_file() and candidate_path.suffix.lower() in image_extensions:
                resolved_images.append(str(candidate_path.resolve()))

    resolved_images = sorted(dict.fromkeys(resolved_images))
    if not resolved_images:
        raise FileNotFoundError("No inference images were found using the configured input_images values.")

    return resolved_images


class InferenceImageDataset(Dataset):

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return image_path, self.transform(image)


def _build_dataloader(cfg, dataset, device):

    workers = int(cfg.get("workers", 8))
    dataloader_kwargs = {
        "batch_size": int(cfg.get("batch_size", 256)),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        dataloader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

    return DataLoader(dataset, **dataloader_kwargs)


def _save_embeddings(embedding_dict, cfg):

    output_pickle = _resolve_path(cfg["output_pickle"])
    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with output_pickle.open("wb") as handle:
        pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_pickle


def _run_inference(feature_extractor, dataloader, device, precision_dtype, flip_test, verbose):

    embeddings = {}

    feature_extractor.eval()
    with torch.inference_mode():
        dataloader_iter = tqdm(dataloader, disable=not verbose)
        for image_paths, image_batch in dataloader_iter:
            image_batch = image_batch.to(device, non_blocking=device.type == "cuda")

            autocast_context = (
                torch.autocast(device_type=device.type, dtype=precision_dtype)
                if precision_dtype is not None
                else nullcontext()
            )
            with autocast_context:
                features = feature_extractor(image_batch)
                if flip_test:
                    flipped_features = feature_extractor(image_batch.flip(-1))
                    features.add_(flipped_features).mul_(0.5)

            feature_batch = features.detach().cpu().numpy()
            for image_path, feature in zip(image_paths, feature_batch):
                embeddings[image_path] = feature

    return embeddings


def inference_main(cfg):

    # Resolve runtime state before constructing heavy objects.
    device = _resolve_device(cfg)
    precision_dtype, precision_name = _resolve_precision(cfg, device)
    _configure_runtime(device)

    print(f" \t => Using device: {device}")
    print(f" \t => Using precision: {precision_name}")

    # Construct all inference inputs.
    input_transform = _build_input_transform(cfg)
    image_paths = _resolve_input_images(cfg)
    dataset = InferenceImageDataset(image_paths, input_transform)
    dataloader = _build_dataloader(cfg, dataset, device)
    print(f" \t => Prepared dataloader for {len(dataset)} images")

    # Build the feature extractor and restore inference weights.
    feature_extractor = _build_feature_extractor(cfg)
    weights_path = _load_feature_extractor_weights(feature_extractor, cfg)
    feature_extractor = feature_extractor.to(device)
    print(f" \t => Loaded feature extractor weights from: {weights_path}")

    # Run batched recognition inference and persist the extracted embeddings.
    embeddings = _run_inference(
        feature_extractor,
        dataloader,
        device,
        precision_dtype,
        bool(cfg.get("flip_test", True)),
        bool(cfg.get("verbose", True)),
    )
    output_pickle = _save_embeddings(embeddings, cfg)
    print(f" \t => Saved {len(embeddings)} embeddings to: {output_pickle}")


def parse_args():

    parser = argparse.ArgumentParser(description="Run recognition inference and save image embeddings.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the inference yaml config.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    inference_main(_load_config(
        args.config,
        required_keys=("feature_extractor", "weights_path", "input_images", "output_pickle"),
    ))
