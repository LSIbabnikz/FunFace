import io
from pathlib import Path

import torch
from PIL import Image

import webdataset as wds

DEFAULT_VALIDATION_URLS = {
    "lfw": "https://huggingface.co/datasets/LSIbabnikz/lfw_bin/resolve/main/shard-{000000..000002}.tar",
    "cplfw": "https://huggingface.co/datasets/LSIbabnikz/cplfw_bin/resolve/main/shard-{000000..000002}.tar",
    "xqlfw": "https://huggingface.co/datasets/LSIbabnikz/xqlfw_bin/resolve/main/shard-{000000..000002}.tar",
}

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VALIDATION_PATHS = {
    dataset_name: str(REPO_ROOT / "ignore_for_git" / "webdatasets" / dataset_name)
    for dataset_name in DEFAULT_VALIDATION_URLS
}


def _decode_validation_pair_sample(sample):
    field_pairs = (("img1", "img2"), ("jpg", "png"))
    image1_field, image2_field = None, None
    for candidate1, candidate2 in field_pairs:
        if candidate1 in sample and candidate2 in sample:
            image1_field, image2_field = candidate1, candidate2
            break
    if image1_field is None or image2_field is None:
        raise KeyError("Validation pair sample is missing the two image payloads")

    label = sample.get("cls")
    if label is None:
        split_name = str(sample.get("__key__", "")).split("/", 1)[0]
        label = 1 if split_name == "genuine" else 0
    elif isinstance(label, bytes):
        label = int(label.decode("utf-8"))
    else:
        label = int(label)

    image1 = Image.open(io.BytesIO(sample[image1_field])).convert("RGB")
    image2 = Image.open(io.BytesIO(sample[image2_field])).convert("RGB")
    return image1, image2, label


def _resolve_validation_source(source):
    if isinstance(source, (list, tuple)):
        return [str(item) for item in source]

    source = str(source)
    if "://" in source or "{" in source:
        return source

    source_path = Path(source)
    if not source_path.is_absolute():
        repo_source_path = REPO_ROOT / source_path
        if repo_source_path.exists():
            source_path = repo_source_path

    if source_path.is_dir():
        shard_paths = sorted(str(path) for path in source_path.glob("*.tar*"))
        if not shard_paths:
            raise FileNotFoundError(f"No validation shards found under '{source}'")
        return shard_paths

    return str(source_path) if source_path.exists() else source


def get_dataset_info(dataset_name):

    if dataset_name == "ms1mv3":
        total_samples = 5179510 
        max_cls = 93431
        urls = "https://huggingface.co/datasets/gaunernst/ms1mv3-wds/resolve/main/ms1mv3-{0000..0099}.tar"
    elif dataset_name == "ms1mv2":
        total_samples = 5822653 
        max_cls = 85742
        urls = "https://huggingface.co/datasets/LSIbabnikz/ms1mv2_wds/resolve/main/shard-{000000..000116}.tar"
    elif dataset_name == "webface4m":
        total_samples = 4235242
        max_cls = 205990
        urls = "https://huggingface.co/datasets/gaunernst/webface4m-wds-gz/resolve/main/webface4m-{0000..0120}.tar.gz"
    else:
        raise NotImplemented

    return total_samples, max_cls, urls


def construct_training_dataloader(urls, trans, wds_epoch, cfg):

    webdataset = (
        wds.WebDataset(
            urls,
            resampled=True,
            shardshuffle=False,
            detshuffle=True,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker
        )
        .shuffle(cfg.get('batch_size', 512) * 10, initial=cfg.get('batch_size', 512) * 2)
        .decode('pilrgb')
        .to_tuple("jpg", "cls")
        .map_tuple(
            trans, 
            lambda cls: torch.tensor(cls, dtype=torch.long)
            )
        .batched(cfg["batch_size"], partial=False)
    )

    train_dataloader = (
        wds.WebLoader(
            webdataset,
            batch_size=None, 
            shuffle=False,      
            num_workers=cfg.get('workers', 12),
            pin_memory=True,
            timeout=120,               
            prefetch_factor=2,         
        )
        .unbatched()
        .shuffle(cfg.get('batch_size', 512) * 10)
        .batched(cfg.get('batch_size', 512))
        .with_epoch(wds_epoch)
    )

    return train_dataloader


def construct_validation_pipeline(val_input_trans, cfg):
    def make_validation_datapipeline(urls):
        return wds.DataPipeline(
            wds.SimpleShardList(urls),
            wds.tarfile_to_samples(),
            wds.non_empty,
            wds.map(_decode_validation_pair_sample),
            wds.map_tuple(
                val_input_trans,
                val_input_trans,
                lambda cls: torch.tensor(cls, dtype=torch.long),
            ),
            wds.batched(cfg.get("batch_size", 512), partial=True),
        )

    validation_sources = DEFAULT_VALIDATION_PATHS.copy()
    validation_sources.update(cfg.get("validation_urls", {}))
    validation_sources.update(cfg.get("validation_paths", {}))

    validation_datasets = {}
    for dataset_name in cfg.get("validation_set", list(validation_sources)):
        if dataset_name not in validation_sources:
            raise KeyError(f"Missing validation source for dataset '{dataset_name}'")
        validation_datasets[dataset_name] = make_validation_datapipeline(
            _resolve_validation_source(validation_sources[dataset_name])
        )

    return validation_datasets
