
import os
import shutil
import argparse
from pathlib import Path
from contextlib import nullcontext
from datetime import timedelta

import torch
from torchvision.transforms import v2 as T
import yaml

import numpy as np
import webdataset as wds
from sklearn import metrics

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import tqdm, ProjectConfiguration, GradScalerKwargs, set_seed, InitProcessGroupKwargs

from loss.adaface import AdaFace
from loss.funface import FunFace

from utils.dataset import *
from utils.augments import *
from utils.scheduler import *
from utils.utils import *


def _resolve_training_setup(cfg):

    # Resolve run-specific state before constructing distributed components.
    cfg_seed = cfg.get("seed", -1)
    if cfg_seed == -1:
        env_seed = os.environ.get("GLOBAL_SEED", None)
        if env_seed is not None:
            seed = int(env_seed)
        else:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    else:
        seed = int(cfg_seed)

    augment = cfg.get('augment', False)

    starting_epoch = cfg.get("start_epoch", 0)
    total_epochs = cfg.get("total_epochs", cfg.get("epochs", 24))
    total_epochs = total_epochs + 2 if augment else total_epochs

    print_aug = 'NA' if not augment else ('AA' if cfg.get("ada_augment", False) else 'EA')
    save_loc = cfg.get("save_loc", ".")
    output_path = f"{save_loc}/{cfg['margin_head']}_{cfg['feature_extractor']}_{cfg['dataset_name']}_{print_aug}"

    checkpoint_name = cfg["checkpoint"]
    if cfg["resume"]:
        if cfg["checkpoint"] == "latest":
            all_checkpoints = os.listdir(os.path.join(output_path, "checkpoints"))
            cp_name, cp_number = sorted([(os.path.join(output_path, "checkpoints", checkpoint), checkpoint.split("_")[-1]) for checkpoint in all_checkpoints], key=lambda x:x[1], reverse=True)[0]
            checkpoint_name = cp_name
            starting_epoch = int(cp_number) + 1 

    return seed, starting_epoch, total_epochs, output_path, checkpoint_name


def _setup_accelerator(cfg, output_path, starting_epoch):

    accelerator : Accelerator = None
    project_iteration = cfg.get("accelerator_iteration", starting_epoch)
    if project_iteration is None:
        project_iteration = starting_epoch
    project_config = ProjectConfiguration(
        output_path, 
        automatic_checkpoint_naming=cfg.get("accelerator_automatic_checkpoint_naming", True),
        total_limit=cfg.get("accelerator_total_limit", 1),
        iteration=project_iteration
        )
    initpg_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=cfg.get("accelerator_timeout_seconds", 1200))
    )
    dataloader_config = DataLoaderConfiguration(
        use_stateful_dataloader=cfg.get("accelerator_use_stateful_dataloader", False),
        split_batches=cfg.get("accelerator_split_batches", False),
        dispatch_batches=cfg.get("accelerator_dispatch_batches", False),
        non_blocking=cfg.get("accelerator_non_blocking", True)
    )
    gradscaler_kwargs = GradScalerKwargs(init_scale=128.)
    accelerator = Accelerator(
        log_with='wandb' if cfg["wandb"] else None,
        mixed_precision=cfg.get("accelerator_mixed_precision", "fp16"),
        project_config=project_config,
        kwargs_handlers=[gradscaler_kwargs, initpg_kwargs],
        dataloader_config=dataloader_config
    )
    if cfg["wandb"]:
        accelerator.init_trackers(
            project_name=cfg["project_name"],
            init_kwargs= {
                "wandb": {
                    "config": cfg,
                    "name": cfg.get("head_type", cfg["margin_head"]),
                    "entity": cfg.get("wandb_entity", "zbabnik"),
                    "group": cfg.get("wandb_group", "cerada"),
                    "resume": "allow",
                    "id": (cfg["wandb_id"] if cfg["checkpoint"] else None),
                }
            }
        )
    accelerator.print(f" \t => Setup accelerator object ")

    return accelerator


def _build_models(cfg, max_cls, accelerator):

    embedding_size = cfg.get("embedding_size", 512)

    feature_extractor: torch.nn.Module = build_feature_extractor(cfg)
    accelerator.print(f" \t => Loaded feature extractor ")

    margin_head : torch.nn.Module
    if cfg["margin_head"] == "adaface":
        margin_head = AdaFace(embedding_size, max_cls)
    elif cfg["margin_head"] == "funface":
        margin_head = FunFace(embedding_size, max_cls)
    else:
        raise NotImplemented
    accelerator.print(f" \t => Using {type(margin_head).__name__} as Classification Head. ")

    return feature_extractor, margin_head


def _build_optimizer_and_scheduler(cfg, feature_extractor, margin_head, wds_epoch, process_count, total_epochs, starting_epoch, accelerator):

    base_lr = cfg.get("base_lr", 0.1)
    weight_decay = cfg.get("weight_decay", 5e-4)
    momentum = cfg.get("momentum", 0.9)
    paras_wo_bn, paras_only_bn = split_parameters(feature_extractor)

    optimizer = torch.optim.SGD([ 
        {'params': paras_wo_bn + list(margin_head.parameters()), 'weight_decay': weight_decay},
        {'params': paras_only_bn}],
        lr=base_lr, momentum=momentum)

    warmup_steps = int(wds_epoch * process_count * cfg.get("warmup", 0.))
    if cfg.get('step_sched', True):
        steps = cfg.get('steps', [12, 20, 24])
        step_scheduler_theta = cfg.get("step_scheduler_theta", 0.1)
        scheduler = StepScheduler(
            optimizer,
            base_lr,
            (total_epochs - starting_epoch) * wds_epoch * process_count, 
            warmup_steps,
            [step * wds_epoch * process_count for step in steps],
            theta=step_scheduler_theta)
    else:
        poly_power = cfg.get('poly_power', 1)
        scheduler = PolyScheduler(
            optimizer, 
            base_lr, 
            (total_epochs - starting_epoch) * wds_epoch * process_count, 
            warmup_steps, 
            poly_power)
    accelerator.print(" \t => Setup Optimizer/Scheduler ")

    accelerator.print(f" \t => Warmup steps: {warmup_steps}")
    accelerator.print(f" \t => Total training steps per process: {(total_epochs - starting_epoch) * wds_epoch * process_count}")

    return optimizer, scheduler


def _run_validation(cfg, accelerator, feature_extractor, dataset_cls_all, epoch):

    epoch_val_batch_trip = False
    feature_extractor.eval()
    validation_set = cfg.get("validation_set", list(dataset_cls_all))
    for val_dataset_name in validation_set:

        print(f" \t => ({accelerator.process_index}) Validation start for {val_dataset_name}")

        val_dataloader = wds.WebLoader(
            dataset_cls_all[val_dataset_name],
            batch_size=None, 
            shuffle=False,     
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

        print(f" \t => ({accelerator.process_index}) Extracting features for {val_dataset_name}")

        validation_scores = []
        true_values = []
        for val_images1, val_images2, is_gen in val_dataloader:

            if not epoch_val_batch_trip: # Debugging
                print(f" \t => ({accelerator.process_index}) Got batch from validation dataloader {epoch}")
                epoch_val_batch_trip = True

            with accelerator.autocast(), torch.no_grad():
                features1 = feature_extractor(
                    val_images1.to(accelerator.device, non_blocking=True)
                ).detach()
                features2 = feature_extractor(
                    val_images2.to(accelerator.device, non_blocking=True)
                ).detach()

            validation_scores.append(torch.nn.functional.cosine_similarity(features1, features2))
            true_values.append(is_gen.to(accelerator.device, non_blocking=True))

        validation_scores = torch.cat(validation_scores, dim=0)
        true_values = torch.cat(true_values, dim=0)

        n_local = torch.tensor([validation_scores.size(0)], device=validation_scores.device)
        ns = accelerator.gather(n_local)
        total_n = int(ns.sum().item())

        scores_padded = accelerator.pad_across_processes(validation_scores, dim=0, pad_index=0)
        labels_padded = accelerator.pad_across_processes(true_values, dim=0, pad_index=0)

        gathered_scores = accelerator.gather(scores_padded)[:total_n].detach().cpu().numpy()
        gathered_labels = accelerator.gather(labels_padded)[:total_n].detach().cpu().numpy().astype(np.int32)

        if accelerator.is_main_process:

            accelerator.print(f" \t => Validation calculation for {val_dataset_name}")

            gathered_scores = np.nan_to_num(gathered_scores, nan=-1.0, posinf=1.0, neginf=-1.0)
            thresholds = np.arange(-1., 1., 0.05).tolist()
            best_acc, best_tau = -1, -1

            for tau in thresholds:

                vs_at_tau = gathered_scores.copy()
                vs_at_tau[vs_at_tau >= tau] = 1
                vs_at_tau[vs_at_tau < tau] = 0

                acc_at_tau = metrics.accuracy_score(gathered_labels, vs_at_tau)

                if acc_at_tau > best_acc:
                    best_acc, best_tau = acc_at_tau, tau

            if cfg["wandb"]:
                accelerator.log({
                    f"{val_dataset_name}_val_acc": best_acc
                    })
        accelerator.wait_for_everyone()


def _save_epoch_state(accelerator, feature_extractor, margin_head, output_path, epoch):

    accelerator.save_state()

    if accelerator.is_main_process:
        u_feature_extractor = accelerator.unwrap_model(feature_extractor)
        u_margin_head = accelerator.unwrap_model(margin_head)

        os.mkdir(os.path.join(output_path, "pth_models", str(epoch)))

        torch.save(u_feature_extractor.state_dict(), os.path.join(output_path, "pth_models", str(epoch), "feature_extractor.pth"))
        torch.save(u_margin_head.state_dict(), os.path.join(output_path, "pth_models", str(epoch), "classification_head.pth"))
        
        if os.path.exists(os.path.join(output_path, "pth_models", str(epoch - 3))):
            shutil.rmtree(os.path.join(output_path, "pth_models", str(epoch - 3)), ignore_errors=True)

    print(f" \t => ({accelerator.process_index}) Saved state for epoch {epoch}")

    accelerator.wait_for_everyone()


TRAIN_CONFIG_REFERENCE_KEYS = (
    "accelerator_config",
    "dataset_config",
    "augmentation_config",
)


def _load_yaml_mapping(config_path):

    config_path = resolve_path(config_path, must_exist=True)
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}

    if not isinstance(cfg, dict):
        raise TypeError(f"Config file must define a mapping: {config_path}")

    return config_path, cfg


def _resolve_train_config_path(parent_config_path, referenced_path):

    referenced_path = Path(referenced_path).expanduser()
    if referenced_path.is_absolute():
        return referenced_path

    return (parent_config_path.parent / referenced_path).resolve()


def _load_train_config(config_path, required_keys=()):

    resolved_config_path, cfg = _load_yaml_mapping(config_path)
    merged_cfg = {}

    # Load referenced section configs first so the main config can still override them.
    for config_key in TRAIN_CONFIG_REFERENCE_KEYS:
        referenced_path = cfg.pop(config_key, None)
        if not referenced_path:
            continue

        _, referenced_cfg = _load_yaml_mapping(
            _resolve_train_config_path(resolved_config_path, referenced_path)
        )
        merged_cfg.update(referenced_cfg)

    merged_cfg.update(cfg)

    missing_keys = [key for key in required_keys if key not in merged_cfg]
    if missing_keys:
        raise KeyError(f"Missing config keys: {missing_keys}")

    return merged_cfg

def train_main(cfg):

    seed, starting_epoch, total_epochs, output_path, checkpoint_name = _resolve_training_setup(cfg)
    accelerator = _setup_accelerator(cfg, output_path, starting_epoch)

    # Seed accelerator instances after the process group is ready.
    set_seed(seed, device_specific=True) 
    print(f" ({accelerator.process_index}) Using seed: {seed}")

    # Cache the distributed world size for dataloader and scheduler calculations.
    process_count = accelerator.num_processes

    input_trans = build_input_transform(cfg, horizontal_flip_probability=0.5)
    val_input_trans = build_input_transform(cfg)
    dataset_name = cfg.get("dataset_name", "webface4m") 
    total_samples, max_cls, urls = get_dataset_info(dataset_name)

    wds_steps = total_samples
    wds_epoch = (wds_steps // (cfg["batch_size"] * process_count)) + 1  # Add one batch per epoch due to integer divison 

    accelerator.print(f" \t => WebDataset Steps : {wds_steps}")
    accelerator.print(f" \t => WebDataset Steps per Epoch : {wds_epoch}")
    accelerator.print(f" \t => Process count : {process_count}")

    feature_extractor, margin_head = _build_models(cfg, max_cls, accelerator)
    optimizer, scheduler = _build_optimizer_and_scheduler(
        cfg,
        feature_extractor,
        margin_head,
        wds_epoch,
        process_count,
        total_epochs,
        starting_epoch,
        accelerator,
    )

    # Define loss function
    loss_ce = torch.nn.CrossEntropyLoss() 

    # Setup augmentations
    augmentor_cfg = cfg["augmentor"]
    augmentations = Augmentor(
        enable=cfg.get("augment", True),
        use_ada=cfg.get("ada_augment", False),
        probabilities=augmentor_cfg["probabilities"],
        color_jitter_kwargs=augmentor_cfg["color_jitter"],
        affine_kwargs=augmentor_cfg["affine"],
        grayscale_kwargs=augmentor_cfg["grayscale"],
        noise_kwargs=augmentor_cfg["noise"],
        erasing_kwargs=augmentor_cfg["erasing"],
        low_res_kwargs=augmentor_cfg["low_res"],
        crop_kwargs=augmentor_cfg["crop"],
    )
    full_trans = T.Compose([
        input_trans,
        augmentations
    ])

    # Construct training dataset
    urls = expand_dataset_urls(urls, accelerator, cfg)
    train_dataloader = construct_training_dataloader(urls, full_trans, wds_epoch, cfg)
    accelerator.print(f" \t => Prepared training dataloader ")

    # Construct validation dataset objects
    dataset_cls_all = construct_validation_pipeline(val_input_trans, cfg)
    accelerator.wait_for_everyone()
    accelerator.print(f" \t => Prepared validation dataloader ")


    # Prepare all the objects using accelerate
    validation_dataset_names = list(dataset_cls_all)
    prepared = accelerator.prepare(
        feature_extractor, margin_head, 
        optimizer, scheduler, 
        train_dataloader, *[dataset_cls_all[dataset_name] for dataset_name in validation_dataset_names]
    )
    feature_extractor, margin_head, optimizer, scheduler, train_dataloader, *prepared_validation_datasets = prepared
    dataset_cls_all = dict(zip(validation_dataset_names, prepared_validation_datasets))
    grad_norm = cfg.get("grad_norm", -1)
    clip_parameters = None if grad_norm == -1 else tuple(feature_extractor.parameters()) + tuple(margin_head.parameters())
    accelerator.print(
        f" Batch size : {cfg['batch_size']} \n \
           Dataset len : {total_samples} \n \
           Dataloader len: {wds_epoch} \n \
           Total samples : {(wds_epoch * accelerator.num_processes * cfg['batch_size'])}"
    )

    # Setup training vairables 
    total_steps = 0 if not cfg["resume"] else starting_epoch * wds_epoch
   
    # Load checkpoint if starting from earlier state
    if cfg["resume"]:
        accelerator.load_state(checkpoint_name)
    f_steps = total_epochs * wds_epoch

    # Sort the output folder
    if accelerator.is_main_process:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(os.path.join(output_path, "pth_models")):
            os.mkdir(os.path.join(output_path, "pth_models"))

    # Sync all processes
    accelerator.wait_for_everyone()

    accelerator.print(f" \t => Starting training ")
    for epoch in range(starting_epoch, total_epochs):

        accelerator.wait_for_everyone()

        print(f" \t => ({accelerator.process_index}) Starting epoch {epoch}")

        # Training loop
        feature_extractor.train()
        margin_head.train()

        with (tqdm(total=wds_epoch) if cfg.get("verbose", False) else nullcontext()) as pbar: 

            # Utility switches for debug prints
            epoch_train_batch_trip = False

            for (image_batch, label_batch) in train_dataloader:

                to_log = {}

                image_batch = image_batch.to(accelerator.device, non_blocking=True)
                label_batch = label_batch.to(accelerator.device, non_blocking=True)

                if not epoch_train_batch_trip: # Debugging
                    print(f" \t => ({accelerator.process_index}) Got batch from dataloader {epoch}")
                    epoch_train_batch_trip = True

                with accelerator.autocast():
                    
                    # Get features 
                    features = feature_extractor(image_batch)

                    # Get logits
                    cls_data = margin_head(features, label_batch)
                    if isinstance(cls_data, tuple):  
                        logit, _, mean_, std_ = cls_data
                        to_log.update({
                            "Head Mean": mean_.item(),
                            "Head Std": std_.item()
                            })
                    else: 
                        logit = cls_data

                    loss = loss_ce(logit, label_batch.squeeze())
                    to_log["Recognition Loss"] = loss.item()
                    
                    to_log["LR"] = optimizer.param_groups[0]['lr']

                accelerator.backward(loss)

                if clip_parameters is not None:
                    accelerator.clip_grad_norm_(clip_parameters, grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_steps += 1

                if cfg.get("wandb", False):
                    accelerator.log(to_log)
                
                if cfg.get("verbose", False):
                    pbar.set_description(f' Loss : {loss.item():.2f}')
                    pbar.update(1)

                #break

        print(f" \t => ({accelerator.process_index}) Ending epoch {epoch} at step: {total_steps}")

        margin_head.eval()
        _run_validation(cfg, accelerator, feature_extractor, dataset_cls_all, epoch)
        _save_epoch_state(accelerator, feature_extractor, margin_head, output_path, epoch)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    cfg = _load_train_config(args.config, required_keys=("margin_head",))
    cfg.setdefault("head_type", cfg["margin_head"])
    return cfg


if __name__ == "__main__":

    cfg = parse_args()
    train_main(cfg)
