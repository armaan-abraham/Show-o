# coding=utf-8
# Copyright 2024 HuggingFace, NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.data import Text2ImageDataset
from training.imagenet_dataset import ImageNetDataset
from training.c4_dataset import C4Dataset
# from parquet import RefinedWebDataset

from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, \
    create_attention_mask_for_mmu
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from llava.llava_data_vq_unified import get_instruct_data_loader

SYSTEM_PROMPT_LEN = 28

from training.utils import get_config, flatten_omega_conf, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = (config.training.batch_size_t2i
                                + config.training.batch_size_lm
                                + config.training.batch_size_mmu)
    total_batch_size = (
            (config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu)
            * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    assert config.training.seed is not None
    set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    print('special tokens : \n', uni_prompting.sptids_dict)

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize Show-o model
    model = Showo(**config.model.showo, accelerator=accelerator).to(accelerator.device)
    mask_id = model.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_t2i_without_accum = config.training.batch_size_t2i * accelerator.num_processes
    total_batch_size_t2i = (
            config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params


    assert config.dataset.gen_type == "imagenet1k"
    dataset_imagenet = ImageNetDataset(
        dataset_config.train_t2i_shards_path_or_url,
        image_size=preproc_config.resolution,
    )

    if accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset_imagenet,
                                    num_replicas=accelerator.num_processes,
                                    rank=accelerator.process_index,
                                    shuffle=True,
                                    )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_dataloader_t2i = DataLoader(dataset_imagenet, batch_size=config.training.batch_size_t2i,
                                    sampler=sampler, collate_fn=dataset_imagenet.collate_fn,
                                    shuffle=shuffle, num_workers=dataset_config.num_workers)
    num_update_steps_per_epoch = math.ceil(len(dataset_imagenet) / total_batch_size_t2i)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Data for image captioning
    assert config.dataset.und_type == "captioning"
    dataset_mmu = Text2ImageDataset(
        train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
        tokenizer=None,  # we want to get raw texts
        max_seq_length=preproc_config.max_seq_length,
        per_gpu_batch_size=config.training.batch_size_mmu,
        num_workers=dataset_config.num_workers,
        seed=config.training.seed,
        resolution=preproc_config.resolution,
        shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        pin_memory=dataset_config.pin_memory,
        persistent_workers=dataset_config.persistent_workers,
        is_captioning=True,
        add_caption_prompt=dataset_config.add_caption_prompt,
    )
    train_dataloader_mmu = dataset_mmu.train_dataloader

    # Assert that a path is provided for language modeling data
    assert dataset_config.train_lm_shards_path_or_url, "train_lm_shards_path_or_url must be provided"

    dataset_lm = C4Dataset(dataset_config.train_lm_shards_path_or_url)
    train_dataloader_lm = torch.utils.data.DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        collate_fn=dataset_lm.collate_fn,
        num_workers=dataset_config.num_workers
    )

    # Combine these dataloaders into a single iterable model
    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "lm_flow": train_dataloader_lm,
        "mmu_flow": train_dataloader_mmu,
    }

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)


    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)

            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.print(f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin")
            state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            del state_dict

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model.to(device=accelerator.device)

    # TODO: move mask
    mask_dtype = torch.bfloat16

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch, batch_idx, dataloader_idx in combined_dataloader:

            # Debug logging - save images and text info for first iteration on main process
            if global_step == 0 and accelerator.is_main_process:
                save_debug_batch_info(batch, config, config.experiment.output_dir)

            # for loss calculation
            batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
            batch_size_lm = len(batch["lm_flow"]["input_ids"])
            batch_size_mmu = batch["mmu_flow"]["images"].shape[0]

            with torch.no_grad():
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                # Build formatted sequences for class-conditional/text-to-image generation
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]
                pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
                data_time_m.update(time.time() - end)

                # Encode images to image tokens and create input and labels
                image_tokens = vq_model.get_code(pixel_values)
                image_tokens = image_tokens + len(uni_prompting.text_tokenizer)
                input_ids, masks, labels = uni_prompting((texts, image_tokens, image_tokens), 't2i')

                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                # Build formatted sequences for language modeling
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                texts_lm = batch["lm_flow"]["input_ids"]
                input_ids_lm, _, labels_lm = uni_prompting((texts_lm, input_ids.shape[-1]), 'lm')
                input_ids = torch.cat((input_ids, input_ids_lm.to(input_ids.device)), dim=0)
                labels = torch.cat((labels, labels_lm.to(input_ids.device)), dim=0)

                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                # Build formatted sequences for captioning/multimodal understanding
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                pixel_values_mmu, texts_mmu = batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"]
                pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
                input_ids_mmu, _, labels_mmu = uni_prompting((image_tokens_mmu, texts_mmu), 'mmu')
                input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)
                input_ids = torch.cat((input_ids, input_ids_mmu.to(input_ids.device)), dim=0)
                labels = torch.cat((labels, labels_mmu.to(input_ids.device)), dim=0)

                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                # Assemble attention mask for all flows
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                attention_mask = model.assemble_attention_mask(
                    input_ids=input_ids,
                    batch_size_t2i=batch_size_t2i,
                    batch_size_lm=batch_size_lm,
                    batch_size_mmu=batch_size_mmu,
                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                    mask_dtype=mask_dtype,
                )

            with accelerator.accumulate(model):
                logits, loss_t2i, loss_lm, loss_mmu, mask_prob = model(
                    input_ids=input_ids,
                    input_embeddings=None,
                    attention_mask=attention_mask,
                    labels=labels,
                    label_smoothing=config.training.label_smoothing,
                    batch_size_t2i=batch_size_t2i,
                    batch_size_lm=batch_size_lm,
                    batch_size_mmu=batch_size_mmu,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                    global_step=global_step,
                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                    config=config,
                    mask_schedule=mask_schedule,
                    ignore_id=uni_prompting.ignore_id,
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_t2i = accelerator.gather(loss_t2i.repeat(config.training.batch_size_t2i)).mean()
                avg_loss_lm = accelerator.gather(loss_lm.repeat(config.training.batch_size_lm)).mean()
                avg_loss_mmu = accelerator.gather(loss_mmu.repeat(config.training.batch_size_mmu)).mean()
                loss = config.training.t2i_coeff * loss_t2i + \
                       config.training.lm_coeff * loss_lm + \
                       config.training.mmu_coeff * loss_mmu

                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size_t2i)).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "step_loss_mmu": avg_loss_mmu.item(),
                        "step_loss_lm": avg_loss_lm.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
                        f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    logger.info("Would be generating images now")

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()


def save_debug_batch_info(batch, config, output_dir):
    """Save debug information for the first batch including text and images."""
    # Save debug info to file
    debug_file_path = f"{output_dir}/debug_batch_info.txt"
    with open(debug_file_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("T2I Flow - Text prompt (first 200 chars):\n")
        f.write(f"{batch['t2i_flow']['input_ids'][0][:200]}\n")
        f.write(f"T2I Flow - Image shape: {batch['t2i_flow']['images'][0].shape}\n")

        f.write("\nLM Flow - Text (first 200 chars):\n")
        f.write(f"{batch['lm_flow']['input_ids'][0][:200]}\n")

        f.write("\nMMU Flow - Text prompt (first 200 chars):\n")
        f.write(f"{batch['mmu_flow']['input_ids'][0][:200]}\n")
        f.write(f"MMU Flow - Image shape: {batch['mmu_flow']['images'][0].shape}\n")
        f.write("=" * 80 + "\n")

    # Save T2I flow image
    t2i_image = batch['t2i_flow']['images'][0]
    t2i_image = torch.clamp((t2i_image + 1.0) / 2.0, min=0.0, max=1.0)
    t2i_image *= 255.0
    t2i_image = t2i_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    t2i_pil = Image.fromarray(t2i_image)
    t2i_pil.save(f"{output_dir}/debug_t2i_flow.png")

    # Save MMU flow image
    mmu_image = batch['mmu_flow']['images'][0]
    mmu_image = torch.clamp((mmu_image + 1.0) / 2.0, min=0.0, max=1.0)
    mmu_image *= 255.0
    mmu_image = mmu_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    mmu_pil = Image.fromarray(mmu_image)
    mmu_pil.save(f"{output_dir}/debug_mmu_flow.png")

    logger.info(f"Saved debug info and images to {output_dir}")

def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
