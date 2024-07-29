#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm

import train_utils
from train_utils import (
    collate_fn,
    init_train_basics,
    log_validation,
    save_model,
    unwrap_model,
    load_models,
    get_optimizer,
    more_init,
    resume_model,
    get_model,
    enable_gradient_checkpointing
)
from types import SimpleNamespace
from types import MethodType

import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam
from tqdm import tqdm
import math
import torchvision.transforms as transforms
from diffusers import DDIMScheduler
import wandb
from PIL import Image



default_arguments = dict(
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    dataset_path="./data/combined.parquet",
    num_validation_images=4,
    output_dir="pets-base",
    seed=124,
    resolution=512,
    train_batch_size=8,
    max_train_steps=100_000,
    validation_steps=250,
    checkpointing_steps=500,
    resume_from_checkpoint="/home/ubuntu/NoveltyGAN/pets-base/checkpoint-50000_generator.pt",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    learning_rate=6.0e-5,
    lr_scheduler="linear",
    lr_warmup_steps=500,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=8,
    use_8bit_adam=False,
    adam_beta1=0.9,
    adam_beta2=0.99,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    report_to="wandb",
    mixed_precision="bf16",
    allow_tf32=True,
    logging_dir="logs",
    local_rank=-1,
    num_processes=1,
    use_wandb=True,
    gan_loss_type="hinge"
)



@torch.no_grad()
def do_validation(generator, scheduler, steps=50, batch_size=64):
    lats = torch.randn(batch_size, 3, 64, 64, device=generator.device)
    scheduler.set_timesteps(steps)

    for i, t in enumerate(scheduler.timesteps):
        noise_pred = generator(lats, t, return_dict=False)[0]
        lats = scheduler.step(noise_pred, t, lats, return_dict=False)[0]

    lats = (lats * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255
    images = [Image.fromarray(l.astype("uint8")) for l in lats]

    # wandb
    wandb.log({"validation": [wandb.Image(i) for i in images]}, commit=False)
    return images


@torch.no_grad()
def prepare_batch(batch, noise_scheduler, weight_dtype):
    pixel_values = batch[0].to(dtype=weight_dtype) * 2 - 1
    noise = torch.randn_like(pixel_values)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (noise.shape[0],), device=pixel_values.device
    ).long()
    noisy_latents = noise_scheduler.add_noise(pixel_values, noise, timesteps)

    return pixel_values, noisy_latents, timesteps, noise


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator, weight_dtype = init_train_basics(args, logger)

    # Load scheduler and models
    # noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler = DDIMScheduler(beta_schedule="linear", beta_start=0.0001, beta_end=0.02)
    generator = get_model().to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        enable_gradient_checkpointing(generator)

    # Optimizer creation
    optimizer, lr_scheduler = get_optimizer(args, args.learning_rate, list(generator.parameters()), accelerator)

    transform = transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])
    dataset = torchvision.datasets.OxfordIIITPet(root='./datasets/OxfordIIITPet/', split='trainval', download=True, transform=transform)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=len(dataset))
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=8, drop_last=True, sampler=sampler)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    # Prepare everything with our `accelerator`.
    generator, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        generator, optimizer, lr_scheduler, train_dataloader
    )

    global_step = -1
    if args.resume_from_checkpoint:
        global_step = resume_model(generator, args.resume_from_checkpoint, accelerator)

    global_step, first_epoch, progress_bar = more_init(accelerator, args, train_dataloader, 
                                                        dataset, logger, num_update_steps_per_epoch, global_step, wandb_name="noveltygan")


    grad_norm = 0
    logs = {}
    for epoch in range(first_epoch, args.num_train_epochs):
        generator.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(generator):
                with torch.no_grad():
                    pixel_values, noisy_latents, timesteps, noise = prepare_batch(batch, noise_scheduler, weight_dtype)
                noise_pred = generator(noisy_latents, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(generator.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm.item(),
                }

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 and global_step > 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        generator_state_dict = unwrap_model(accelerator, generator).state_dict()
                        torch.save(generator_state_dict, save_path + "_generator.pt")
                        logger.info(f"Saved state to {save_path}")

                        
            progress_bar.set_postfix(**logs)
            if args.use_wandb:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step % args.validation_steps == 0 and global_step > 0:
                    images = do_validation(generator, noise_scheduler, steps=50, batch_size=32)

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        generator_state_dict = unwrap_model(accelerator, generator).state_dict()
        torch.save(generator_state_dict, save_path + "_generator.pt")
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    arguments = {k: v for k, v in default_arguments.items()}
    train(arguments)







