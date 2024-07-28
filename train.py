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
    default_arguments,
    load_models,
    get_optimizer,
    get_dataset,
    more_init,
    resume_model,
)
from types import SimpleNamespace
from types import MethodType


def get_discriminator_loss(real_predictions, fake_predictions, gan_loss_type="hinge"):
    if gan_loss_type == "hinge":
        discrim_loss = (F.relu(1.0 - real_predictions).mean() + F.relu(1.0 + fake_predictions).mean())
    elif gan_loss_type == "relative-hinge":
        real_rel = real_predictions - fake_predictions.mean(dim=0, keepdim=True)
        fake_rel = fake_predictions - real_predictions.mean(dim=0, keepdim=True)
        discrim_loss = F.relu(1 - real_rel).mean(0) + F.relu(1 + fake_rel).mean(0)
        discrim_loss = discrim_loss.mean()
    elif gan_loss_type == "relative":
        real_rel = real_predictions - fake_predictions.mean(dim=0, keepdim=True)
        fake_rel = fake_predictions - real_predictions.mean(dim=0, keepdim=True)
        discrim_loss = -F.logsigmoid(real_rel).mean() - F.logsigmoid(1 - fake_rel).mean()
    else:
        discrim_loss = (F.binary_cross_entropy_with_logits(real_predictions, torch.ones_like(real_predictions)) \
                        + F.binary_cross_entropy_with_logits(fake_predictions, torch.zeros_like(fake_predictions)))

    return discrim_loss


def get_generator_loss(fake_predictions, real_predictions, gan_loss_type="hinge"):
    if gan_loss_type == "hinge":
        gen_loss = (-fake_predictions).mean()
    elif gan_loss_type == "relative-hinge":
        real_rel = real_predictions - fake_predictions.mean(dim=0, keepdim=True)
        fake_rel = fake_predictions - real_predictions.mean(dim=0, keepdim=True)
        gen_loss = F.relu(1 + real_rel).mean(0) + F.relu(1 - fake_rel).mean(0)
        gen_loss = gen_loss.mean()
    elif gan_loss_type == "relative":
        real_rel = real_predictions - fake_predictions.mean(dim=0, keepdim=True)
        fake_rel = fake_predictions - real_predictions.mean(dim=0, keepdim=True)
        gen_loss = -F.logsigmoid(1 - real_rel).mean() - F.logsigmoid(fake_rel).mean()
    else:
        gen_loss = F.binary_cross_entropy_with_logits(fake_predictions, torch.ones_like(fake_predictions))

    return gen_loss


@torch.no_grad()
def prepare_batch(batch, tokenizer, text_encoder, noise_scheduler, vae, weight_dtype, ):
    # # Get the text embedding for conditioning
    input_ids = tokenizer(batch["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids
    encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device),return_dict=False,)[0]

    # 
    clean_latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
    timesteps_clean = torch.zeros(clean_latents.shape[0], device=clean_latents.device).long()
    noise = torch.randn_like(clean_latents)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (noise.shape[0],), device=clean_latents.device
    ).long()
    noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)


    return clean_latents, noisy_latents, timesteps, timesteps_clean, encoder_hidden_states

def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    args.validation_prompt = [f"majestic fantasy painting", f"a comic book drawing", f"HD cinematic photo", f"oil painting"]
    accelerator, weight_dtype = init_train_basics(args, logger)

    tokenizer, noise_scheduler, text_encoder, vae, generator, discriminator, density_model = load_models(args, accelerator, weight_dtype)

    # Optimizer creation
    optimizer_gen, lr_scheduler_gen = get_optimizer(args, args.learning_rate_gen, list(generator.parameters()), accelerator)
    optimizer_disc, lr_scheduler_disc = get_optimizer(args, args.learning_rate_disc, list(discriminator.parameters()), accelerator)
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args, tokenizer)

    # Prepare everything with our `accelerator`.
    generator, discriminator, text_encoder, optimizer_gen, lr_scheduler_gen, optimizer_disc, lr_scheduler_disc, train_dataloader = accelerator.prepare(
        generator, discriminator, text_encoder, optimizer_gen, lr_scheduler_gen, optimizer_disc, lr_scheduler_disc, train_dataloader
    )

    global_step = -1
    if args.resume_from_checkpoint:
        global_step = resume_model(generator, args.resume_from_checkpoint, accelerator)
        resume_model(discriminator, args.resume_from_checkpoint, accelerator)

    global_step, first_epoch, progress_bar = more_init(accelerator, args, train_dataloader, 
                                                        train_dataset, logger, num_update_steps_per_epoch, global_step, wandb_name="noveltygan")


    grad_norm = 0
    logs = {}
    for epoch in range(first_epoch, args.num_train_epochs):
        generator.train()
        discriminator.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                clean_latents, noisy_latents, timesteps, timesteps_clean, encoder_hidden_states = prepare_batch(batch, tokenizer, text_encoder, noise_scheduler, vae, weight_dtype)
                noises = torch.randn_like(clean_latents)


            if step % 2 == 0:
                with accelerator.accumulate(discriminator):
                    with torch.no_grad():
                        # Predict the noise residual
                        pred_latents = generator(
                            noises,
                            timesteps_clean,
                            encoder_hidden_states,
                            return_dict=False,
                        )[0]

                    # Discriminator predictions
                    extra_kwargs = {"encoder_hidden_states": encoder_hidden_states}
                    real_preds = discriminator(clean_latents, timesteps_clean, extra_kwargs)
                    fake_preds = discriminator(pred_latents, timesteps_clean, extra_kwargs)

                    # Discriminator loss
                    disc_loss = get_discriminator_loss(real_preds, fake_preds, args.gan_loss_type)
                    loss = disc_loss

                    accelerator.backward(disc_loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=True)

                    logs["disc_loss"] = disc_loss.detach().item()
                    logs["lr"] = lr_scheduler_disc.get_last_lr()[0]
                    logs["disc_grad_norm"] = grad_norm

            else:
                with accelerator.accumulate(generator):
                    noises = torch.randn_like(clean_latents)

                    # Generator predictions
                    pred_latents = generator(
                        noises,
                        timesteps_clean,
                        encoder_hidden_states,
                        return_dict=False,
                    )[0]

                    # Discriminator predictions
                    fake_preds = discriminator(pred_latents, timesteps_clean, extra_kwargs)

                    # Generator loss
                    gen_loss = get_generator_loss(fake_preds, real_preds, args.gan_loss_type)

                    # density model loss
                    # autocast
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        density_loss = density_model(pred_latents, extra_kwargs)

                    loss = gen_loss + density_loss * args.density_loss_factor

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(generator.parameters(), args.max_grad_norm)
                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad(set_to_none=True)

                    logs["gen_loss"] = gen_loss.detach().item()
                    logs["density_loss"] = density_loss.detach().item()
                    logs["lr"] = lr_scheduler_gen.get_last_lr()[0]
                    logs["gen_grad_norm"] = grad_norm


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 and global_step > 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_model(generator, discriminator, accelerator,save_path, args, logger)
                        
            progress_bar.set_postfix(**logs)
            if args.use_wandb:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if args.validation_prompt is not None and global_step % args.validation_steps == 0 and global_step > 0:
                    images = log_validation(
                        generator,
                        vae,
                        tokenizer,
                        text_encoder,
                        weight_dtype,
                        args,
                        accelerator,
                        epoch=epoch,
                        logger=logger,
                    )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_model(generator, discriminator, accelerator, save_path, args, logger)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)