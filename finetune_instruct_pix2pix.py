#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to fine-tune InstructPix2Pix for robotic arm frame prediction."""

import argparse
import logging
import math
import os
from pathlib import Path

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Features
from datasets.features import Image as DatasetImage, Value
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

# 设置 Hugging Face 镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 检查 diffusers 最低版本要求
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Fine-tune InstructPix2Pix for robotic arm frame prediction.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="timbrooks/instruct-pix2pix",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="./sample_data/train",
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./robot_arm_model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The integration to report the results and logs to.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="预测50帧后的机械臂状态",
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run fine-tuning validation every X epochs.",
    )
    parser.add_argument(
        "--scene_id",
        type=int,
        default=1,
        help="The scene ID for training.",
    )

    args = parser.parse_args()
    return args

def convert_to_np(image, resolution):
    """将 PIL 图像转换为 numpy 数组并调整大小"""
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def main():
    """主函数：执行模型微调"""
    args = parse_args()

    # 配置日志和加速器
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(total_limit=5, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # 加载调度器、tokenizer 和模型
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # 冻结 VAE 和文本编码器
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 初始化优化器，仅优化 UNet 参数
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # 定义数据集特征
    features = Features({
        "image": Value("string"),
        "edited_image": Value("string"),
        "edit_prompt": Value("string"),
    })

    # 加载数据集
    train_data_dir = Path(args.train_data_dir) / f"scene{args.scene_id}"
    validation_data_dir = Path(args.train_data_dir.replace("train", "validation")) / f"scene{args.scene_id}"

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_data_dir / f"metadata_scene{args.scene_id}.jsonl"),
            "validation": str(validation_data_dir / f"metadata_scene{args.scene_id}_validation.jsonl"),
        },
        features=features,
        cache_dir=args.cache_dir,
    )

    print(f"加载训练数据: {train_data_dir / f'metadata_scene{args.scene_id}.jsonl'}")
    print(f"加载验证数据: {validation_data_dir / f'metadata_scene{args.scene_id}_validation.jsonl'}")

    # 数据预处理函数
    def preprocess_images(examples, data_dir):
        """预处理图像"""
        original_images = []
        for img_path in examples["image"]:
            full_path = os.path.join(data_dir, img_path)
            print(f"尝试加载原始图像: {full_path}")  # 调试信息
            image = Image.open(full_path).convert("RGB")
            original_images.append(convert_to_np(image, args.resolution))
        original_images = np.stack(original_images)

        edited_images = []
        for img_path in examples["edited_image"]:
            full_path = os.path.join(data_dir, img_path)
            print(f"尝试加载编辑图像: {full_path}")  # 调试信息
            image = Image.open(full_path).convert("RGB")
            edited_images.append(convert_to_np(image, args.resolution))
        edited_images = np.stack(edited_images)

        return original_images, edited_images

    def preprocess_train(examples):
        """训练数据预处理"""
        data_dir = str(train_data_dir)
        original_images, edited_images = preprocess_images(examples, data_dir)
        original_images = torch.tensor(original_images, dtype=torch.float32) / 255.0 * 2 - 1
        edited_images = torch.tensor(edited_images, dtype=torch.float32) / 255.0 * 2 - 1
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images
        captions = [caption for caption in examples["edit_prompt"]]
        examples["input_ids"] = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return examples

    def preprocess_validation(examples):
        """验证数据预处理"""
        data_dir = str(validation_data_dir)
        original_images, edited_images = preprocess_images(examples, data_dir)
        original_images = torch.tensor(original_images, dtype=torch.float32) / 255.0 * 2 - 1
        edited_images = torch.tensor(edited_images, dtype=torch.float32) / 255.0 * 2 - 1
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images
        captions = [caption for caption in examples["edit_prompt"]]
        examples["input_ids"] = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)
    validation_dataset = dataset["validation"].with_transform(preprocess_validation)

    def collate_fn(examples):
        """数据批处理函数"""
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }

    # 创建 DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
    )

    # 计算训练步数和调度器
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # 使用加速器准备模型和数据
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
    validation_dataloader = accelerator.prepare(validation_dataloader)

    # 设置混合精度
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # 训练日志
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** 开始训练 *****")
    logger.info(f"  样本数 = {len(train_dataset)}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  每设备批大小 = {args.train_batch_size}")
    logger.info(f"  总训练批大小 = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # 训练循环
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 编码图像到潜在空间
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 添加噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 编码文本
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()

                # 拼接输入
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)
                target = noise

                # 前向传播
                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=step)

            if step >= args.max_train_steps:
                break

        # 验证
        if epoch % args.validation_epochs == 0:
            unet.eval()
            for val_batch in validation_dataloader:
                original_image = val_batch["original_pixel_values"].to(weight_dtype)
                edited_image = val_batch["edited_pixel_values"].to(weight_dtype)
                input_ids = val_batch["input_ids"]

                # 生成预测
                with torch.no_grad():
                    latents = vae.encode(edited_image).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    encoder_hidden_states = text_encoder(input_ids)[0]
                    original_image_embeds = vae.encode(original_image).latent_dist.mode()
                    concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)
                    model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample

                # 可根据需要保存或记录验证结果

    # 保存模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()