#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to predict robotic arm states using a fine-tuned InstructPix2Pix model."""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from datasets import load_dataset, Features
from datasets.features import Value

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Predict robotic arm states using a fine-tuned InstructPix2Pix model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory (e.g., ./robot_arm_model_scene1).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./sample_data/validation",
        help="Directory containing validation data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./predictions",
        help="Directory to save predicted images.",
    )
    parser.add_argument(
        "--scene_id",
        type=int,
        default=1,
        help="Scene ID for prediction (e.g., 1 for scene1).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="预测50帧后的机械臂状态",
        help="Prompt for the prediction.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Size to resize input images.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for prediction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda or cpu).",
    )
    return parser.parse_args()

def load_image(image_path, image_size):
    """加载并预处理图像"""
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((image_size, image_size))
        print(f"成功加载图像: {image_path}, 大小: {image.size}, 模式: {image.mode}")
        return image
    except Exception as e:
        print(f"加载图像失败: {image_path}, 错误: {str(e)}")
        return None

def is_black_image(image):
    """检查图像是否全黑"""
    if image is None:
        return True
    img_array = np.array(image)
    return np.all(img_array == 0)

def main():
    """主函数：执行预测"""
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载微调模型
    print(f"加载模型: {args.model_path}")
    try:
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        ).to(args.device)
    except Exception as e:
        print(f"加载模型失败: {args.model_path}, 错误: {str(e)}")
        return

    # 加载验证数据集
    validation_data_dir = Path(args.data_dir) / f"scene{args.scene_id}"
    validation_metadata_path = validation_data_dir / f"metadata_scene{args.scene_id}_validation.jsonl"
    print(f"加载验证数据: {validation_metadata_path}")

    features = Features({
        "image": Value("string"),
        "edited_image": Value("string"),
        "edit_prompt": Value("string"),
    })

    try:
        dataset = load_dataset(
            "json",
            data_files=str(validation_metadata_path),
            features=features
        )["train"]
        dataset = dataset.select_columns(["image", "edit_prompt"])
        print(f"数据集加载成功，样本数: {len(dataset)}")
    except Exception as e:
        print(f"加载数据集失败: {validation_metadata_path}, 错误: {str(e)}")
        return

    # 预测
    for idx, example in enumerate(dataset):
        image_path = os.path.join(validation_data_dir, example["image"])
        prompt = example["edit_prompt"]
        print(f"处理样本 {idx}: 图像: {image_path}, 提示: {prompt}")

        # 加载输入图像
        input_image = load_image(image_path, args.image_size)
        if input_image is None:
            print(f"跳过样本 {idx}: 无法加载图像")
            continue

        # 生成预测图像
        try:
            with torch.no_grad():
                predicted_image = pipeline(
                    prompt=prompt,
                    image=input_image,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    image_guidance_scale=1.5
                ).images[0]
        except Exception as e:
            print(f"预测失败 样本 {idx}: 错误: {str(e)}")
            continue

        # 检查预测图像是否全黑
        if is_black_image(predicted_image):
            print(f"警告: 样本 {idx} 的预测图像全黑")
        else:
            print(f"样本 {idx} 预测图像正常")

        # 保存预测图像
        output_path = os.path.join(args.output_dir, f"predicted_scene{args.scene_id}_{idx}.png")
        predicted_image.save(output_path)
        print(f"预测图像已保存: {output_path}")

if __name__ == "__main__":
    main()