#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂状态预测脚本
功能：加载微调后的模型预测50帧后的机械臂状态
"""

import os
import torch
import argparse
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline


class RobotArmPredictor:
    def __init__(self, model_dir, device=None):
        """初始化预测器"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = self._load_model(model_dir)

    def _load_model(self, model_dir):
        """加载微调后的模型"""
        print(f"[系统] 正在加载模型从: {model_dir}")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32,
            safety_checker=None
        ).to(self.device)
        print("[系统] 模型加载完成")
        return pipe

    def predict(self, input_image, prompt, **kwargs):
        """执行单次预测"""
        return self.pipe(
            prompt,
            image=input_image,
            **kwargs
        ).images[0]


def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./robot_arm_model",
                        help="训练好的模型目录")
    parser.add_argument("--input_dir", default="./sample_data/predict/origin",
                        help="待预测图像目录")
    parser.add_argument("--output_dir", default="./sample_data/predict/predict_50frames",
                        help="预测结果输出目录")
    parser.add_argument("--prompt", default="预测50帧后的机械臂状态",
                        help="预测指令文本")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    # 初始化预测器
    predictor = RobotArmPredictor(args.model_dir)

    # 预测参数配置
    predict_kwargs = {
        "num_inference_steps": 50,
        "image_guidance_scale": 1.3,
        "guidance_scale": 6.5,
        "generator": torch.Generator(predictor.device).manual_seed(args.seed)
    }

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理所有输入图像
    print(f"\n[任务] 开始处理 {args.input_dir} 中的图像")
    for img_name in sorted(os.listdir(args.input_dir)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(args.input_dir, img_name)
            output_path = os.path.join(args.output_dir, img_name)

            print(f"  → 正在预测: {img_name}", end="", flush=True)

            try:
                # 执行预测
                input_image = Image.open(input_path).convert("RGB")
                result = predictor.predict(input_image, args.prompt, **predict_kwargs)

                # 保存结果
                result.save(output_path)
                print(" - 预测成功 ✅")

            except Exception as e:
                print(f" - 预测失败 ❌ (错误: {str(e)})")

    print(f"\n[结果] 所有预测已完成，保存至: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()