#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path
import os

# 设置 Hugging Face 镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def generate_metadata(scene_id, train_ranges, val_original, val_targets):
    """
    为指定场景生成训练和验证元数据，并复制图像到对应目录。

    参数：
    - scene_id: 场景编号（1, 2, 3）
    - train_ranges: 训练集的图像范围列表，例如 [(1, 47), (51, 97)]
    - val_original: 验证集原始图像的编号列表，例如 [48, 49, 50]
    - val_targets: 验证集目标图像的编号列表，例如 [98, 99, 100]
    """
    # 定义目录
    raw_dir = Path("./sample_data/raw")
    train_dir = Path(f"./sample_data/train/scene{scene_id}")
    val_dir = Path(f"./sample_data/validation/scene{scene_id}")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 1. 复制训练图像
    train_imgs = []
    for start, end in train_ranges:
        for i in range(start, end + 1):
            img = f"robo{i}.png"
            shutil.copy(raw_dir / img, train_dir / img)
            train_imgs.append(img)

    # 2. 复制验证图像（原始和目标）
    val_imgs = []
    for orig, target in zip(val_original, val_targets):
        orig_img = f"robo{orig}.png"
        target_img = f"robo{target}.png"
        shutil.copy(raw_dir / orig_img, val_dir / orig_img)
        shutil.copy(raw_dir / target_img, val_dir / target_img)
        val_imgs.append((orig_img, target_img))

    # 3. 生成训练元数据
    meta_file_train = train_dir / f"metadata_scene{scene_id}.jsonl"
    train_records = []
    for img in train_imgs:
        idx = int(img.replace("robo", "").replace(".png", ""))
        target_idx = idx + 50
        target_img = f"robo{target_idx}.png"
        if (train_dir / target_img).exists():  # 只包含有目标图像的样本
            rec = {
                "image": img,
                "edited_image": target_img,
                "edit_prompt": "预测50帧后的机械臂状态",
            }
            train_records.append(rec)

    with open(meta_file_train, "w", encoding="utf-8") as fw:
        for r in train_records:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ 场景{scene_id}的训练集metadata.jsonl生成完成，共 {len(train_records)} 条记录")

    # 4. 生成验证元数据
    meta_file_val = val_dir / f"metadata_scene{scene_id}_validation.jsonl"
    val_records = []
    for orig_img, target_img in val_imgs:
        val_record = {
            "image": orig_img,
            "edited_image": target_img,
            "edit_prompt": "预测50帧后的机械臂状态",
        }
        val_records.append(val_record)

    with open(meta_file_val, "w", encoding="utf-8") as fw:
        for r in val_records:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ 场景{scene_id}的验证集metadata.jsonl生成完成")

if __name__ == "__main__":
    # 场景1
    generate_metadata(1, [(1, 47), (51, 97)], [48, 49, 50], [98, 99, 100])
    # 场景2
    generate_metadata(2, [(101, 147), (151, 197)], [148, 149, 150], [198, 199, 200])
    # 场景3
    generate_metadata(3, [(201, 247), (251, 297)], [248, 249, 250], [298, 299, 300])