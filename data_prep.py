#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path
import os

# 设置 Hugging Face 镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def generate_metadata_for_task(task_name, scene_id, train_indices, val_indices):
    """
    为指定任务生成训练和验证元数据，并复制图像到对应目录。

    参数：
    - task_name: 任务名称（'block_hammer_beat_images', 'block_handover_images', 'blocks_stack_easy_images'）
    - scene_id: 场景编号（1, 2, 3）
    - train_indices: 训练集的图像索引列表，例如 [0, 1, ..., 79]
    - val_indices: 验证集的图像索引列表，例如 [80, 81, ..., 99]
    """
    # 定义原始数据目录
    raw_input_dir = Path(f"data/{task_name}/input")
    raw_target_dir = Path(f"data/{task_name}/target")

    # 定义目标目录（生成数据到 data/train 和 data/validation）
    train_dir = Path(f"data/train/scene{scene_id}")
    val_dir = Path(f"data/validation/scene{scene_id}")
    train_input_dir = train_dir / "input"
    train_target_dir = train_dir / "target"
    val_input_dir = val_dir / "input"
    val_target_dir = val_dir / "target"

    # 创建目录
    train_input_dir.mkdir(parents=True, exist_ok=True)
    train_target_dir.mkdir(parents=True, exist_ok=True)
    val_input_dir.mkdir(parents=True, exist_ok=True)
    val_target_dir.mkdir(parents=True, exist_ok=True)

    # 1. 处理训练集
    train_records = []
    for idx in train_indices:
        img_name = f"{idx:03d}.png"
        shutil.copy(raw_input_dir / img_name, train_input_dir / img_name)
        shutil.copy(raw_target_dir / img_name, train_target_dir / img_name)
        rec = {
            "image": f"input/{img_name}",
            "edited_image": f"target/{img_name}",
            "edit_prompt": "预测50帧后的机械臂状态",
        }
        train_records.append(rec)

    # 2. 处理验证集
    val_records = []
    for idx in val_indices:
        img_name = f"{idx:03d}.png"
        shutil.copy(raw_input_dir / img_name, val_input_dir / img_name)
        shutil.copy(raw_target_dir / img_name, val_target_dir / img_name)
        rec = {
            "image": f"input/{img_name}",
            "edited_image": f"target/{img_name}",
            "edit_prompt": "预测50帧后的机械臂状态",
        }
        val_records.append(rec)

    # 3. 生成训练元数据
    meta_file_train = train_dir / f"metadata_scene{scene_id}.jsonl"
    with open(meta_file_train, "w", encoding="utf-8") as fw:
        for r in train_records:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ 任务 {task_name}（场景 {scene_id}）的训练集 metadata.jsonl 生成完成，共 {len(train_records)} 条记录")
    print(f"训练数据路径: {train_dir}")

    # 4. 生成验证元数据
    meta_file_val = val_dir / f"metadata_scene{scene_id}_validation.jsonl"
    with open(meta_file_val, "w", encoding="utf-8") as fw:
        for r in val_records:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ 任务 {task_name}（场景 {scene_id}）的验证集 metadata.jsonl 生成完成，共 {len(val_records)} 条记录")
    print(f"验证数据路径: {val_dir}")

if __name__ == "__main__":
    # 定义任务和对应的场景 ID
    tasks = [
        ("block_hammer_beat_images", 1),
        ("block_handover_images", 2),
        ("blocks_stack_easy_images", 3),
    ]

    # 划分训练和验证集：前 80 张用于训练，后 20 张用于验证
    train_indices = list(range(0, 80))
    val_indices = list(range(80, 100))

    # 为每个任务生成元数据
    for task_name, scene_id in tasks:
        generate_metadata_for_task(task_name, scene_id, train_indices, val_indices)