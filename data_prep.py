#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

def generate_metadata(scene_id, start, end, offset):
    train_dir = Path("./sample_data/train")
    meta_file = train_dir / f"metadata_scene{scene_id}.jsonl"
    if not train_dir.is_dir():
        print(f"[错误] 训练目录不存在: {train_dir}", file=sys.stderr)
        sys.exit(1)

    # 找到指定范围内的 PNG 文件并按数字排序
    imgs = sorted(
        [p for p in train_dir.iterdir() if p.suffix.lower() == ".png" and start <= int(p.stem.replace("robo", "")) <= end],
        key=lambda p: int(p.stem.replace("robo", ""))
    )
    if not imgs:
        print(f"[错误] 场景{scene_id}没有找到任何 PNG 文件！", file=sys.stderr)
        sys.exit(1)

    records = []
    for p in imgs:
        idx = int(p.stem.replace("robo", ""))
        target_idx = idx + offset
        target = train_dir / f"robo{target_idx}.png"
        if target.exists():
            # 真正的 “原图→+50帧” 样本
            rec = {
                "file_name": p.name,
                "original_image": p.name,
                "edited_image": target.name,
                "edit_prompt": "预测50帧后的机械臂状态",
            }
        else:
            # 占位，防止后续处理报错
            rec = {
                "file_name": p.name,
                "original_image": p.name,
                "edited_image": p.name,
                "edit_prompt": "",
            }
        records.append(rec)

    # 写入 metadata.jsonl
    with open(meta_file, "w", encoding="utf-8") as fw:
        for r in records:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 场景{scene_id}的metadata.jsonl生成完成，共 {len(records)} 条记录")
    print(f"文件位置: {meta_file}")

if __name__ == "__main__":
    # 场景1: robo1-47 和 robo51-97
    generate_metadata(1, 1, 47, 50)
    # 场景2: robo101-147 和 robo151-197
    generate_metadata(2, 101, 147, 50)
    # 场景3: robo201-247 和 robo251-297
    generate_metadata(3, 201, 247, 50)