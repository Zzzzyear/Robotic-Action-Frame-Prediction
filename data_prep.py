# data_prep.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, sys
from pathlib import Path

def main():
    train_dir = Path("./sample_data/train")
    meta_file = train_dir / "metadata.jsonl"
    if not train_dir.is_dir():
        print(f"[错误] 训练目录不存在: {train_dir}", file=sys.stderr)
        sys.exit(1)

    # 1) 找到所有 PNG 并按数字排序
    imgs = sorted(
        [p for p in train_dir.iterdir() if p.suffix.lower()==".png"],
        key=lambda p: int(p.stem.replace("robo", ""))
    )
    if not imgs:
        print("[错误] 没有找到任何 PNG 文件！", file=sys.stderr)
        sys.exit(1)

    records = []
    for p in imgs:
        idx = int(p.stem.replace("robo",""))
        target = train_dir / f"robo{idx+50}.png"
        if target.exists():
            # 真正的 “原图→+50 帧” 样本
            rec = {
                "file_name":      p.name,
                "original_image": p.name,
                "edited_image":   target.name,
                "edit_prompt":    "预测50帧后的机械臂状态",
            }
        else:
            # 占位，防止 ImageFolderBuilder 报 missing metadata
            rec = {
                "file_name":      p.name,
                "original_image": p.name,
                "edited_image":   p.name,
                "edit_prompt":    "",
            }
        records.append(rec)

    # 2) 写入 metadata.jsonl
    with open(meta_file, "w", encoding="utf-8") as fw:
        for r in records:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ metadata.jsonl 生成完成，共 {len(records)} 条记录")
    print(f"文件位置: {meta_file}")

if __name__ == "__main__":
    main()
