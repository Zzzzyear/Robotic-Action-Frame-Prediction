#!/usr/bin/env bash
# finetune.sh
# 用于为每个场景微调 InstructPix2Pix 模型
set -euo pipefail

for scene in 1 2 3; do
    LOG="finetune_scene${scene}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== 机械臂动作预测微调 - 场景${scene} ==="         | tee "$LOG"
    echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
    echo "----------------------------------------" | tee -a "$LOG"

    # 1. 校验 metadata.jsonl
    echo "[1/3] 校验 metadata.jsonl"       | tee -a "$LOG"
    python3 - <<PYCODE 2>&1             | tee -a "$LOG"
import json, sys
from pathlib import Path

scene = $scene
base = Path("sample_data/train")
meta = base / f"metadata_scene{scene}.jsonl"
if not meta.exists():
    print(f"[错误] 找不到 {meta}", file=sys.stderr); sys.exit(1)

bad = []
with open(meta, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        d = json.loads(line)
        for k in ("original_image", "edited_image"):
            p = base / d[k]
            if not p.exists():
                bad.append((i, k, d[k]))
if bad:
    for i, k, fn in bad:
        print(f"[错误] 第{i}行字段{k}文件缺失: {fn}", file=sys.stderr)
    sys.exit(1)
print("✅ 校验通过")
PYCODE

    # 2. 启动微调
    echo "[2/3] 启动微调"                   | tee -a "$LOG"
    accelerate launch --num_processes=1 --mixed_precision=fp16 \
        finetune_instruct_pix2pix.py \
        --pretrained_model_name_or_path "timbrooks/instruct-pix2pix" \
        --train_data_dir "./sample_data/train" \
        --dataset_name "imagefolder" \
        --dataset_config_name "metadata_scene${scene}.jsonl" \
        --original_image_column "original_image" \
        --edited_image_column "edited_image" \
        --edit_prompt_column "edit_prompt" \
        --output_dir "./robot_arm_model_scene${scene}" \
        --resolution 128 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-5 \
        --lr_scheduler "constant_with_warmup" \
        --lr_warmup_steps 10 \
        --max_train_steps 100 \
        --validation_prompt "预测50帧后的机械臂状态" \
        --num_validation_images 1 \
        --validation_epochs 5 \
        --checkpointing_steps 50 \
        --mixed_precision "fp16" \
        --seed 42 \
        --report_to "tensorboard" 2>&1 | tee -a "$LOG"

    echo "微调日志: $LOG"
    echo "模型输出目录: ./robot_arm_model_scene${scene}"
    echo "=== 场景${scene}任务完成 ==="                 | tee -a "$LOG"
done