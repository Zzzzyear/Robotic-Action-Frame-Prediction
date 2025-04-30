#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com  # Hugging Face 镜像加速

for scene in 1 2 3; do
    LOG="predict_scene${scene}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== 机械臂动作预测 - 场景${scene} ==="         | tee "$LOG"
    echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
    echo "----------------------------------------" | tee -a "$LOG"

    python predict_with_model.py \
        --model_path "./robot_arm_model_scene${scene}" \
        --data_dir "./data/validation" \
        --output_dir "./predictions/scene${scene}" \
        --scene_id ${scene} \
        --prompt "预测50帧后的机械臂状态" \
        --image_size 320 \
        --batch_size 1 \
        --device cuda 2>&1 | tee -a "$LOG"

    echo "预测日志: $LOG"
    echo "预测输出目录: ./predictions/scene${scene}"
    echo "=== 场景${scene}预测完成 ==="                 | tee -a "$LOG"
done