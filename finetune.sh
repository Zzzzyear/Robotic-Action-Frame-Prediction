#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com  # Hugging Face 镜像加速

for scene in 1 2 3; do
    LOG="finetune_scene${scene}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== 机械臂动作预测微调 - 场景${scene} ==="         | tee "$LOG"
    echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
    echo "----------------------------------------" | tee -a "$LOG"

    export SCENE_ID=$scene

    accelerate launch --num_processes=1 --mixed_precision=fp16 \
        finetune_instruct_pix2pix.py \
        --pretrained_model_name_or_path "timbrooks/instruct-pix2pix" \
        --train_data_dir "./sample_data/train" \
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
        --report_to "tensorboard" \
        --scene_id $scene 2>&1 | tee -a "$LOG"

    echo "微调日志: $LOG"
    echo "模型输出目录: ./robot_arm_model_scene${scene}"
    echo "=== 场景${scene}任务完成 ==="                 | tee -a "$LOG"
done