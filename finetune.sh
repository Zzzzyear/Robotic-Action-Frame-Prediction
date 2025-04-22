#!/bin/bash
# 机械臂微调执行脚本
# 使用方法: ./finetune_robot_arm.sh

echo "=== 机械臂动作预测模型微调 ==="
echo "开始时间: $(date)"

# 核心参数配置
export MODEL_NAME="timbrooks/instruct-pix2pix"
export DATA_DIR="./sample_data/train"
export OUTPUT_DIR="./robot_arm_model"
export LOG_FILE="finetune_$(date +%Y%m%d).log"

echo "----------------------------------------"
echo "模型微调参数配置:"
echo "基础模型: $MODEL_NAME"
echo "训练数据: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "----------------------------------------"

# 检查训练数据
if [ ! -f "$DATA_DIR/metadata.jsonl" ]; then
    echo "错误：未找到metadata.jsonl文件"
    exit 1
fi

# 执行微调命令
python finetune_instruct_pix2pix.py \
    --pretrained_model_name_or_path "$MODEL_NAME" \
    --train_data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resolution 512 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 100 \
    --max_train_steps 3000 \
    --validation_prompt "预测50帧后的机械臂状态" \
    --num_validation_images 3 \
    --validation_epochs 5 \
    --checkpointing_steps 200 \
    --mixed_precision "fp16" \
    --use_ema \
    --seed 42 \
    --original_image_column "original_image" \
    --edited_image_column "edited_image" \
    --edit_prompt_column "edit_prompt" \
    2>&1 | tee "$LOG_FILE"

echo "----------------------------------------"
echo "微调完成! 模型已保存至: $OUTPUT_DIR"
echo "结束时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "=== 操作结束 ==="