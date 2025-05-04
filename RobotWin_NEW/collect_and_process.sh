#!/bin/bash

# 设置GPU ID
GPU_ID=0

# 收集数据
echo "Collecting data for block_hammer_beat..."
bash run_task.sh block_hammer_beat $GPU_ID

echo "Collecting data for block_handover..."
bash run_task.sh block_handover $GPU_ID

echo "Collecting data for blocks_stack_easy..."
bash run_task.sh blocks_stack_easy $GPU_ID

# 处理数据
echo "Processing collected data..."
python process_data.py

echo "Done!" 