import os
import json
import shutil
from pathlib import Path
import numpy as np
import cv2

# 任务配置
TASKS = {
    'block_hammer_beat': 'beat the block with the hammer',
    'block_handover': 'handover the blocks',
    'blocks_stack_easy': 'stack the blocks'
}

def process_task_data(task_name, source_dir, target_dir):
    """处理单个任务的数据"""
    # 创建目标目录
    input_dir = os.path.join(target_dir, 'images', 'input')
    target_dir = os.path.join(target_dir, 'images', 'target')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有数据文件
    data_files = [f for f in os.listdir(source_dir) if f.endswith('.pkl')]
    
    instructions = []
    
    for idx, data_file in enumerate(data_files):
        # 读取数据文件
        data_path = os.path.join(source_dir, data_file)
        data = np.load(data_path, allow_pickle=True)
        
        # 提取第0帧和第50帧的图像
        frame_0 = data['rgb'][0]  # 当前帧
        frame_50 = data['rgb'][50]  # 目标帧
        
        # 保存图像
        input_path = os.path.join(input_dir, f'{task_name}_{idx:03d}.png')
        target_path = os.path.join(target_dir, f'{task_name}_{idx:03d}.png')
        
        cv2.imwrite(input_path, frame_0)
        cv2.imwrite(target_path, frame_50)
        
        # 添加指令
        instructions.append({
            'input_image': f'{task_name}_{idx:03d}.png',
            'target_image': f'{task_name}_{idx:03d}.png',
            'instruction': TASKS[task_name]
        })
    
    return instructions

def main():
    # 创建主数据集目录
    dataset_dir = 'instructpix2pix_dataset'
    os.makedirs(dataset_dir, exist_ok=True)
    
    all_instructions = []
    
    # 处理每个任务
    for task_name in TASKS.keys():
        source_dir = os.path.join('data', f'{task_name}_D435')
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory {source_dir} does not exist")
            continue
            
        instructions = process_task_data(task_name, source_dir, dataset_dir)
        all_instructions.extend(instructions)
    
    # 保存指令文件
    with open(os.path.join(dataset_dir, 'instructions.json'), 'w') as f:
        json.dump(all_instructions, f, indent=2)
    
    print(f"Processed {len(all_instructions)} image pairs")
    print(f"Dataset saved to {dataset_dir}")

if __name__ == '__main__':
    main() 