import os
import json
import numpy as np
import cv2
from pathlib import Path

# 指定任务名称和文字指导
TASK_NAME = 'block_hammer_beat'
TASK_INSTRUCTION = 'beat the block with the hammer'

# 数据路径配置
PKL_DIR = f'./data/{TASK_NAME}_D435_pkl'
OUTPUT_DIR = f'./data/{TASK_NAME}_images'

# 创建输出目录
os.makedirs(os.path.join(OUTPUT_DIR, 'input'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'target'), exist_ok=True)

# 处理数据函数
def process_data():
    instructions = []
    
    # 获取所有PKL文件
    import pickle
    pkl_files = [f for f in os.listdir(PKL_DIR) if f.endswith('.pkl')]
    
    for idx, pkl_file in enumerate(pkl_files):
        # 跳过非数字命名的文件
        if not pkl_file[0].isdigit():
            continue
            
        # 读取PKL文件
        pkl_path = os.path.join(PKL_DIR, pkl_file)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 提取第0帧和第50帧
        frame_0 = data['rgb'][0]  # 初始帧
        frame_50 = data['rgb'][50]  # 目标帧(若数据不足50帧则取最后一帧)
        
        # BGR转RGB (如果需要的话)
        if frame_0.shape[2] == 3:
            frame_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2RGB)
            frame_50 = cv2.cvtColor(frame_50, cv2.COLOR_BGR2RGB)
        
        # 保存图片
        input_path = os.path.join(OUTPUT_DIR, 'input', f'{idx:03d}.png')
        target_path = os.path.join(OUTPUT_DIR, 'target', f'{idx:03d}.png')
        
        cv2.imwrite(input_path, frame_0)
        cv2.imwrite(target_path, frame_50)
        
        # 添加指令
        instructions.append({
            'input_image': f'{idx:03d}.png',
            'target_image': f'{idx:03d}.png',
            'instruction': TASK_INSTRUCTION
        })
    
    # 保存指令文件
    with open(os.path.join(OUTPUT_DIR, 'instructions.json'), 'w') as f:
        json.dump(instructions, f, indent=2)
    
    print(f"处理了 {len(instructions)} 对图像")
    print(f"数据保存在 {OUTPUT_DIR}")

if __name__ == '__main__':
    process_data()