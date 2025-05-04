import os
import pickle
import json
import cv2
import numpy as np
from pathlib import Path

# 配置参数
TASK_NAME = 'blocks_stack_easy'
PKL_DIR = f'/root/autodl-tmp/RoboTwin-main/data_pkl/{TASK_NAME}_D435_pkl'
OUTPUT_DIR = f'/root/autodl-tmp/RoboTwin-main/data/{TASK_NAME}_images'

# 使用的摄像头 (可以是 'head_camera', 'front_camera', 'left_camera', 'right_camera')
CAMERA = 'head_camera'

# 帧索引设置
FIRST_FRAME_INDEX = 0  # 第一帧索引
SECOND_FRAME_INDEX = 50  # 第二帧索引

# 指令列表 - 描述从第0帧到第50帧的变化
INSTRUCTIONS = [
    "Place the black block on top of the red block at the center",
    "Stack the black block on the red block",
    "Put the black block above the red block",
    "Position the black block on top of the red block",
    "Set the black block over the red block in the center",
    "Place one block on top of another to create a stack"
]

def extract_rgb_from_data(data):
    """从数据中提取RGB图像"""
    # 检查observation字段
    if 'observation' in data and CAMERA in data['observation'] and 'rgb' in data['observation'][CAMERA]:
        return data['observation'][CAMERA]['rgb']
    return None

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'input'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'target'), exist_ok=True)
    
    # 检查输入目录是否存在
    if not os.path.exists(PKL_DIR):
        print(f"错误: PKL目录不存在: {PKL_DIR}")
        print(f"请先确认数据目录: {PKL_DIR}")
        return
    
    # 获取所有episode目录
    episode_dirs = sorted([d for d in os.listdir(PKL_DIR) if d.startswith('episode') and os.path.isdir(os.path.join(PKL_DIR, d))])
    
    if not episode_dirs:
        print("未找到任何episode目录!")
        return
        
    print(f"找到 {len(episode_dirs)} 个episode目录")
    
    # 用于保存指令信息的列表
    instruction_data = []
    img_count = 0
    
    # 处理每个episode目录
    for idx, episode_dir in enumerate(episode_dirs):
        if idx >= 100:  # 只处理前100个episode
            break
            
        episode_path = os.path.join(PKL_DIR, episode_dir)
        print(f"处理目录 {episode_dir} ({idx+1}/{min(100, len(episode_dirs))})")
        
        # 定义要提取的具体pkl文件名
        frame0_pkl = f"{FIRST_FRAME_INDEX}.pkl"  # 0.pkl
        frame50_pkl = f"{SECOND_FRAME_INDEX}.pkl"  # 50.pkl
        
        # 检查这两个文件是否存在
        first_pkl_path = os.path.join(episode_path, frame0_pkl)
        second_pkl_path = os.path.join(episode_path, frame50_pkl)
        
        if not os.path.exists(first_pkl_path):
            print(f"  警告: {episode_dir} 中找不到 {frame0_pkl}，跳过该episode")
            continue
            
        try:
            # 读取第一个帧的PKL文件
            with open(first_pkl_path, 'rb') as f:
                first_data = pickle.load(f)
            
            # 提取RGB数据
            frame_0 = extract_rgb_from_data(first_data)
            if frame_0 is None:
                print(f"  警告: {episode_dir}/{frame0_pkl} 中未找到RGB数据")
                continue
                
            # 读取第50帧的PKL文件
            if os.path.exists(second_pkl_path):
                with open(second_pkl_path, 'rb') as f:
                    second_data = pickle.load(f)
                    
                frame_50 = extract_rgb_from_data(second_data)
                if frame_50 is None:
                    print(f"  警告: {episode_dir}/{frame50_pkl} 中未找到RGB数据，使用第一帧作为目标")
                    frame_50 = frame_0.copy()
            else:
                print(f"  警告: {episode_dir} 中找不到 {frame50_pkl}，使用第一帧作为目标")
                frame_50 = frame_0.copy()
            
            # 保存图片
            input_filename = f"{img_count:03d}.png"
            target_filename = f"{img_count:03d}.png"
            
            input_path = os.path.join(OUTPUT_DIR, 'input', input_filename)
            target_path = os.path.join(OUTPUT_DIR, 'target', target_filename)
            
            # 检查图像形状和类型
            if not isinstance(frame_0, np.ndarray) or not isinstance(frame_50, np.ndarray):
                print(f"  警告: {episode_dir} 图像数据不是numpy数组，跳过")
                continue
                
            # 确保图像是3通道的
            if len(frame_0.shape) < 3 or frame_0.shape[2] < 3:
                print(f"  警告: {episode_dir} 图像不是3通道，跳过")
                continue
                
            # OpenCV默认使用BGR格式，如果图像是RGB，需要转换
            if frame_0.shape[2] == 3:
                frame_0 = cv2.cvtColor(frame_0, cv2.COLOR_RGB2BGR)
                frame_50 = cv2.cvtColor(frame_50, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(input_path, frame_0)
            cv2.imwrite(target_path, frame_50)
            
            # 选择一个随机指令
            instruction = np.random.choice(INSTRUCTIONS)
            
            # 添加到指令数据
            instruction_data.append({
                "input_image": input_filename,
                "target_image": target_filename,
                "instruction": instruction
            })
            
            print(f"  成功处理 {frame0_pkl} 和 {frame50_pkl}")
            img_count += 1
            
        except Exception as e:
            print(f"  处理 {episode_dir} 时出错: {e}")
    
    # 保存指令文件
    if instruction_data:
        # 创建指令文件格式
        instructions_json = {"instructions": instruction_data}
        
        with open(os.path.join(OUTPUT_DIR, 'instructions.json'), 'w', encoding='utf-8') as f:
            json.dump(instructions_json, f, indent=2, ensure_ascii=False)
        
        print(f"\n已处理 {len(instruction_data)} 个图像对")
        print(f"数据已保存到 {OUTPUT_DIR}")
        print(f"- 输入图像: {OUTPUT_DIR}/input/")
        print(f"- 目标图像: {OUTPUT_DIR}/target/")
        print(f"- 指令文件: {OUTPUT_DIR}/instructions.json")
    else:
        print("警告: 没有处理任何有效的图像对")

if __name__ == '__main__':
    main() 