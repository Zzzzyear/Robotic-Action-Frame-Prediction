import pickle
import numpy as np
import os
import cv2
from pathlib import Path

# 配置参数
PKL_PATH = '/root/autodl-tmp/RoboTwin-main/data_pkl/block_hammer_beat_D435_pkl/episode0/0.pkl'
OUTPUT_DIR = '/root/autodl-tmp/RoboTwin-main/data/pkl_analysis'

def analyze_dict(data, prefix='', max_depth=3, current_depth=0):
    """递归分析字典结构"""
    if current_depth >= max_depth:
        return f"{prefix} (达到最大深度)"
    
    result = []
    if isinstance(data, dict):
        for key, value in data.items():
            shape_info = ""
            if isinstance(value, np.ndarray):
                shape_info = f" - shape: {value.shape}, dtype: {value.dtype}"
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    shape_info = f" - list of arrays, first shape: {value[0].shape}"
            
            result.append(f"{prefix}{key}{shape_info}")
            
            if isinstance(value, (dict, list)) and current_depth < max_depth:
                if isinstance(value, dict):
                    result.extend(analyze_dict(value, prefix + key + '.', max_depth, current_depth + 1))
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    result.extend(analyze_dict(value[0], prefix + key + '[0].', max_depth, current_depth + 1))
    
    return result

def save_image(img, filename):
    """保存图像"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            # 转换为BGR（如果是RGB）
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, img_bgr)
            return True
    return False

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载PKL文件
    print(f"加载PKL文件: {PKL_PATH}")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # 分析顶层结构
    print("PKL文件顶层结构:")
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(f"- {key}: ndarray, shape={data[key].shape}, dtype={data[key].dtype}")
        elif isinstance(data[key], list):
            if len(data[key]) > 0:
                if isinstance(data[key][0], np.ndarray):
                    print(f"- {key}: list of ndarrays, first shape={data[key][0].shape}")
                else:
                    print(f"- {key}: list of {type(data[key][0]).__name__}, length={len(data[key])}")
            else:
                print(f"- {key}: empty list")
        elif isinstance(data[key], dict):
            print(f"- {key}: dict with keys {list(data[key].keys())}")
        else:
            print(f"- {key}: {type(data[key]).__name__}")
    
    # 详细分析observation字段
    if 'observation' in data:
        print("\n观察字段详细结构:")
        obs_structure = analyze_dict(data['observation'], 'observation.')
        for item in obs_structure:
            print(f"- {item}")
    
    # 寻找可能的图像数据
    print("\n尝试提取图像数据:")
    
    # 方法1: 直接查找顶层的rgb字段
    if 'rgb' in data:
        print("在顶层找到rgb字段")
        if isinstance(data['rgb'], list) and len(data['rgb']) > 0:
            img = data['rgb'][0]
            out_path = os.path.join(OUTPUT_DIR, 'rgb_toplevel.png')
            if save_image(img, out_path):
                print(f"已保存顶层RGB图像: {out_path}")
    
    # 方法2: 在observation中查找rgb或image字段
    if 'observation' in data:
        obs = data['observation']
        if isinstance(obs, dict):
            # 常见的图像字段名
            for img_field in ['rgb', 'image', 'images', 'camera', 'head_camera', 'front_camera']:
                if img_field in obs:
                    print(f"在observation中找到{img_field}字段")
                    img_data = obs[img_field]
                    if isinstance(img_data, np.ndarray):
                        out_path = os.path.join(OUTPUT_DIR, f'obs_{img_field}.png')
                        if save_image(img_data, out_path):
                            print(f"已保存observation.{img_field}图像: {out_path}")
                    elif isinstance(img_data, list) and len(img_data) > 0:
                        for i, img in enumerate(img_data[:3]):  # 只尝试前3个
                            out_path = os.path.join(OUTPUT_DIR, f'obs_{img_field}_{i}.png')
                            if save_image(img, out_path):
                                print(f"已保存observation.{img_field}[{i}]图像: {out_path}")
    
    # 方法3: 扫描所有numpy数组，找出形状符合图像的数组
    print("\n扫描所有numpy数组:")
    
    def scan_for_images(obj, path=''):
        if isinstance(obj, np.ndarray):
            # 判断是否可能是图像
            if len(obj.shape) == 3 and obj.shape[2] == 3 and obj.shape[0] > 10 and obj.shape[1] > 10:
                print(f"可能的图像: {path}, shape={obj.shape}")
                out_path = os.path.join(OUTPUT_DIR, f'possible_img_{path.replace(".", "_")}.png')
                if save_image(obj, out_path):
                    print(f"已保存可能的图像: {out_path}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                scan_for_images(v, path + '.' + k if path else k)
        elif isinstance(obj, list) and len(obj) > 0:
            for i, item in enumerate(obj[:3]):  # 只检查前3个
                scan_for_images(item, f"{path}[{i}]")
    
    scan_for_images(data)
    
    print(f"\n分析完成，图像保存在: {OUTPUT_DIR}")

if __name__ == '__main__':
    main() 