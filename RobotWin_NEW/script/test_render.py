import sys
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

# 尝试设置无头模式
os.environ["SAPIEN_HEADLESS"] = "1"

sys.path.append(os.path.join(parent_dir, '../../tools'))
import numpy as np
import pdb
import json
import torch
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import gymnasium as gym
import toppra as ta
import transforms3d as t3d
from collections import OrderedDict
import random

class Sapien_TEST(gym.Env):
    def __init__(self):
        super().__init__()
        ta.setup_logging("CRITICAL") # hide logging
        
        # 诊断信息
        print("CUDA是否可用:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA设备数量:", torch.cuda.device_count())
            print("当前CUDA设备:", torch.cuda.current_device())
            print("CUDA设备名称:", torch.cuda.get_device_name(0))
        
        try:
            print("开始尝试设置场景...")
            self.setup_scene()
            print("渲染设置成功!")
        except Exception as e:
            print(f"渲染设置失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def setup_scene(self,**kwargs):
        '''
        Set the scene using updated SAPIEN API
        '''
        # 直接创建scene，不使用deprecated的Engine
        self.scene = sapien.Scene()
        
        # 设置渲染配置
        from sapien.render import set_global_config
        set_global_config(max_num_materials=50000, max_num_textures=50000)
        
        # 尝试获取可用设备
        try:
            from sapien.render import get_available_devices
            available_devices = get_available_devices()
            print("可用渲染设备:", available_devices)
        except:
            print("无法获取可用渲染设备信息")
        
        # 设置光线追踪相关配置
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

if __name__ == '__main__':
    print("开始测试SAPIEN渲染...")
    a = Sapien_TEST()