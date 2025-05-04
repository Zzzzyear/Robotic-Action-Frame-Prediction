import yaml
import os

# 配置文件路径
config_path = 'task_config/block_hammer_beat.yml'

# 读取当前配置
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 修改配置
config['save_freq'] = 50  # 设置为每50帧保存一次
config['data_type']['rgb'] = True  # 启用RGB图像
config['data_type']['depth'] = False  # 禁用深度图
config['data_type']['pointcloud'] = False  # 禁用点云

# 保存修改后的配置
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"已更新配置文件 {config_path}:")
print(f"- save_freq: {config['save_freq']}")
print(f"- rgb: {config['data_type']['rgb']}")
print(f"- depth: {config['data_type']['depth']}")
print(f"- pointcloud: {config['data_type']['pointcloud']}")
print("\n使用以下命令收集数据:")
print("bash run_task.sh block_hammer_beat 0")
print("\n然后使用以下命令转换数据:")
print("python convert_hammer_data.py") 