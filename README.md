# Robotic-Action-Frame-Prediction
.AI Model for Robotic Action Frame Prediction

项目目录结构：`~/Robotic-Action-Frame-Prediction`（运行目录）

```
Robotic-Action-Frame-Prediction/
├── instruct-pix2pix
│   ├── ...  # instruct-pix2pix原有的文件和文件夹
├── RoboTwin
│   ├── ...  # RoboTwin原有的文件和文件夹
├── data
│   ├── block_hammer_beat_images
│   │   ├── input
│   │   │   ├── 000.png  # 初始000图像
│   │   │   └── ...
│   │   │   ├── 099.png  # 初始099图像
│   │   ├── target
│   │   │   ├── 000.png  # 50帧以后的000图像
│   │   │   └── ...
│   │   │   ├── 099.png  # 50帧以后的099图像
│   ├── block_handover_images
│   │   ├── input
│   │   │   ├── 000.png  # 初始000图像
│   │   │   └── ...
│   │   │   ├── 099.png  # 初始099图像
│   │   ├── target
│   │   │   ├── 000.png  # 50帧以后的000图像
│   │   │   └── ...
│   │   │   ├── 099.png  # 50帧以后的099图像
│   │   │   ├── 099.png  # 50帧以后的099图像
│   ├── blocks_stack_easy_images
│   │   ├── input
│   │   │   ├── 000.png  # 初始000图像
│   │   │   └── ...
│   │   │   ├── 099.png  # 初始099图像
│   │   ├── target
│   │   │   ├── 000.png  # 50帧以后的000图像
│   │   │   └── ...
│   │   │   ├── 099.png  # 50帧以后的099图像
├── data_prep.py               # 数据预处理
├── finetune.sh                # 微调脚本（现有）
├── finetune_instruct_pix2pix.py # 微调主程序
├── predict_with_model.py      # 预测主程序
└── predict.sh               # 预测脚本

```

