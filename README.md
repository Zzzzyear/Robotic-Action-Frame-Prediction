# Robotic-Action-Frame-Prediction
.AI Model for Robotic Action Frame Prediction

项目目录结构：`~/Robotic-Action-Frame-Prediction`（运行目录）

```
Robotic-Action-Frame-Prediction/
├── instruct-pix2pix/
│   ├── ...                # instruct-pix2pix 原有的文件和文件夹
├── RoboTwin/
│   ├── ...                # RoboTwin 原有的文件和文件夹
├── data/
│   ├── block_hammer_beat/
│   │   ├── 0000.png       # 观测结果图像
│   │   ├── 0001.png
│   │   └── ...
│   ├── block_handover/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   └── blocks_stack_easy/
│       ├── 0000.png
│       ├── 0001.png
│       └── ...
├── generate_dataset.py           # 用于生成数据集的脚本
├── finetune_instruct_pix2pix.py  # 用于微调 instruct-pix2pix 的脚本
├── evaluate_predictions.py       # 用于评估模型预测结果的脚本
└── predict_with_model.py         # 用于使用微调后的模型进行预测的脚本
```

