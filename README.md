# Robotic-Action-Frame-Prediction
这是一个用于机械臂动作帧预测的 AI 模型项目，以下是详细的环境搭建和运行步骤说明。

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


## 项目概述
本项目旨在通过对机械臂动作帧进行预测，借助预训练模型 InstructPix2Pix 进行微调，以实现对机械臂未来动作状态的预测。项目包含数据预处理、模型微调、预测和结果评估等环节。

## 环境搭建

### 1. 创建并激活 conda 环境
所有脚本都需要在 `~/Robotic-Action-Frame-Prediction/` 目录下运行。首先，创建一个名为 `robotic` 的 conda 环境，并激活它：
```bash
conda create -n robotic python=3.10 -y
conda activate robotic
```

### 2. 安装 PyTorch（CUDA 版本）
优先安装 PyTorch 的 CUDA 版本，以获得更好的性能：
```bash
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装其他依赖
使用 `--no-deps` 避免覆盖 PyTorch 的依赖：
```bash
pip install -r requirements_finetune.txt --no-deps
```

### 4. 手动补全依赖
手动安装 `scikit-image`，因为 `evaluate_predictions.py` 需要用到它：
```bash
pip install scikit-image
```

### 5. 处理缺失包
如果在运行过程中发现仍有缺失的包，请手动安装关键组件。

## 运行步骤

### 1. 数据预处理
所有脚本都需要在 `~/Robotic-Action-Frame-Prediction/` 目录下运行。运行以下命令对数据进行处理，每个场景将使用 80 个样本进行训练，20 个样本进行验证：
```bash
python data_prep.py
```
此脚本会将原始数据处理为训练集和验证集，并生成对应的元数据文件。具体任务包括将指定任务的图像复制到相应的训练和验证目录，并为每个数据集生成 `metadata.jsonl` 文件。

### 2. 模型微调
运行 `finetune.sh` 脚本进行模型微调，微调完成后的参数将保存在新生成的 `robot_arm_model_scene{i}` 目录中：
```bash
./finetune.sh
```
该脚本会针对三个场景（场景 1、场景 2、场景 3）分别进行微调操作，并记录每个场景的微调日志。微调过程中使用 `accelerate` 库进行分布式训练配置，使用预训练的 `InstructPix2Pix` 模型，并根据指定的参数进行训练。

### 3. 模型预测
运行 `predict.sh` 脚本进行预测，预测结果将保存在 `predictions` 目录下：
```bash
./predict.sh
```
此脚本会针对三个场景分别进行预测，并记录每个场景的预测日志。预测过程中使用微调后的模型对验证集中的图像进行预测，并将预测结果保存为图像文件。

### 4. 结果评估
运行以下命令使用指标评估预测结果，将生成三个场景的评估指标：
```bash
python evaluate_predictions.py
```
该脚本会使用 SSIM（Structural Similarity Index）和 PSNR（Peak Signal-to-Noise Ratio）指标对预测结果进行评估，并输出每个场景的平均 SSIM 和 PSNR 值。

## 项目文件说明
- `data_prep.py`：数据预处理脚本，用于将原始数据处理为训练集和验证集，并生成元数据文件。
- `finetune.sh`：模型微调脚本，用于对 `InstructPix2Pix` 模型进行微调。
- `finetune_instruct_pix2pix.py`：模型微调的具体实现脚本，包含数据加载、模型训练等逻辑。
- `predict.sh`：模型预测脚本，用于使用微调后的模型进行预测。
- `predict_with_model.py`：模型预测的具体实现脚本，包含图像加载、模型推理等逻辑。
- `evaluate_predictions.py`：结果评估脚本，用于使用 SSIM 和 PSNR 指标评估预测结果。
- `requirements_finetune.txt`：项目所需的依赖包列表。
- `environment_finetune.yml`：项目的 conda 环境配置文件。

通过以上步骤，你可以完成整个机械臂动作帧预测的流程，包括数据预处理、模型微调、预测和结果评估。