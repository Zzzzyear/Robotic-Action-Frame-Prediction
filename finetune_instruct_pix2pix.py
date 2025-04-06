# TODO: Fine-tuning the model with the text descriptions 'beat the block with the hammer', 'handover the blocks', and 'stack blocks' respectively.


import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from instruct_pix2pix.ldm.util import instantiate_from_config

# 假设 RoboTwin 生成的数据存储在 data 目录下
data_dir = 'data'
# 任务列表
tasks = ['block_hammer_beat', 'block_handover', 'blocks_stack_easy']
# 每个任务的观测数量
num_observations = 100
# 任务对应的文本描述
task_descriptions = {
    'block_hammer_beat': 'beat the block with the hammer',
    'block_handover': 'handover the blocks',
    'blocks_stack_easy': 'stack blocks'
}

# 加载配置文件
config_path = "instruct_pix2pix/configs/train.yaml"
config = OmegaConf.load(config_path)

# 加载预训练模型
ckpt_path = "instruct_pix2pix/stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt"
model = instantiate_from_config(config.model)
pl_sd = torch.load(ckpt_path, map_location="cpu")
sd = pl_sd["state_dict"]
model.load_state_dict(sd, strict=False)

# 准备数据集
# 这里假设 RoboTwin 已经生成了数据，并且数据存储在 data 目录下
# 每个任务的观测数据存储在对应的子目录中，如 data/block_hammer_beat
# 每个观测数据为一个图像文件，命名为 0000.png, 0001.png, ...

# 配置训练参数
trainer_config = config.pop("lightning", OmegaConf.create()).get("trainer", OmegaConf.create())
trainer_config["accelerator"] = "ddp"
trainer_opt = argparse.Namespace(**trainer_config)

# 训练器和回调
trainer_kwargs = dict()

# 默认日志配置
default_logger_cfgs = {
    "wandb": {
        "target": "pytorch_lightning.loggers.WandbLogger",
        "params": {
            "name": "instruct_pix2pix-finetune",
            "save_dir": os.getcwd(),
            "id": "instruct_pix2pix-finetune",
        }
    },
    "testtube": {
        "target": "pytorch_lightning.loggers.TestTubeLogger",
        "params": {
            "name": "testtube",
            "save_dir": os.getcwd(),
        }
    },
}
default_logger_cfg = default_logger_cfgs["wandb"]
if "logger" in config:
    logger_cfg = config.logger
else:
    logger_cfg = OmegaConf.create()
logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

# 模型检查点
default_modelckpt_cfg = {
    "target": "pytorch_lightning.callbacks.ModelCheckpoint",
    "params": {
        "dirpath": os.path.join(os.getcwd(), "checkpoints"),
        "filename": "{epoch:06}",
        "verbose": True,
        "save_last": True,
    }
}
if "modelcheckpoint" in config:
    modelckpt_cfg = config.modelcheckpoint
else:
    modelckpt_cfg = OmegaConf.create()
modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
if pl.__version__ < '1.4.0':
    trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

# 回调
default_callbacks_cfg = {
    "setup_callback": {
        "target": "instruct_pix2pix.main.SetupCallback",
        "params": {
            "resume": False,
            "now": "instruct_pix2pix-finetune",
            "logdir": os.getcwd(),
            "ckptdir": os.path.join(os.getcwd(), "checkpoints"),
            "cfgdir": os.path.join(os.getcwd(), "configs"),
            "config": config,
            "lightning_config": config.lightning,
        }
    },
    "image_logger": {
        "target": "instruct_pix2pix.main.ImageLogger",
        "params": {
            "batch_frequency": 750,
            "max_images": 4,
            "clamp": True
        }
    },
    "learning_rate_logger": {
        "target": "instruct_pix2pix.main.LearningRateMonitor",
        "params": {
            "logging_interval": "step",
        }
    },
    "cuda_callback": {
        "target": "instruct_pix2pix.main.CUDACallback"
    },
}
if pl.__version__ >= '1.4.0':
    default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

if "callbacks" in config:
    callbacks_cfg = config.callbacks
else:
    callbacks_cfg = OmegaConf.create()

callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

# 微调模型
trainer.fit(model, data)