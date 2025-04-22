import json
import os
from pathlib import Path
from PIL import Image

train_dir = Path("./sample_data/train")
output_metadata = train_dir / "metadata.jsonl"

all_frames = sorted(
    [f for f in os.listdir(train_dir) if f.startswith("robo") and f.endswith(".png")],
    key=lambda x: int(x[4:-4])
)

valid_pairs = 0  # 新增计数器

with open(output_metadata, "w", encoding="utf-8") as meta_file:
    for i in range(len(all_frames)):
        current_frame = all_frames[i]
        current_num = int(current_frame[4:-4])
        target_num = current_num + 50
        target_frame = f"robo{target_num}.png"

        if target_frame in all_frames:
            try:
                with Image.open(train_dir / current_frame) as img:
                    img.verify()
                with Image.open(train_dir / target_frame) as img:
                    img.verify()

                meta_file.write(json.dumps({
                    "original_image": current_frame,
                    "edited_image": target_frame,
                    "edit_prompt": "预测50帧后的机械臂状态"
                }, ensure_ascii=False) + "\n")

                valid_pairs += 1  # 成功配对时计数

            except (IOError, SyntaxError) as e:
                print(f"损坏图像跳过: {current_frame} -> {target_frame}, 错误: {str(e)}")

print(f"数据集准备完成，共生成 {valid_pairs} 个有效训练样本")  # 使用实际计数
print(f"元数据文件已保存至: {output_metadata}")