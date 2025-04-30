import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def evaluate_predictions(ground_truth_dir, predictions_dir, scene_id):
    ground_truth_dir = os.path.join(ground_truth_dir, 'target')
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_files = []
    for f in sorted(os.listdir(ground_truth_dir)):
        if f.lower().endswith(image_extensions):
            try:
                index = int(os.path.splitext(f)[0])
                if 80 <= index <= 99:
                    image_files.append(f)
            except ValueError:
                print(f"警告: 无效文件名 {f}, 无法提取编号")

    ssim_scores = []
    psnr_scores = []

    for image_file in image_files:
        try:
            index = int(os.path.splitext(image_file)[0])
            pred_image_name = f"predicted_scene{scene_id}_{index}.png"
            pred_image_path = os.path.join(predictions_dir, pred_image_name)

            gt_image = Image.open(os.path.join(ground_truth_dir, image_file))
            pred_image = Image.open(pred_image_path)

            if gt_image.size != pred_image.size:
                print(f"警告: {image_file} 尺寸不一致，跳过 (真实尺寸:{gt_image.size}, 预测尺寸:{pred_image.size})")
                continue

            gt_array = np.array(gt_image).astype(np.float64)
            pred_array = np.array(pred_image).astype(np.float64)

            # 获取图像尺寸信息
            height, width, channels = gt_array.shape
            # 设置win_size为小于等于较小边的奇数（这里取11，可根据实际情况调整）
            win_size = 11 if min(height, width) >= 11 else min(height, width) if min(height, width) % 2 == 1 else min(
                height, width) - 1

            # 显式指定channel_axis=2（因为图像数组格式为HWC，通道轴是最后一维）
            ssim_score = ssim(gt_array, pred_array,
                              multichannel=True,
                              win_size=win_size,
                              channel_axis=2,  # 关键修复：指定通道轴
                              data_range=255)  # 假设图像像素值范围是0-255

            psnr_score = psnr(gt_array, pred_array, data_range=255)  # PSNR也显式指定数据范围

            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)

        except FileNotFoundError as e:
            print(f"错误: 未找到预测图像 {pred_image_name}, 原因: {e}")
        except Exception as e:
            print(f"处理 {image_file} 时发生未知错误: {e}")

    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0.0
    return avg_ssim, avg_psnr


if __name__ == "__main__":
    scene_tasks = {
        1: ('block_hammer_beat_images', 'scene1'),
        2: ('block_handover_images', 'scene2'),
        3: ('blocks_stack_easy_images', 'scene3')
    }

    for scene_id, (task, scene_dir) in scene_tasks.items():
        ground_truth_base = os.path.join('data', task)
        predictions_base = os.path.join('predictions', scene_dir)

        avg_ssim, avg_psnr = evaluate_predictions(ground_truth_base, predictions_base, scene_id)
        print(f"场景 {scene_id} ({task}): SSIM = {avg_ssim:.4f}, PSNR = {avg_psnr:.4f}")