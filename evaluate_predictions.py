# TODO: Evaluate using the SSIM (Structural Similarity Index) and PSNR (Peak Signal-to-Noise Ratio) metrics.


import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_predictions(ground_truth_dir, predictions_dir):
    image_files = sorted(os.listdir(ground_truth_dir))
    ssim_scores = []
    psnr_scores = []

    for image_file in image_files:
        gt_image_path = os.path.join(ground_truth_dir, image_file)
        pred_image_path = os.path.join(predictions_dir, image_file)

        gt_image = Image.open(gt_image_path)
        pred_image = Image.open(pred_image_path)

        gt_image = np.array(gt_image).astype(np.float64)
        pred_image = np.array(pred_image).astype(np.float64)

        # 计算 SSIM 和 PSNR
        ssim_score = ssim(gt_image, pred_image, multichannel=True)
        psnr_score = psnr(gt_image, pred_image)

        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)

    # 计算平均 SSIM 和 PSNR
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)

    return avg_ssim, avg_psnr

if __name__ == "__main__":
    tasks = ['block_hammer_beat', 'block_handover', 'blocks_stack_easy']
    for task in tasks:
        ground_truth_dir = os.path.join('data', task)
        predictions_dir = os.path.join('predictions', task)
        avg_ssim, avg_psnr = evaluate_predictions(ground_truth_dir, predictions_dir)
        print(f'{task}: SSIM = {avg_ssim:.4f}, PSNR = {avg_psnr:.4f}')


