import os
import argparse
import time
from tqdm.auto import tqdm
from dataLoader import get_dataloaders
import logging

from RetinexFormer_arch import RetinexFormer
# from waveletRetinexFormer import RetinexFormer

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage import img_as_ubyte
import math
import cv2
from utils import PSNR, calculate_ssim

parser = argparse.ArgumentParser(description='RetinexFormer Testing Script')
parser.add_argument('--dataset', type=str, help='dataset name', choices=['LOLv1', 'LOLv2_synthetic', 'LOLv2_real'], required=True)


    
def test(model, loader, device, logger):
    psnr_list = []
    ssim_list = []
    factor = 4  # Padding factor for image dimensions
    
    model.eval()
    
    loader_tqdm = tqdm(loader, desc="Validation", unit="batch", leave=False)
    
    with torch.no_grad():
        for data_batch in loader_tqdm:
            name = data_batch['name']
            input_ = data_batch['low']
            target = data_batch['high']

            # Padding in case images are not multiples of 4
            h, w = input_.shape[2], input_.shape[3]
            H = ((h + factor - 1) // factor) * factor
            W = ((w + factor - 1) // factor) * factor
            padh = H - h
            padw = W - w
            input_ = F.pad(input_, (0, padw, 0, padh), mode='reflect')
          
            restored = model(input_.to(device))

            # Unpad restored images back to original dimensions
            restored = restored[:, :, :h, :w]

            # Process image-by-image to handle batch dimension cleanly
            batch_size = input_.size(0)
            for i in range(batch_size):
                # Convert single image from Tensor (C, H, W) to Numpy (H, W, C)
                target_img = target[i].cpu().permute(1, 2, 0).numpy()
                restored_img = torch.clamp(restored[i], 0, 1).cpu().permute(1, 2, 0).numpy()

                img_name = name[i] if isinstance(name, (list, tuple)) else name

                # Compute metrics for individual image
                val_psnr = PSNR(target_img, restored_img)
                val_ssim = calculate_ssim(img_as_ubyte(target_img), img_as_ubyte(restored_img))

                # logger.info(f"Image: {img_name}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

                psnr_list.append(val_psnr)
                ssim_list.append(val_ssim)

    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)          

    return mean_psnr, mean_ssim


if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', f"psnr_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger('psnr_logger')
    
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if the function/script is called multiple times
    if not logger.handlers:
        # 3. Define the log message format
        formatter = logging.formatters if hasattr(logging, 'formatters') else None
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # 4. File Handler (writes logs to disk)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 5. Console Handler (prints logs to stdout)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if args.dataset == 'LOLv1':
        #### LOLv1 dataset paths
        low_dir_train = '/home/htoo/Retinexformer/Datasets/LOLv1/Train/input/'
        normal_dir_train = '/home/htoo/Retinexformer/Datasets/LOLv1/Train/target/'
        low_dir_test = '/home/htoo/Retinexformer/Datasets/LOLv1/Test/input/'
        normal_dir_test = '/home/htoo/Retinexformer/Datasets/LOLv1/Test/target/'
        checkpoint_path = '/home/htoo/MyRetinexformer/pretrained_weights/LOL_v1.pth'
    elif args.dataset == 'LOLv2_synthetic':
        #### LOLv2_synthetic dataset paths
        low_dir_train = '/home/htoo/Retinexformer/Datasets/LOLv2/Synthetic/Train/Low/'
        normal_dir_train = '/home/htoo/Retinexformer/Datasets/LOLv2/Synthetic/Train/Normal/'
        low_dir_test = '/home/htoo/Retinexformer/Datasets/LOLv2/Synthetic/Test/Low/'
        normal_dir_test = '/home/htoo/Retinexformer/Datasets/LOLv2/Synthetic/Test/Normal/'
        checkpoint_path = '/home/htoo/MyRetinexformer/pretrained_weights/LOL_v2_synthetic.pth'
    elif args.dataset == 'LOLv2_real':
        #### LOLv2_real dataset paths
        low_dir_train = '/home/htoo/Retinexformer/Datasets/LOLv2/Real_captured/Train/Low'
        normal_dir_train = '/home/htoo/Retinexformer/Datasets/LOLv2/Real_captured/Train/Normal'
        low_dir_test = '/home/htoo/Retinexformer/Datasets/LOLv2/Real_captured/Test/Low'
        normal_dir_test = '/home/htoo/Retinexformer/Datasets/LOLv2/Real_captured/Test/Normal'
        checkpoint_path = '/home/htoo/MyRetinexformer/pretrained_weights/LOL_v2_real.pth'
        # checkpoint_path = '/home/htoo/MyRetinexformer/LOLv2_real/best_wavelet_retinexformer_LOLv2_real.pth'
    else:
        raise ValueError("Invalid dataset specified")


    train_loader, test_loader = get_dataloaders(low_dir_train, normal_dir_train, low_dir_test, normal_dir_test,use_mixup=True)

    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(state_dict['params'])
    # model.load_state_dict(state_dict)
    print("Model setup complete.")

    test_psnr, test_ssim = test(model, test_loader, device, logger)
    print(f'Test PSNR: {test_psnr:.2f}')
    print(f'Test SSIM: {test_ssim:.4f}')
    print("Testing complete.")
