import cv2
import numpy as np
import ot
from tqdm import tqdm
import os, glob
from scipy.stats import wasserstein_distance

# 이미지 불러오기
sharp_image = cv2.imread(f'/raid/joowan/GoPro/test/sharp/000124.png', cv2.IMREAD_GRAYSCALE)
# deblur_image = cv2.imread(f'/raid/joowan/results/GoPro/UFPDeblur/000366.png', cv2.IMREAD_GRAYSCALE)
# deblur_image = cv2.imread(f'/raid/joowan/results/GoPro/Uformer/000366.png', cv2.IMREAD_GRAYSCALE)
deblur_image = cv2.imread(f'/raid/joowan/results/GoPro/NAF_DPS/best/000124.png', cv2.IMREAD_GRAYSCALE)

# Sobel 필터 적용
sharp_sobel = cv2.Sobel(sharp_image, cv2.CV_64F, 1, 1, ksize=3)
deblur_sobel = cv2.Sobel(deblur_image, cv2.CV_64F, 1, 1, ksize=3)



# 음수 값을 0으로 클리핑
sharp_sobel = np.maximum(sharp_sobel, 0)
deblur_sobel = np.maximum(deblur_sobel, 0)


# Flatten the 2D arrays
sharp_sobel_flat = sharp_sobel.flatten()
deblur_sobel_flat = deblur_sobel.flatten()

# Calculate Wasserstein distance
distance = wasserstein_distance(sharp_sobel_flat, deblur_sobel_flat)

print(distance)



# # 이미지를 100x100 크기의 블록으로 나누기
# block_size = 50

# # 이미지를 블록 크기의 배수로 제로 패딩
# sharp_sobel = np.pad(sharp_sobel, ((0, block_size - sharp_sobel.shape[0] % block_size), (0, block_size - sharp_sobel.shape[1] % block_size)))
# deblur_sobel = np.pad(deblur_sobel, ((0, block_size - deblur_sobel.shape[0] % block_size), (0, block_size - deblur_sobel.shape[1] % block_size)))

# sharp_blocks = [sharp_sobel[i:i+block_size, j:j+block_size].flatten() for i in range(0, sharp_sobel.shape[0], block_size) for j in range(0, sharp_sobel.shape[1], block_size)]
# deblur_blocks = [deblur_sobel[i:i+block_size, j:j+block_size].flatten() for i in range(0, deblur_sobel.shape[0], block_size) for j in range(0, deblur_sobel.shape[1], block_size)]

# # 각 블록에 대한 Optimal Transport 계산
# total_distance = 0.0


# for sharp_block, deblur_block in tqdm(zip(sharp_blocks, deblur_blocks),ncols = 80, total=len(sharp_blocks), desc="Calculating Optimal Transport"):
#     sharp_block = sharp_block / np.sum(sharp_block)
#     deblur_block = deblur_block / np.sum(deblur_block)

#     M = ot.dist(sharp_block.reshape((len(sharp_block), 1)), deblur_block.reshape((len(deblur_block), 1)))
    
#     G = ot.emd(sharp_block, deblur_block, M)
#     total_distance += np.sum(G * M)

# print(total_distance)