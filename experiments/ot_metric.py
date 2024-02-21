import cv2
import numpy as np
import ot
from tqdm import tqdm, trange
import os, glob, pickle, json

sharp_folder = sorted(glob.glob('/raid/joowan/GoPro/test/sharp/*.png'))
# deblur_folder = glob.glob('/raid/joowan/results/GoPro/Uformer/*.png')
deblur_folder = sorted(glob.glob('/raid/joowan/results/GoPro/UFPDeblur/*.png'))
# deblur_folder = sorted(glob.glob('/raid/joowan/results/GoPro/NAF_DPS/best/*.png'))

total_distance = 0.0
di = {}

with tqdm(deblur_folder, ncols=80, desc="Calculating Optimal") as pbar:
    for img in pbar:
        img_distance = 0
        img_name = img.split('/')[-1]

        # 이미지 불러오기
        sharp_image = cv2.imread(f'/raid/joowan/GoPro/test/sharp/{img_name}', cv2.IMREAD_GRAYSCALE)
        deblur_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        # Sobel 필터 적용
        sharp_sobel = cv2.Sobel(sharp_image, cv2.CV_64F, 1, 1, ksize=3)
        deblur_sobel = cv2.Sobel(deblur_image, cv2.CV_64F, 1, 1, ksize=3)

        # 음수 값을 0으로 클리핑
        sharp_sobel = np.maximum(sharp_sobel, 0)
        deblur_sobel = np.maximum(deblur_sobel, 0)

        # 이미지를 50x50 크기의 블록으로 나누기
        block_size = 50
        # 이미지를 블록 크기의 배수로 제로 패딩
        sharp_sobel = np.pad(sharp_sobel, ((0, block_size - sharp_sobel.shape[0] % block_size), (0, block_size - sharp_sobel.shape[1] % block_size)))
        deblur_sobel = np.pad(deblur_sobel, ((0, block_size - deblur_sobel.shape[0] % block_size), (0, block_size - deblur_sobel.shape[1] % block_size)))

        sharp_blocks = [sharp_sobel[i:i+block_size, j:j+block_size].flatten() for i in range(0, sharp_sobel.shape[0], block_size) for j in range(0, sharp_sobel.shape[1], block_size)]
        deblur_blocks = [deblur_sobel[i:i+block_size, j:j+block_size].flatten() for i in range(0, deblur_sobel.shape[0], block_size) for j in range(0, deblur_sobel.shape[1], block_size)]

        for sharp_block, deblur_block in zip(sharp_blocks, deblur_blocks):
            if np.sum(sharp_block) > 0 and np.sum(deblur_block)>0:
                sharp_block = sharp_block / (np.sum(sharp_block) + 1e-10)
                deblur_block = deblur_block / (np.sum(deblur_block) + 1e-10)

                M = ot.dist(sharp_block.reshape((len(sharp_block), 1)), deblur_block.reshape((len(deblur_block), 1)))
                
                G = ot.emd(sharp_block, deblur_block, M,numItermax=500000)
                img_distance += np.sum(G * M)
          
        # 진행 확인용
        di[img_name] = round(img_distance * 10000,3)
        with open('UFP.json', 'w') as file:
            json.dump(di, file, indent=4)

        total_distance += img_distance


print(total_distance/ len(sharp_folder))
print(total_distance/ len(sharp_folder) * 10000)

