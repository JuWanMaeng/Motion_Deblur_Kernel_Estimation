import cv2
import numpy as np
import ot
from tqdm import tqdm, trange
import os, glob, pickle, json
from scipy.stats import wasserstein_distance


sharp_folder = sorted(glob.glob('/raid/joowan/GoPro/test/sharp/*.png'))
# deblur_folder = glob.glob('/raid/joowan/results/GoPro/Uformer/*.png')
# deblur_folder = sorted(glob.glob('/raid/joowan/results/GoPro/UFPDeblur/*.png'))
deblur_folder = sorted(glob.glob('/raid/joowan/results/GoPro/NAF_DPS/best/*.png'))
# deblur_folder = sorted(glob.glob('/raid/joowan/results/GoPro/UFPDeblur/*.png'))

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

        # # 음수 값을 0으로 클리핑
        # sharp_sobel = np.maximum(sharp_sobel, 0)
        # deblur_sobel = np.maximum(deblur_sobel, 0)

        # Flatten the 2D arrays
        sharp_sobel_flat = sharp_sobel.flatten()
        deblur_sobel_flat = deblur_sobel.flatten()

        # Calculate Wasserstein distance
        distance = wasserstein_distance(sharp_sobel_flat, deblur_sobel_flat)
        di[img_name] = round(distance,2)
        
        with open('UFP.json', 'w') as file:
            json.dump(di, file, indent=4)

        total_distance += distance


print(total_distance/ len(sharp_folder))


