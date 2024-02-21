import lpips
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch as th
import os
from tqdm import tqdm
from basicsr.metrics.niqe import calculate_niqe

def calculate_niqe_from_image(img, crop_border):
    img = np.array(Image.open(img).convert('L'))
    score = calculate_niqe(img,crop_border=crop_border,convert_to='gray')
    return score


mother_path = 'results/NAF/gopro_v5_3/recon'
l = os.listdir(mother_path)

total_niqe = 0
total_avg_niqe = 0
count = 0


for folder in tqdm(l):
    no_pair = False
    if folder.split('_')[0] == 'img':
        output_img_path = os.path.join(mother_path,folder)
        
        name = '_'.join(folder.split('_')[1:])

        scores = []
        for i in range(3):
            output_path = os.path.join(f'results/NAF/gopro_v5_{i+1}/recon', f'img_{name}')  # default blindDPS
            if not os.path.exists(output_path):
                no_pair = True
                break
            score = calculate_niqe_from_image(output_path,crop_border=0)[0]
            scores.append(score)
        if no_pair:
            continue

        best_score = min(scores)
        avg_score = sum(scores) / 3


        count+=1
        
        
print('best niqe', best_score)
print('avg niqe', avg_score)
print(count)
