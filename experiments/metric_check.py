import lpips
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch as th
import os
from tqdm import tqdm
from basicsr.metrics.niqe import calculate_niqe

def caculate_LPIPS(sharp_path, blur_path, loss_fn):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

    # 이미지 로드 (예: 'image1.png' 및 'image2.png')
    img1 = Image.open(sharp_path).convert('RGB')
    img2 = Image.open(blur_path).convert('RGB')

    # 이미지 변환
    img1 = transform(img1)
    img2 = transform(img2)

    # 배치 차원 추가 (LPIPS는 배치를 기반으로 계산)
    img1 = img1.unsqueeze(0).to(device='cuda')  # 첫 번째 이미지에 대해 배치 차원 추가
    img2 = img2.unsqueeze(0).to(device='cuda')  # 두 번째 이미지에 대해 배치 차원 추가


    # 두 이미지 간의 LPIPS 스코어 계산
    with torch.no_grad():  # 그라디언트 계산을 비활성화
        lpips_score = loss_fn(img1, img2)  # GPU를 사용할 경우 .cuda()를 호출

    return lpips_score.item()
 

def caculate_PSNR(sharp_path, blur_path):
    img1 = Image.open(sharp_path).convert('RGB')
    img2 = Image.open(blur_path).convert('RGB')

    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)
    
    # niqe = calculate_niqe(img2,crop_border=1)
    niqe = 0

    # MSE 계산
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        # 두 이미지가 완전히 동일한 경우
        return float('inf')

    # 최대 픽셀 값
    max_pixel = 1. if img1.max() <= 1 else 255.

    # PSNR 계산
    psnr = 20. * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def calculate_niqe_from_image(img, crop_border):
    img = np.array(Image.open(img).convert('L'))
    score = calculate_niqe(img,crop_border=crop_border)
    return score


mother_path = '/raid/joowan/results/HIDE/NAF_DPS/HIDE_1'
l = os.listdir(mother_path)

total_psnr = 0
total_lpips = 0


total_avg_psnr = 0
total_avg_lpips = 0

count = 0
# LPIPS 모델 초기화 (VGG를 사용하는 버전을 예로 듭니다)
loss_fn = lpips.LPIPS(net='alex').cuda()  # GPU를 사용할 경우 .cuda()를 추가



for folder in tqdm(l):

    output_img_path = os.path.join(mother_path,folder)
    
    name = folder
    sharp_img_path = os.path.join('/raid/joowan/Data/HIDE/sharp', name)


    psnr = []
    lpips_score = []

    for i in range(1,3):
        output_path = os.path.join(f'/raid/joowan/results/HIDE/NAF_DPS/HIDE_{i}/{name}')  # default blindDPS
        lpips_score.append([caculate_LPIPS(sharp_img_path, output_path, loss_fn), caculate_PSNR(sharp_img_path, output_path)])

    best_scores = min(lpips_score, key=lambda x:x[0])

    avg_psnr = (lpips_score[0][1] + lpips_score[1][1] ) / 2
    avg_lpips = (lpips_score[0][0] + lpips_score[1][0] ) / 2


    total_avg_psnr += avg_psnr
    total_avg_lpips += avg_lpips
    # total_avg_niqe += avg_niqe


    total_psnr += best_scores[1]
    total_lpips += best_scores[0]
    # total_niqe += best_scores[2]
    count+=1
        
        

print('psnr', total_psnr / count)
print('lpips', total_lpips / count)

print('avg_psnr', total_avg_psnr / count)
print('avg_lpips', total_avg_lpips / count)


print(count)
