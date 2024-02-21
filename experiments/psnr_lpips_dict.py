import lpips
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch as th
import os
from tqdm import tqdm
from basicsr.metrics.niqe import calculate_niqe
import json

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




mother_path = '/raid/joowan/results/GoPro/NAFNet'
l = sorted(os.listdir(mother_path))


loss_fn = lpips.LPIPS(net='alex').cuda()  # GPU를 사용할 경우 .cuda()를 추가


di = dict()
total_lpips = 0
total_psnr = 0

for img_name in tqdm(l):

    output_img_path = os.path.join(mother_path,img_name)
    sharp_img_path = os.path.join('/raid/joowan/GoPro/test/sharp', img_name)


    
    lpips_score = caculate_LPIPS(sharp_img_path, output_img_path, loss_fn)
    psnr_score = caculate_PSNR(sharp_img_path, output_img_path)

    total_lpips += lpips_score
    total_psnr += psnr_score
    
    # di[img_name] = [round(lpips_score,4), round(psnr_score,2)]



    # with open('scores/metric-ours.json', 'w') as file:
    #     json.dump(di, file, indent=4)

print(total_psnr / len(l))
print(total_lpips / len(l))