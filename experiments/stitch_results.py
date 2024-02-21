import os, glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import lpips
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch as th
import os
from tqdm import tqdm

output = glob.glob('results/average/myselect2/*.png')
output = sorted(output,reverse=True)
output = output[:100]

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

def caculate_PSNR(sharp_array, out_array):


    img1 = sharp_array.astype(np.float64)
    img2 = out_array.astype(np.float64)

    

    # MSE 계산
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        # 두 이미지가 완전히 동일한 경우
        return float('inf')

    # 최대 픽셀 값
    max_pixel = 1. if img1.max() <= 1 else 255.

    # PSNR 계산
    psnr = 20. * np.log10(max_pixel / np.sqrt(mse))
    
    # print(psnr)
    return psnr

def caculate_diff(img1, img2):

    # numpy 배열로 변환
    img1_np = np.array(img1)/255.0
    img2_np = np.array(img2)/255.0

    # 두 이미지 간의 절대 차이 계산
    diff = np.abs(img1_np - img2_np) * 255
    diff = diff.astype('uint8')


    return diff



for out in tqdm(output):
    img  = np.zeros((1440,2560,3))
    print(out)
    img_name = out.split('/')[-1].split('_')[-1]
    score = out.split('/')[-1].split('_')[0]
    blur_img = f'/raid/joowan/Data/myselect/blur/{img_name}'
    sharp_img = f'/raid/joowan/Data/myselect/sharp/{img_name}'
    # kernel_img = f'results/NAF/GoPro_train/recon/ker_{img_name}'
    # UFP_img = f'/raid/joowan/results/GoPro/UFPDeblur/{img_name}'

    blur = Image.open(blur_img).convert('RGB')
    sharp = Image.open(sharp_img).convert('RGB')
    # kernel_img_copy = Image.open(kernel_img).convert('L')
    # kernel_img_copy.save(f'/raid/joowan/GoPro/train/hard_kernel/{img_name}')
    # blur.save(f'/raid/joowan/GoPro/train/hard_kernel/{img_name}')
    # sharp.save(f'/raid/joowan/GoPro/train/hard_kernel/{img_name}')

    
    blur_array = np.array(blur)
    sharp_array = np.array(sharp)
    # kernel_array = np.array(Image.open(kernel_img).convert('RGB').resize((1280,720)))
    out_array = np.array(Image.open(out).convert('RGB'))
    diff_array = caculate_diff(sharp_array, out_array)

    

    img[:720,:1280,:] = out_array
    img[:720,1280:,:] = blur_array
    img[720:,:1280,:] = sharp_array
    img[720:, 1280:,:] = diff_array


    img = img.astype(np.uint8)
    img = Image.fromarray(img)

    psnr = round(caculate_PSNR(sharp_array, out_array), 2)

    img.save(f'results/average/my_select_UFP_stitch/{score}_{psnr}_{img_name}')

    


