import lpips
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch as th
from basicsr.metrics.niqe import calculate_niqe

def caculate_LPIPS(sharp_path, blur_path):
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
    img1 = img1.unsqueeze(0)  # 첫 번째 이미지에 대해 배치 차원 추가
    img2 = img2.unsqueeze(0)  # 두 번째 이미지에 대해 배치 차원 추가
 


    # LPIPS 모델 초기화 (VGG를 사용하는 버전을 예로 듭니다)
    loss_fn = lpips.LPIPS(net='alex')  # GPU를 사용할 경우 .cuda()를 추가

    # 두 이미지 간의 LPIPS 스코어 계산
    with torch.no_grad():  # 그라디언트 계산을 비활성화
        lpips_score = loss_fn(img1, img2)  # GPU를 사용할 경우 .cuda()를 호출

    # print('LPIPS Score:', lpips_score.item())
    return lpips_score.item()


def caculate_PSNR(sharp_path, blur_path):
    img1 = Image.open(sharp_path).convert('RGB')
    img2 = Image.open(blur_path).convert('RGB')

    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)

    

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
    # 이미지를 열고 그레이스케일로 변환
    img1 = Image.open(img1).convert('RGB')
    img2 = Image.open(img2).convert('RGB')

    # numpy 배열로 변환
    img1_np = np.array(img1)/255.0
    img2_np = np.array(img2)/255.0

    # 두 이미지 간의 절대 차이 계산
    diff = np.abs(img1_np - img2_np) * 255
    

    # 차이를 이미지로 변환
    diff_img = Image.fromarray(diff.astype('uint8'))

    return diff_img


id = '000366'
sharp_path = f'/raid/joowan/GoPro/test/sharp/{id}.png'
blur_path = f'/raid/joowan/GoPro/test/blur/{id}.png'
NAF_diffusion_path = f'../guided-diffusion/aa.png'
# NAF_diffusion_path =  f'/raid/joowan/Data/000366/recon/img_{id}.png'


lpips_score = caculate_LPIPS(sharp_path, NAF_diffusion_path)
psnr_score = caculate_PSNR(sharp_path, NAF_diffusion_path)
# diff_img = caculate_diff(sharp_path, NAF_diffusion_path)


print(f'psnr:{round(psnr_score,3)}, lpips:{round(lpips_score,4)}')
# caculate_LPIPS(sharp_path, ufp_path)
# caculate_PSNR(sharp_path, ufp_path)