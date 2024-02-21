import torch
from torchvision.transforms import ToPILImage
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision as v
import numpy as np
import os
from tqdm import tqdm


transform = transforms.ToTensor()

def reblur():
    region_size = 128
    unfold_size = region_size + 63


    sharp = Image.open('/raid/joowan/GoPro/test/sharp/000833.png').convert('RGB')
    kernel = Image.open('/raid/joowan/results/GoPro/NAF_DPS/gopro_v5_1/recon/ker_000833.png').convert('L')
    blur = Image.open('/raid/joowan/GoPro/test/blur/000833.png').convert('RGB')

    blur.save('blur.png')
    # kernel.save('debug/kernel.png')
    # sharp.save('debug/sharp.png')

    sharp_tensor = transform(sharp)
    kernel_tensor = transform(kernel)
    
    sharp_tensor = sharp_tensor.unsqueeze(0)
    kernel_tensor = kernel_tensor.unsqueeze(0)

    kernel_tensor = kernel_tensor.unfold(2,64,64).unfold(3,64,64)
    kernel_tensor = kernel_tensor.permute(2,3,0,1,4,5).reshape(60,1,64,64)
    kernel_sum = kernel_tensor.sum(dim=(2,3), keepdim=True)
    kernel_tensor= kernel_tensor / kernel_sum
    B, C, o_H, o_W = sharp_tensor.shape

    pad_h = (o_H % region_size) if o_H % region_size == 0 else region_size - (o_H % region_size)
    pad_w = (o_W % region_size) if o_W % region_size == 0 else region_size - (o_W % region_size)
    sharp_tensor = F.pad(sharp_tensor, (0, pad_w, 0, pad_h), "reflect", 0)
    B, C, H, W = sharp_tensor.shape

    # image: [1, 3, H, W] -> [P, 3, 256, 256] 
    # kernel: [P, 1, 64, 64] 간의 depth-wise Conv 

    b_img = torch.zeros_like(sharp_tensor)
    for i in range(3):
        data=sharp_tensor[:,i:i+1,:,:]
        #data=F.pad(data,(32,31,32,31))
        data = F.pad(data, (32, 31, 32, 31), mode='reflect')
        unfolded = F.unfold(data, kernel_size=(unfold_size, unfold_size), stride=region_size)
        patches = unfolded.permute(0, 2, 1).reshape(1, unfolded.shape[-1], 1, unfold_size, unfold_size).squeeze(0)
        patches = patches.permute(1,0,2,3)

        # depth wise convolutoin 
        output = F.conv2d(patches, kernel_tensor, stride=1, groups=kernel_tensor.shape[0]) 


        output=output.contiguous().view(output.shape[0],output.shape[1], -1) # [1,,ntp, r_s^2]
        output=output.permute(0,2,1)   # [1,r_s^2, ntp]
        output=F.fold(output, output_size=(H,W), kernel_size=region_size, stride=region_size)
        b_img[:,i,:,:]=output



  
    b_img = b_img[:,:,:o_H,:o_W]

    v.utils.save_image(b_img,'reblur.png')

    img1 = Image.open('out.png').convert('RGB')

    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(blur).astype(np.float64)
    

    # MSE 계산
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        # 두 이미지가 완전히 동일한 경우
        return float('inf')

    # 최대 픽셀 값
    max_pixel = 1. if img1.max() <= 1 else 255.

    # PSNR 계산
    psnr = 20. * np.log10(max_pixel / np.sqrt(mse))
    
    print(psnr)

    
def reblur_folder(sharp_path, kernel_path, blur_path):
    region_size = 128
    unfold_size = region_size + 63


    sharp = Image.open(sharp_path).convert('RGB')
    kernel = Image.open(kernel_path).convert('L')
    blur = Image.open(blur_path).convert('RGB')

    sharp_tensor = transform(sharp)
    kernel_tensor = transform(kernel)
    
    sharp_tensor = sharp_tensor.unsqueeze(0)
    kernel_tensor = kernel_tensor.unsqueeze(0)

    kernel_tensor = kernel_tensor.unfold(2,64,64).unfold(3,64,64)
    kernel_tensor = kernel_tensor.permute(2,3,0,1,4,5).reshape(60,1,64,64)
    kernel_sum = kernel_tensor.sum(dim=(2,3), keepdim=True)
    kernel_tensor= kernel_tensor / kernel_sum
    B, C, o_H, o_W = sharp_tensor.shape

    pad_h = (o_H % region_size) if o_H % region_size == 0 else region_size - (o_H % region_size)
    pad_w = (o_W % region_size) if o_W % region_size == 0 else region_size - (o_W % region_size)
    sharp_tensor = F.pad(sharp_tensor, (0, pad_w, 0, pad_h), "reflect", 0)
    B, C, H, W = sharp_tensor.shape

    # image: [1, 3, H, W] -> [P, 3, 256, 256] 
    # kernel: [P, 1, 64, 64] 간의 depth-wise Conv 

    b_img = torch.zeros_like(sharp_tensor)
    for i in range(3):
        data=sharp_tensor[:,i:i+1,:,:]
        data=F.pad(data,(32,31,32,31))
        unfolded = F.unfold(data, kernel_size=(unfold_size, unfold_size), stride=region_size)
        patches = unfolded.permute(0, 2, 1).reshape(1, unfolded.shape[-1], 1, unfold_size, unfold_size).squeeze(0)
        patches = patches.permute(1,0,2,3)

        # depth wise convolutoin 
        output = F.conv2d(patches, kernel_tensor, stride=1, groups=kernel_tensor.shape[0]) 


        output=output.contiguous().view(output.shape[0],output.shape[1], -1) # [1,,ntp, r_s^2]
        output=output.permute(0,2,1)   # [1,r_s^2, ntp]
        output=F.fold(output, output_size=(H,W), kernel_size=region_size, stride=region_size)
        b_img[:,i,:,:]=output



  
    b_img = b_img[:,:,:o_H,:o_W]

    v.utils.save_image(b_img,'reblur.png')

    img1 = Image.open('reblur.png').convert('RGB')

    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(blur).astype(np.float64)
    

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


reblur()

# mother_path = 'results/NAF/gopro_v5_3'
# l = os.listdir(os.path.join(mother_path,'recon'))


# total_psnr = 0

# for folder in tqdm(l, ncols=50):
#     if folder.split('_')[0] == 'ker':
#         kernel_path = os.path.join(mother_path,'recon',folder)
        
#         name = '_'.join(folder.split('_')[1:])

#         blur_path = os.path.join(mother_path,'input',name)
#         sharp_path = f'/raid/joowan/GoPro/test/sharp/{name}'

#         psnr = reblur_folder(sharp_path, kernel_path, blur_path)
#         total_psnr += psnr

# print(total_psnr / 1111)
