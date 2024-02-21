import torch.nn.functional as F
from PIL import Image,ImageOps
from torchvision import transforms
import glob, torch
import torchvision
import numpy as np
from PIL import Image
import os
def replicate_kernel(kernel,orig_H,orig_W, name):
    # 영역 크기 (64, 64)
    patch_size = (64, 64)

    # 스트라이드 (stride) 크기 (64, 64)
    stride = (64, 64)
    kernel_size = 19

    C,H,W = kernel.shape

    row = H//patch_size[0]
    col = W//patch_size[0]
    num_total_patches = row * col

        # 패치 크기
    patch_height = 19
    patch_width = 19

    # 이어붙일 이미지의 크기
    output_height = 6 * patch_height
    output_width = 10 * patch_width


    tensor = kernel.unfold(1, patch_size[0], stride[0]).unfold(2,patch_size[1],stride[1]).reshape(1,-1,patch_size[0],patch_size[1]).permute(1,0,2,3)
    tensor = tensor[:,:,22:41,22:41]
    output_image = np.zeros((114,190))

    for i in range(6):
        for j in range(10):
            patch = tensor[i * 10 + j, 0, :, :]
            output_image[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] = patch


    return output_image




k = '/raid/joowan/GoPro/train/kernel/001822.png'
H,W = 720,1280


kernel_name = k.split('/')[-1][:-4]
kernel_image = Image.open(k)
kernel_image = kernel_image.convert('L')
kernel_tensor = transforms.ToTensor()(kernel_image)
kernel_tensor = kernel_tensor
output_image = replicate_kernel(kernel_tensor,H,W,kernel_name)
output_image = (output_image * 255).astype(np.uint8)  # 이미지는 0~255 범위의 정수값이어야 함
output_image = Image.fromarray(output_image)

save_path = os.path.join('/raid/joowan/GoPro/train/kernel22', kernel_name+'.png')
# 이미지 저장
output_image.save(save_path)








# kernel_path = glob.glob('/raid/joowan/GoPro/train//kernel_origin/*.png')
# H,W = 720,1280

# for k in kernel_path:
#     kernel_name = k.split('/')[-1][:-4]
#     kernel_image = Image.open(k)
#     kernel_image = kernel_image.convert('L')
#     kernel_tensor = transforms.ToTensor()(kernel_image)
#     kernel_tensor = kernel_tensor
#     output_image = replicate_kernel(kernel_tensor,H,W,kernel_name)
#     output_image = (output_image * 255).astype(np.uint8)  # 이미지는 0~255 범위의 정수값이어야 함
#     output_image = Image.fromarray(output_image)

#     save_path = os.path.join('/raid/joowan/GoPro/train/kernel', kernel_name+'.png')
#     # 이미지 저장
#     output_image.save(save_path)



