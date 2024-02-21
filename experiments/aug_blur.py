import os, glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torchvision as v
from tqdm import tqdm


transform = transforms.ToTensor()

img_path ='/raid/joowan/GoPro/train/sharp/*.png'
index = [i for i in range(0,2103,50)]
imgs= np.array(sorted(glob.glob(img_path)))
imgs = imgs[index]

hard_kernels = sorted(glob.glob('/raid/joowan/GoPro/train/hard_kernel/*.png'))[:50]
aug_blur_path = '/raid/joowan/GoPro/train/aug_blur'

region_size = 128
unfold_size = region_size + 63
count = 1


for img in tqdm(imgs):
    sharp = Image.open(img).convert('RGB')
    img_name = img.split('/')[-1]

    for ker in hard_kernels:
        kernel = Image.open(ker).convert('L')

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



    
        reblur = b_img[:,:,:o_H,:o_W]
        v.utils.save_image(reblur, f'/raid/joowan/GoPro/train/aug_blur/{count:06}.png')
        sharp.save(f'/raid/joowan/GoPro/train/aug_sharp/{count:06}.png')

        count+=1



    


