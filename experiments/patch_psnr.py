import os
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import glob
import numpy as np

def extract_patches(image, patch_size):

    # Convert PIL image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)


    B,C,H,W= tensor.shape


    new_H=H
    new_W=W
    # Padding
    pad_height = (patch_size - tensor.shape[2] % patch_size) % patch_size
    pad_width = (patch_size - tensor.shape[3] % patch_size) % patch_size
    tensor_padded = F.pad(tensor, (0, pad_width, 0, pad_height))
    print('padded_tensor shape', tensor_padded.shape)

    # Extract patches
    patches = []
    for i in range(0, tensor_padded.shape[2], patch_size):
        for j in range(0, tensor_padded.shape[3], patch_size):
            patches.append(tensor_padded[:, :, i:i+patch_size, j:j+patch_size])

    return patches


def image_to_patch():

    swim_DPS = Image.open('results/people/recon/img_swim_letter.png')
    swim_MSG = Image.open('results/people/recon/msg_swim_letter.png')
    swim_sharp = Image.open('results/people/sharp/swim_letter.png')
    swim_blur = Image.open('results/people/input/swim_letter.png')


    kid_DPS = Image.open('results/people/recon/img_kid.png')
    kid_MSG = Image.open('results/people/recon/msg_kid.png')
    kid_sharp = Image.open('results/people/sharp/kid.png')
    kid_blur = Image.open('results/people/input/kid.png')

    patch_swim_DPS = extract_patches(swim_DPS,256)
    for idx,patch in enumerate(patch_swim_DPS):
        torchvision.utils.save_image(patch,f'debug/swim_dps/{idx}.png')

    patch_swim_MSG = extract_patches(swim_MSG,256)
    for idx,patch in enumerate(patch_swim_MSG):
        torchvision.utils.save_image(patch,f'debug/swim_MSG/{idx}.png')

    patch_kid_DPS = extract_patches(kid_DPS,256)
    for idx,patch in enumerate(patch_kid_DPS):
        torchvision.utils.save_image(patch,f'debug/kid_dps/{idx}.png')

    patch_kid_MSG = extract_patches(kid_MSG,256)
    for idx,patch in enumerate(patch_kid_MSG):
        torchvision.utils.save_image(patch,f'debug/kid_MSG/{idx}.png')

    patch_kid_sharp = extract_patches(kid_sharp,256)
    for idx,patch in enumerate(patch_kid_sharp):
        torchvision.utils.save_image(patch,f'debug/kid_sharp/{idx}.png')

    patch_swim_sharp = extract_patches(swim_sharp,256)
    for idx,patch in enumerate(patch_swim_sharp):
        torchvision.utils.save_image(patch,f'debug/swim_sharp/{idx}.png')

    patch_swim_blur = extract_patches(swim_blur,256)
    for idx,patch in enumerate(patch_swim_blur):
        torchvision.utils.save_image(patch,f'debug/swim_blur/{idx}.png')

    patch_kid_blur = extract_patches(kid_blur,256)
    for idx,patch in enumerate(patch_kid_blur):
        torchvision.utils.save_image(patch,f'debug/kid_blur/{idx}.png')

def kernel_to_patch():
    swim_ker = Image.open('debug/ker_swim_letter.png')
    kid_ker = Image.open('debug/ker_kid.png')

    patch_swim_ker = extract_patches(swim_ker,64)
    patch_kid_ker = extract_patches(kid_ker,64)

    for idx,patch in enumerate(patch_swim_ker):
        torchvision.utils.save_image(patch,f'debug/swim_ker/{idx}.png')
    
    for idx,patch in enumerate(patch_kid_ker):
        torchvision.utils.save_image(patch,f'debug/kid_ker/{idx}.png')


def caculate_psnr():
    l = [0, 2, 4, 6, 8, 20, 22, 24, 26, 28, 40, 42, 44, 46, 48]

    d = dict()
    for idx,index in enumerate(l):
        d[idx] = [index, index+1, index+10, index+11]

    swim_path = 'debug/kid_sharp/'
    swim_sharp = glob.glob('debug/kid_sharp/*.png')

    for img_path in swim_sharp:
        img_name = img_path.split('/')[-1]
        swim_MSG = os.path.join('debug/kid_MSG',img_name)
        swim_dps = os.path.join('debug/kid_dps',img_name)
        swim_MSG_img = np.array(Image.open(swim_MSG)).astype('float')
        swim_dps_img = np.array(Image.open(swim_dps)).astype('float')
        swim_sharp_img = np.array(Image.open(img_path)).astype('float')
        img_num = img_name[:-4]

        pad_index = [10,11,12,13,14]
        if int(img_num) in pad_index:
            swim_MSG_img = swim_MSG_img[:256-48,:,:]
            swim_dps_img = swim_dps_img[:256-48,:,:]
            swim_sharp_img = swim_sharp_img[:256-48,:,:]
 

        MSG_mse = np.mean((swim_MSG_img - swim_sharp_img) ** 2)
        dps_mse = np.mean((swim_dps_img - swim_sharp_img) ** 2)

        MSG_psnr = 20. * np.log10( 255. / np.sqrt(MSG_mse))
        dps_psnr = 20. * np.log10( 255. / np.sqrt(dps_mse))

        
        kernel_index = d[int(img_num)]

        k_imgs =[]
        for i in kernel_index:
            k_imgs.append(Image.open(f'debug/kid_ker/{str(i)}.png').convert('L'))
        
        kernel_imgs = []
        for img in k_imgs:
            debug = img
            debug = np.array(debug)
            rgb_array = np.zeros((debug.shape[0], debug.shape[1], 3), dtype=np.uint8)

            if MSG_psnr > dps_psnr:
                rgb_array = debug # white MSG
            else:
                rgb_array[:,:,0] = debug # RED DPS
            win = abs(MSG_psnr - dps_psnr)
            kernel_imgs.append(rgb_array)

        top_row = np.concatenate((kernel_imgs[0], kernel_imgs[1]), axis=1)
        bottom_row = np.concatenate((kernel_imgs[2], kernel_imgs[3]), axis=1)
        combined_image = np.concatenate((top_row, bottom_row), axis=0)

        # NumPy 배열을 이미지로 변환
        final_image = Image.fromarray(combined_image)
        final_image.save(f'debug/kid_result/{win:.3f}_patch_num:{img_name[:-4]}.png')
    

# image_to_patch()
caculate_psnr()