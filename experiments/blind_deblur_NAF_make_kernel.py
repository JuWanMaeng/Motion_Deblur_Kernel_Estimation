from functools import partial
import os,glob,lpips
import argparse
import yaml
import torch.nn.functional as F

import numpy as np
import torch, glob

import matplotlib.pyplot as plt

from guided_diffusion.blind_condition_methods import get_conditioning_method
from guided_diffusion.measurements import  get_noise
from guided_diffusion.NAF_unet import create_NAF_model
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion_NAFNet import create_sampler
from util.img_utils import clear_color
from util.logger import get_logger
from PIL import Image
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import math
from guided_diffusion.image_datasets import PairImageDataset_NAF
from guided_diffusion.measurements import MSG_operator







def main():
    # logger
    logger = get_logger()

    ## patch size
    region_size = 128
    logger.info(f"kernel region_size is {region_size}")

    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='configs/NAFNet/img_model.yaml')
    parser.add_argument('--kernel_model_config', type=str, default='configs/kernel_model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/motion_deblur_gopro_config.yaml')
    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='/raid/joowan/GoPro/train/kernel')
    # Regularization
    parser.add_argument('--reg_scale', type=float, default=1)
    parser.add_argument('--reg_ord', type=int, default=0, choices=[0, 1])
    parser.add_argument('--down_scale',type=int,default=0)

    
    args = parser.parse_args()
   
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    img_model_config = load_yaml(args.img_model_config)
    kernel_model_config = load_yaml(args.kernel_model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # Kernel configs to namespace save space
    args.kernel = task_config["kernel"]
    args.kernel_size = task_config["kernel_size"]
    args.intensity = task_config["intensity"]
   
    # Load model
    img_model = create_NAF_model(**img_model_config)
    img_model = img_model.to(device)
    img_model.eval()
    kernel_model = create_model(**kernel_model_config)
    kernel_model = kernel_model.to(device)
    kernel_model.eval()
    model = {'img': img_model, 'kernel': kernel_model}

    # Prepare Operator and noise  ### forward method로 blur 와 noise를 주는 객체들
    measure_config = task_config['measurement']
    operator = MSG_operator(device=device, region_size=region_size)
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
    measurement_cond_fn = cond_method.conditioning

    # Add regularization
    # Not to use regularization, set reg_scale = 0 or remove this part.
    regularization = {'kernel': (args.reg_ord, args.reg_scale)}
    measurement_cond_fn = partial(measurement_cond_fn, regularization=regularization)
    if args.reg_scale == 0.0:
        logger.info(f"Got kernel regularization scale 0.0, skip calculating regularization term.")
    else:
        logger.info(f"Kernel regularization : L{args.reg_ord}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop_NAF, model=model, measurement_cond_fn=measurement_cond_fn)
   

    # metric
 
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.to(device)

    # Prepare dataloader

    sharp_files = glob.glob(f'/raid/joowan/GoPro/train/sharp/*.png')



    dataset = PairImageDataset_NAF(image_paths=sharp_files, random_crop=False)
    loader = DataLoader(dataset,batch_size=1, shuffle=False, num_workers=1)
    
    # Working directory
    out_path = args.save_dir
    logger.info(f"work directory is created as {out_path}")
    os.makedirs(out_path, exist_ok=True)


    # set seed for reproduce
    np.random.seed(123)

    error_image=[]
    error_count = 0
    to_pil = ToPILImage()
    
    # Do Inference
    for i, images in enumerate(loader):

        logger.info(f'@@@@@@ sampling {i+1}th image / total:{len(dataset)}imgs @@@@@@' )

        sharp_img = images['sharp'].to(device)
        total_img = images['blur'].to(device)
        sharp_path = images['sharp_path']
        blur_path = images['blur_path']


        orig_H, orig_W = sharp_img.shape[2], sharp_img.shape[3]


        if sharp_path[0].split('/')[-2] == 'gt':   # Realblur-J format
            img_name = '_'.join(blur_path[0].split('/')[-3:])  # ex) 'scene230_blur_blur_7.png'

        else:  # gopro format

            img_name = sharp_path[0].split('/')[-1][:-4]
        
        if os.path.exists(os.path.join(out_path,img_name+'.png')):
            print('exist!')
            continue

        y = total_img

        B,C,H,W = y.shape
        pad_h = (H % region_size) if H % region_size == 0 else region_size - (H % region_size)
        pad_w = (W % region_size) if W % region_size == 0 else region_size - (W % region_size)
        y = F.pad(y, (0, pad_w, 0, pad_h), mode='reflect')
        
        kernel_batch = (y.shape[2]//region_size) * (y.shape[3]//region_size)

        fname = str(i).zfill(5) + '.png'

        model['img'].NAFNet_MSG_modules.feed_data(data={'lq':y })
        x_prime , features = model['img'].NAFNet_MSG_modules.inference()


        x_start = {'img': torch.randn(y.shape, device=device).requires_grad_(),
                'kernel': torch.randn((kernel_batch,1,args.kernel_size,args.kernel_size), device=device).requires_grad_()}
        
        # sample 
        sample = sample_fn(x_start=x_start, measurement=y.to(device), features = features, x_prime = x_prime, record = False)
        

        result_image = sample['img'][:,:,:H,:W]
        result_kernel = sample['kernel']


    
        # caculate LPIPS tensor 
        lpips_score = loss_fn_alex(sharp_img, result_image)
        if lpips_score ==float('-inf'):

            print(f'inf !!! file name is: {img_name}')
            error_image.append(img_name)
            error_count +=1

        # Stitch the kernels
        stitched_kernel = stitch_patches(result_kernel, region_size, H, W) # 1,192,320
        kernel_image = to_pil(stitched_kernel.cpu().squeeze(0))
        kernel_image.save(os.path.join(out_path, img_name +'.png'))




        


def compute_psnr(img1, img2):
    img1 = (img1 * 255.0).clamp(0, 255).to(torch.float64).permute(0,2,3,1).contiguous().cpu().numpy()
    img2 = (img2 * 255.0).clamp(0, 255).to(torch.float64).permute(0,2,3,1).contiguous().cpu().numpy()

    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
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

def tensor_to_numpy(tensor):
    batch = tensor.shape[0]
    tensor = (tensor * 255.0).clamp(0, 255).to(torch.uint8)
    tensor = tensor.permute(0, 2, 3, 1)
    tensor = tensor.contiguous()

    if batch == 1:
        tensor = tensor.cpu().numpy()[0]
    else:
        tensor = tensor.cput().numpy()

    return tensor


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def stitch_patches(patches,patch_size, h,w):
    """
    Stitch patches to form a single image.

    :param patches: List of patches
    :return: Stitched image of shape 1x3x720x1280 (or larger if padded)
    """
    patches = ((patches + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    rows = []

    # Determine number of patches in width and height direction
    patches_height = math.ceil(h / patch_size)
    patches_width = math.ceil(w / patch_size)
    
    for i in range(patches_height):
        row = torch.cat(tuple(patches[i*patches_width : (i+1)*patches_width]), dim=2)
        rows.append(row)
    
    stitched_tensor = torch.cat(rows, dim=1) # concat rows (along height)

    # Extract the original region (without padding)
    return stitched_tensor


 

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    main()
