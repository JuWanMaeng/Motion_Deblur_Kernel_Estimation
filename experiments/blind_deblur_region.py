from functools import partial
import os
import argparse
import yaml
import torch.nn.functional as F

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.blind_condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_operator, get_noise
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion_origin import create_sampler
from data.dataloader import get_dataset, get_dataloader
from motionblur.motionblur import Kernel
from util.img_utils import Blurkernel, clear_color
from util.logger import get_logger
import cv2, math, time


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def extract_patches(tensor, patch_size,downscale):
    """
    Extract patch_sizexpatch_size patches from a 1x3x720x1280 tensor.

    :param tensor: Input tensor of shape 1x3x720x1280
    :return: List of patch_sizexpatch_size patches
    """

    B,C,H,W= tensor.shape

    # Rescale tensor if downscale is provided
    if downscale!=0:
        new_H = int(H * (1 / downscale))
        new_W = int(W * (1 / downscale))
        tensor = F.interpolate(tensor, size=(new_H, new_W), mode='bilinear', align_corners=True)
    else:
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

    return patches, B,C ,new_H, new_W, 

def stitch_patches(patches,h,w):
    """
    Stitch patches to form a single image.

    :param patches: List of patches
    :return: Stitched image of shape 1x3x720x1280 (or larger if padded)
    """

    rows = []

    # Determine number of patches in width and height direction
    patches_height = math.ceil(h / 256)
    patches_width = math.ceil(w / 256)
    
    for i in range(patches_height):
        row = torch.cat(patches[i*patches_width : (i+1)*patches_width], dim=3) # concat along width
        rows.append(row)
    
    stitched_tensor = torch.cat(rows, dim=2) # concat rows (along height)

    # Extract the original region (without padding)
    return stitched_tensor[:, :, :h, :w]





def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='configs/256_imagenet_model_config.yaml')
    parser.add_argument('--kernel_model_config', type=str, default='configs/kernel_model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/motion_deblur_region_config.yaml')
    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/ImageNet/16_cos_schedule')
    # Regularization
    parser.add_argument('--reg_scale', type=float, default=0.1)
    parser.add_argument('--reg_ord', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=64)
    parser.add_argument('--region_size', type=int, default=16)
    
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
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
    img_model = create_model(**img_model_config)
    img_model = img_model.to(device)
    img_model.eval()
    kernel_model = create_model(**kernel_model_config)
    kernel_model = kernel_model.to(device)
    kernel_model.eval()
    model = {'img': img_model, 'kernel': kernel_model}

    # Prepare Operator and noise  ### forward method로 blur 와 noise를 주는 객체들
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
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
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])

    logger.info(f"work directory is created as {out_path}")
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # set seed for reproduce
    np.random.seed(123)
    
    kernel_size=args.kernel_size
    region_size=args.region_size

    
    # Do Inference
    for i, total_img in enumerate(loader):
        kernel_results=[]
        logger.info(f'@@@@@@ sampling {i+1}th image / total:{len(dataset)}imgs @@@@@@' )


        result_imgs=[]
        #result_kernels=[]

        total_img=total_img.to(device)
        y = total_img
        y_n = noiser(y)   ### y * noise(0.02)   ###

        _,C,H,W = y_n.shape
        img_patches,B,C,H,W=extract_patches(y_n, 256, downscale=2)

        fname = loader.dataset.fpaths[i].split('/')[-1]
        kernel_result_root='/raid/joowan/blur_kernel_result/ImageNet_16_cos_schedule'
        kernel_result_folder=os.path.join(kernel_result_root,fname)

        if not os.path.exists(kernel_result_folder):
            os.makedirs(kernel_result_folder)


        debug=False
        # sampling patch by patch
        for j in range(len(img_patches)):
            
            if i ==0:
                debug=True
                print('debug True')
            else:
                debug=False
                print('debug False')

            logger.info(f"Inference for {i}_image {j}/{len(img_patches)} patch ")
            
            
            _,C,h,w = img_patches[j].shape
            x_start = {'img': torch.randn(img_patches[j].shape, device=device).requires_grad_(),
                    'kernel': torch.randn(( (h//region_size) * (w//region_size) ,1,kernel_size,kernel_size), device=device).requires_grad_()}
            
            # !prior check: keys of model (line 74) must be the same as those of x_start to use diffusion prior.
            for k in x_start:
                if k in model.keys():
                    logger.info(f"{k} will use diffusion prior")
                else:
                    logger.info(f"{k} will use uniform prior.")
        
            # sample 
            start_time=time.time()

            sample = sample_fn(x_start=x_start, measurement=img_patches[j].to(device), record=True, save_root=out_path, debug=debug, patch_number=j, img_name = fname)

            sampling_time=time.time()-start_time
            logger.info(f"samping {j}th/{len(img_patches)} PATCH for {sampling_time:.1f}sec")


            
            # stack samples
            result_imgs.append(sample['img'])
            kernel_results.append(sample['kernel'].cpu().numpy())
        kernel_result=np.concatenate(kernel_results, axis=0)
        np.save(f'{kernel_result_folder}/kernel.npy', kernel_result)


        # # Stitch the images
        stitched_image = stitch_patches(result_imgs,H,W)
        plt.imsave(os.path.join(out_path, 'recon', 'img_'+fname), clear_color(stitched_image))

        # # Stitch the kernels
        # stitched_kernel = stitch_patches(result_kernels,H,W)
        # plt.imsave(os.path.join(out_path, 'recon', 'ker_'+fname), clear_color(stitched_kernel))

        # save original blur image
        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(total_img))
        


 

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    main()
