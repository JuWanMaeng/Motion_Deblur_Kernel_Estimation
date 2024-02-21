from functools import partial
import os
import argparse
import yaml

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
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
from torchvision.transforms.functional import crop


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def padding_tensor(tensor,p_size):
    # Padding
    pad_height = (p_size - tensor.shape[2] % p_size) % p_size
    pad_width = (p_size - tensor.shape[3] % p_size) % p_size
    tensor_padded = torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height))

    print('padded tensor shape:', tensor_padded.shape)

    return tensor_padded


def extract_patches(tensor,p_size,r):
    """
    Extract p_sizexp_size patches from a 1x3x720x1280 tensor.

    :param tensor: Input tensor of shape 1x3x720x1280
    :param p_size: patch size
    :param r: overlapping gap


    :return: List of p_sizexp_size patches
    """


    _,C,h,w = tensor.shape

    h_list = [i for i in range(0, h - p_size + 1, r)]
    w_list = [i for i in range(0, w - p_size+ 1, r)]

    corners = [(i, j) for i in h_list for j in w_list]  # [(0,0), (0,64), (0,128), ... (  ,  )]
    x_grid_mask = torch.zeros_like(tensor, device=tensor.device)

    for (hi, wi) in corners:
        x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1


    x_patch = torch.cat([crop(tensor, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
    print(x_patch.shape)

    return x_patch, x_grid_mask, corners
            
    




def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='configs/model_gopro_config.yaml')
    parser.add_argument('--kernel_model_config', type=str, default='configs/kernel_model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_patch_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/motion_deblur_gopro_config.yaml')
    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    # Regularization
    parser.add_argument('--reg_scale', type=float, default=0.1)
    parser.add_argument('--reg_ord', type=int, default=0, choices=[0, 1])
    
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

    batching=4
    p_size=256
    r=64
    
    # Do Inference
    for i, total_img in enumerate(loader):
        fname = str(i).zfill(5) + '.png'
        logger.info(f'@@@@@@ sampling {i+1}th image / total:{len(dataset)}imgs @@@@@@' )
        _,C,H,W=total_img.shape


        total_img=total_img.to(device)
        y = total_img
        y_n = noiser(y)  # [1,3,720,1280]

        padded_img=padding_tensor(y_n,p_size=p_size)  # [1,3,768,1280]
        _,c,h,w=padded_img.shape
        
        
        # sampling patch by batches
        x_start = {'img': torch.randn((padded_img.shape), device=device),
                   'kernel': torch.randn((15,1,64,64), device=device)}
            
        for k in x_start:
            if k in model.keys():
                logger.info(f"{k} will use diffusion prior")
            else:
                logger.info(f"{k} will use uniform prior.")

        
        # sample_fn= diffusion model
        sample = sample_fn(x_start=x_start, measurement=y_n.to(device), record=False, save_root=out_path)

        

        batch_end_time=time.time()
        logger.info(f"samping {i}th IMAGE for {batch_end_time:.1f}sec")

        # Stitch the images
        result_image = sample['img'][:,:,:H,:W]
        plt.imsave(os.path.join(out_path, 'recon', 'img_'+fname), clear_color(result_image))

        # kernel images
        grid = make_grid(sample['kernel'], nrow=9, normalize=True, pad_value=1.0) # normalize=True will map values in [0,1]
        save_image(grid, os.path.join(out_path, 'recon', 'ker_'+fname))

        # save original blur image
        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(total_img))
        


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    main()
