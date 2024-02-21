from guided_diffusion.msg_unet2 import UNet_MSG,Multiscale_Structure_Guidance, ResBlock, Upsample, Downsample, MSG_Resblock, Guided_ResBlock,TimestepEmbedSequential
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

import torch
from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import functools,cv2

if __name__ == '__main__':
    device = torch.device(1)
    # 16x3x1280x720 크기의 랜덤 텐서 생성
    tensor = torch.rand(1, 3, 1280, 720).to(device)
    y = torch.rand(1,3,1280,720).to(device)
    guidance_ch = 32

    
    # Conv = conv_nd(2, 3, guidance_ch, kernel_size=3, padding=1)
    # Resblock = MSG_Resblock(guidance_ch,guidance_ch*2,dims=2)
    # MSG_module = Multiscale_Structure_Guidance(scale=2,guidance_ch=32)
    # downsampler = Downsample(dims=2)
    # res_encoder = TimestepEmbedSequential(Guided_ResBlock(channels=2*guidance_ch, emb_channels=guidance_ch*4, out_channels=guidance_ch*3, guidance_ch=guidance_ch))
    
    # upsampler = Upsample(dims=2)
    # res_decoder = ResBlock(channels=3*guidance_ch, emb_channels=4*guidance_ch, out_channels=guidance_ch )



    # time_embed = nn.Sequential(
    #     linear(guidance_ch,guidance_ch*4),
    #     nn.SiLU(),
    #     linear(guidance_ch *4, guidance_ch * 4),
    #     )
    # emb = time_embed(timestep_embedding(torch.tensor([999]), guidance_ch))


    # x_prime, feature = MSG_module(tensor)

    # x = Conv(tensor)
    # x = Resblock(x)

    # x = downsampler(x)  # [2,64,640,360]
    # x = res_encoder(x,emb,feature)

    # x = upsampler(x)
    # x = res_decoder(x,emb)

    # x = conv_nd(2,guidance_ch,3,kernel_size=3, padding=1)(x)

    # print(x.shape)


    Unet = UNet_MSG(in_channels=6, guidance_channels=guidance_ch).to(device)
    timestep = torch.tensor([999]).to(device)
    output = Unet(tensor, timestep, y )