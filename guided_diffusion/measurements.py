'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel
import torch.nn as nn


from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m, perform_tilt
from einops import rearrange


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)

@register_operator(name='blind_blur')
class BlindBlurOperator(LinearOperator):
    def __init__(self, device, **kwargs) -> None:
        self.device = device
    
    def forward(self, data, kernel, **kwargs):
        return self.apply_kernel(data, kernel)

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel):
        #TODO: faster way to apply conv?:W
        
        b_img = torch.zeros_like(data).to(self.device)
        for i in range(3):
            b_img[:, i, :, :] = F.conv2d(data[:, i:i+1, :, :], kernel, padding='same')
        return b_img
    

class MSG_operator(LinearOperator):
    def __init__(self, device, **kwargs) -> None:
        self.device = device
        self.region_size = kwargs['region_size']

    def forward(self, data, kernel, **kwargs):
        if 'only_kernel' in kwargs.keys():
            return self.apply_kernel_msg2(data,kernel)
        elif kernel.shape[0] > 60 :
            return self.apply_kernel_overlap(data,kernel)
        else:
            return self.apply_kernel_msg(data,kernel)


    def transpose(self, data, **kwargs):
        return data

    
    def apply_kernel_msg(self, input, kernel):

        B, C, H, W = input.size()

        # image: [1, 3, H, W] -> [P, 3, 256, 256] 
        # kernel: [P, 1, 64, 64] 간의 depth-wise Conv 
        region_size = self.region_size
        unfold_size = region_size + 63

        b_img = torch.zeros_like(input)
        for i in range(3):
            data=input[:,i:i+1,:,:]
            # data=F.pad(data,(32,31,32,31))
            data = F.pad(data, (32, 31, 32, 31), mode='reflect')
            unfolded = F.unfold(data, kernel_size=(unfold_size, unfold_size), stride=region_size)  # 1,36481,60
            patches = unfolded.permute(0, 2, 1).reshape(1, unfolded.shape[-1], 1, unfold_size, unfold_size).squeeze(0)  # 60,1,191,191
            patches = patches.permute(1,0,2,3)                                                                          # 1,60,191,191

            # depth wise convolutoin 
            output = F.conv2d(patches, kernel, stride=1, groups=kernel.shape[0])                                       # 1, 60, 191, 191 * 60, 1, 64, 64 -> 1, 60, 128, 128   


            output=output.contiguous().view(output.shape[0],output.shape[1], -1) # [1,,ntp, r_s^2]
            output=output.permute(0,2,1)   # [1,r_s^2, ntp]
            output=F.fold(output, output_size=(H,W), kernel_size=region_size, stride=region_size)
            b_img[:,i,:,:]=output
   

        return b_img

    def apply_kernel_overlap(self, input, kernel):

        input = input.unfold(2,128,64).unfold(3,128,64)              # 1,3,11,19,128,128
        input = input.permute(0,1,2,4,3,5)  
        input = input.reshape(1,3,11*128,19*128)

        B, C, H, W = input.size()

        # image: [1, 3, H, W] -> [P, 3, 256, 256] 
        # kernel: [P, 1, 64, 64] 간의 depth-wise Conv 
        region_size = self.region_size
        unfold_size = region_size + 63

        b_img = torch.zeros_like(input)
        for i in range(3):
            data=input[:,i:i+1,:,:]
            # data=F.pad(data,(32,31,32,31))
            data = F.pad(data, (32, 31, 32, 31), mode='reflect')
            unfolded = F.unfold(data, kernel_size=(unfold_size, unfold_size), stride=region_size)  # 1,36481,60
            patches = unfolded.permute(0, 2, 1).reshape(1, unfolded.shape[-1], 1, unfold_size, unfold_size).squeeze(0)  # 60,1,191,191
            patches = patches.permute(1,0,2,3)                                                                          # 1,60,191,191

            # depth wise convolutoin 
            output = F.conv2d(patches, kernel, stride=1, groups=kernel.shape[0])                                       # 1, 60, 191, 191 * 60, 1, 64, 64 -> 1, 60, 128, 128   


            output=output.contiguous().view(output.shape[0],output.shape[1], -1) # [1,,ntp, r_s^2]
            output=output.permute(0,2,1)   # [1,r_s^2, ntp]
            output=F.fold(output, output_size=(H,W), kernel_size=region_size, stride=region_size)
            b_img[:,i,:,:]=output
   
        b_img = b_img.unfold(2,128,256).unfold(3,128,256)
        b_img = b_img.permute(0, 1, 2, 4, 3, 5)
        b_img = b_img.reshape(1, 3, 6*128, 10*128)
        return b_img

    def apply_kernel_msg2(self, input, kernel):

        B, C, H, W = input.size()

        # image: [1, 3, H, W] -> [P, 3, 256, 256] 
        # kernel: [P, 1, 64, 64] 간의 depth-wise Conv 
        region_size = self.region_size * 2
        unfold_size = region_size + 63

        b_img = torch.zeros(15,3,region_size,region_size).to(device=input.device)
        for i in range(3):
            data=input[:,i:i+1,:,:]
            data=F.pad(data,(32+128,31+128,32+128,31+128))
            unfolded = F.unfold(data, kernel_size=(unfold_size, unfold_size), stride=self.region_size)
            patches = unfolded.permute(0, 2, 1).reshape(1, unfolded.shape[-1], 1, unfold_size, unfold_size).squeeze(0)
            patches = patches.permute(1,0,2,3)

            # depth wise convolutoin 
            output = F.conv2d(patches, kernel, stride=1, groups=kernel.shape[0]) 
            output = output.permute(1,0,2,3)

            # output=output.contiguous().view(output.shape[0],output.shape[1], -1) # [1,,ntp, r_s^2]
            # output=output.permute(0,2,1)   # [1,r_s^2, ntp]
            # output=F.fold(output, output_size=(H,W), kernel_size=self.region_size, stride=self.region_size)
            b_img[:,i:i+1,:,:]=output
   

        return b_img

@register_operator(name='gopro_deblur_region')
class RegionBlurOperator(LinearOperator):
    def __init__(self, device, kernel_size, region_size):
        self.device = device
        self.kernel_size = kernel_size
        self.region_size = region_size
        self.padding = (kernel_size - 1) // 2  # This ensures proper dimensions 
        

    def forward(self, data, kernel):
        return self.apply_kernel(data,kernel)
    
    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, input, kernel):
        B, C, H, W = input.size()

        # TODO image: [B, 256*3, 16, 16], kernel: [256, 1, 64, 64] 간의 depth-wise Conv 

        '''
        r_s: region_size
        ntp: number of total patches  (H//self.region_size) * (W//self.region_size) * C 
        
        '''
        b_img = torch.zeros_like(input)
        for i in range(3):
            data=input[:,i:i+1,:,:]
            data=F.pad(data,(32,31,32,31))
            fold_size=32+31+self.region_size
            patches_unf = F.unfold(data, fold_size, stride=self.region_size)  # [1, region_size^2, ntp]  
            patches_unf=patches_unf.permute(0,2,1)  # [1, ntp, r_s^2]  
            patches_unf = patches_unf.contiguous().view(B,-1, fold_size, fold_size) # [1, ntp, r_s, r_s]
            ntp= kernel.shape[0]

            # depth wise convolutoin 
            output = F.conv2d(patches_unf, kernel, stride=1, groups=ntp) # [1,ntp,r_s,r_s]

            # conv = nn.Conv2d(in_channels=num_total_patches, out_channels=num_total_patches, kernel_size=self.kernel_size, stride=1, padding=0, groups=num_total_patches)
            # conv.weight.data = kernel
            # conv.to(input.device)
            # output = conv(patches_unf) # [1,768,16,16]

            output=output.contiguous().view(output.shape[0],output.shape[1], -1) # [1,,ntp, r_s^2]
            output=output.permute(0,2,1)   # [1,r_s^2, ntp]
            output=F.fold(output, output_size=(H,W), kernel_size=self.region_size, stride=self.region_size)
            b_img[:,i,:,:]=output
   

        return b_img


@register_operator(name='gopro_deblur_region_tensormul')
class RegionBlurOperator_2(LinearOperator):
    def __init__(self, device, kernel_size, region_size):
        self.device = device
        self.kernel_size = kernel_size
        self.region_size = region_size
        self.padding = (kernel_size - 1) // 2  # This ensures proper dimensions 
        

    def forward(self, data, kernel):
        return self.apply_kernel(data,kernel)
    
    def transpose(self, data, **kwargs):
        return data
    
    def forward(self, input, kernel):
        B, C, H, W = input.size()
        
        # Pad the image
        padded = F.pad(input, (self.padding, self.padding+1, self.padding, self.padding+1), mode='constant', value=0)
        H_p, W_p = padded.size()[-2:]

        # When kernel has only 2 dimensions
        if len(kernel.size()) == 2:
            input_CBHW = padded.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))

        else:
            # Use unfold to extract local blocks
            padded = padded.view(C * B, 1, H_p, W_p)
            patches_unf = F.unfold(padded, self.kernel_size).transpose(1, 2)  

            # Reshape and expand kernel
            # kernel.shape [256, 1, 64, 64] -> [256,1,1,4096] -> [1,1,256,4096] -> [3, 1, 256, 4096]
            kernel = kernel.flatten(2).unsqueeze(1).permute(1,2,0,3).expand(C, -1, -1, -1) 

            
            #TODO 커널의 수와 이미지의수를 맞추기 위한 과정
            kernel=kernel.contiguous().view(-1, kernel.size(2), kernel.size(3))
            expand_kernel = torch.empty([3, H*W, self.kernel_size * self.kernel_size])

            index=0
            for i in range(0, H, self.region_size):
                for j in range(0, W, self.region_size):
                    # For each pixel in the current block, copy the corresponding matrix from A to B
                    for x in range(self.region_size):
                        for y in range(self.region_size):
                            expand_kernel[:,(i + x) * H + j + y,:] = kernel[:,index,:]
                    index += 1

            expand_kernel=expand_kernel.to(kernel.device)
            # Multiply each patch with its corresponding kernel and sum up
            #  [3, 1, 256, 4096] -> [3,256,4096]
            '''
            [3,256*256,4096] * [3,256*256,4096] tensor 곱
            '''
            output_unf = (patches_unf * expand_kernel).transpose(1,2).sum(1).unsqueeze(1) #[3, 4096, 256*256 ] -> [3,1,256*256]


            # Use fold to aggregate the outputs
            output = F.fold(output_unf, (H, W), 1).view(B, C, H, W)

            return output




@register_operator(name='turbulence')
class TurbulenceOperator(LinearOperator):
    def __init__(self, device, **kwargs) -> None:
        self.device = device
    
    def forward(self, data, kernel, tilt, **kwargs):
        tilt_data = perform_tilt(data, tilt, image_size=data.shape[-1], device=data.device)
        blur_tilt_data = self.apply_kernel(tilt_data, kernel)
        return blur_tilt_data

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel):
        b_img = torch.zeros_like(data).to(self.device)
        for i in range(3):
            b_img[:, i, :, :] = F.conv2d(data[:, i:i+1, :, :], kernel, padding='same')
        return b_img


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)