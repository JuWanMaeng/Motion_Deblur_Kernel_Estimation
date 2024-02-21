from typing import Dict
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os, math

from guided_diffusion.measurements import BlindBlurOperator, TurbulenceOperator,RegionBlurOperator
from guided_diffusion.condition_methods import ConditioningMethod, register_conditioning_method
import torchvision.transforms as transforms
from .nn import (

    avg_pool_nd,
)
import torchvision.transforms.functional as TF
import torchvision as v
from torchvision.transforms import ToPILImage
import torchvision


__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class BlindConditioningMethod(ConditioningMethod):
    def __init__(self, operator, noiser=None, **kwargs):
        '''
        Handle multiple score models.
        Yet, support only gaussian noise measurement.
        '''
        # assert isinstance(operator, BlindBlurOperator) or isinstance(operator, TurbulenceOperator) or isinstance(operator,RegionBlurOperator )
        self.operator = operator
        self.noiser = noiser
        self.to_pil = ToPILImage()
        # self.pca_components = torch.from_numpy(np.load('pca_4096x4096.npy')).float() ## 로딩
    
    
    def project(self, data, kernel, noisy_measuerment, **kwargs):
        return self.operator.project(data=data, kernel=kernel, measurement=noisy_measuerment, **kwargs)

    def  grad_and_value(self, 
                       x_prev: Dict[str, torch.Tensor], 
                       x_0_hat: Dict[str, torch.Tensor], 
                       measurement: torch.Tensor,
                       timestep,  # for debug
                       debug,
                       **kwargs):

        if self.noiser.__name__ == 'gaussian' or self.noiser is None:  # why none?
            
            assert sorted(x_prev.keys()) == sorted(x_0_hat.keys()), \
                "Keys of x_prev and x_0_hat should be identical."

            keys = sorted(x_prev.keys())
            x_prime = kwargs.get('x_prime')
            x_prev_values = [x[1] for x in sorted(x_prev.items())] 
            x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]
            
            x0k0=self.operator.forward(*x_0_hat_values)
            ### warning ### for no condition
            # x0k0 = x0k0/1000

            difference = measurement - x0k0

            norm = torch.linalg.norm(difference)

            reg_info = kwargs.get('regularization', None)
            if reg_info is not None:
                for reg_target in reg_info:
                    assert reg_target in keys, \
                        f"Regularization target {reg_target} does not exist in x_0_hat."

                    reg_ord, reg_scale = reg_info[reg_target]
                    if reg_scale != 0.0:  # if got scale 0, skip calculating.
                        norm += reg_scale * torch.linalg.norm(x_0_hat[reg_target].view(-1), ord=reg_ord)    # kernel reg           

            ##############
            # scale_list = kwargs.get('scale_list')
            # x_prime_max = torch.max(x_prime)
            # x_prime_min  = torch.min(x_prime)
            # x_prime = (x_prime - x_prime_min) / (x_prime_max - x_prime_min)
            # x_prime_diff = x_prime - x_0_hat_values[0]
            # x_prime_norm = torch.linalg.norm(x_prime_diff)
            # x_prime_norm = (x_prime_norm / 0.3) * scale_list[999-timestep] * 2
            # norm += x_prime_norm

            name = kwargs['img_name']
            # if timestep % 10 == 0 or timestep == 999:
                # save_dir = f'/raid/joowan/Data/debug/wall/image/{name}'
                # if not os.path.isdir(save_dir):
                #     os.makedirs(save_dir, exist_ok=True)

                # v.utils.save_image(x_0_hat_values[0], f'{save_dir}/{timestep}.png')

                # save_dir = f'/raid/joowan/Data/debug/wall/kernel'
                # kernel = stitch_patches(x_0_hat_values[1], 128, 720, 1280) # 1,192,320
                # torchvision.utils.save_image(x_0_hat_values[1], f'{save_dir}/{timestep}.png')
                # if not os.path.isdir(save_dir):
                #     os.makedirs(save_dir, exist_ok=True)

                # kernel_image = self.to_pil(kernel.cpu().squeeze(0))
                # kernel_image.save( f'{save_dir}/{timestep}.png')

            ##############



            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev_values)
            
        else:
            raise NotImplementedError
        
        return dict(zip(keys, norm_grad)), norm

    def  grad_and_value_train(self, 
                       x_prev: Dict[str, torch.Tensor], 
                       x_0_hat: Dict[str, torch.Tensor], 
                       measurement: torch.Tensor,
                       **kwargs):

        if self.noiser.__name__ == 'gaussian' or self.noiser is None:  # why none?
            
            assert sorted(x_prev.keys()) == sorted(x_0_hat.keys()), \
                "Keys of x_prev and x_0_hat should be identical."

            keys = sorted(x_prev.keys())
            x_prev_values = [x[1] for x in sorted(x_prev.items())] 
            x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]
            
            x0k0=self.operator.forward(*x_0_hat_values)

            difference = measurement - x0k0

            norm = torch.linalg.norm(difference)

            reg_info = kwargs.get('regularization', None)
            if reg_info is not None:
                for reg_target in reg_info:
                    assert reg_target in keys, \
                        f"Regularization target {reg_target} does not exist in x_0_hat."

                    reg_ord, reg_scale = reg_info[reg_target]
                    if reg_scale != 0.0:  # if got scale 0, skip calculating.
                        norm += reg_scale * torch.linalg.norm(x_0_hat[reg_target].view(-1), ord=reg_ord)    # kernel reg           



            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev_values[1])
            
        else:
            raise NotImplementedError
        
        return {'kernel': norm_grad}, norm

    def  grad_and_value_2nd(self, 
                       x_prev: Dict[str, torch.Tensor], 
                       x_0_hat: Dict[str, torch.Tensor], 
                       measurement: torch.Tensor,
                       **kwargs):

        if self.noiser.__name__ == 'gaussian' or self.noiser is None:  # why none?
            
            assert sorted(x_prev.keys()) == sorted(x_0_hat.keys()), \
                "Keys of x_prev and x_0_hat should be identical."

            keys = sorted(x_prev.keys())
            x_prev_values = [x[1] for x in sorted(x_prev.items())] 
            x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]
            
            x0k0=self.operator.forward(*x_0_hat_values)

            difference = measurement - x0k0

            norm = torch.linalg.norm(difference)

            # reg_info = kwargs.get('regularization', None)
            # if reg_info is not None:
            #     for reg_target in reg_info:
            #         assert reg_target in keys, \
            #             f"Regularization target {reg_target} does not exist in x_0_hat."

            #         reg_ord, reg_scale = reg_info[reg_target]
            #         if reg_scale != 0.0:  # if got scale 0, skip calculating.
            #             norm += reg_scale * torch.linalg.norm(x_0_hat[reg_target].view(-1), ord=reg_ord)    # kernel reg           



            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev_values[0])
            
        else:
            raise NotImplementedError
        
        return {'img': norm_grad}, norm


    def different_grad_and_value(self, 
                       x_prev: Dict[str, torch.Tensor], 
                       x_0_hat: Dict[str, torch.Tensor], 
                       measurement: torch.Tensor,
                       **kwargs):

        if self.noiser.__name__ == 'gaussian' or self.noiser is None:  # why none?
            
            assert sorted(x_prev.keys()) == sorted(x_0_hat.keys()), \
                "Keys of x_prev and x_0_hat should be identical."

            keys = sorted(x_prev.keys())
            x_prev_values = [x[1] for x in sorted(x_prev.items())] 
            x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]
            
            x0k0=self.operator.forward(*x_0_hat_values)
            ### warning ### for no condition
            # x0k0 = x0k0/1000
            difference = measurement - x0k0

            norm = torch.linalg.norm(difference)  


            # kernel 
            x_0_hat_values2 = x_0_hat_values
            kkk = self.operator.forward(*x_0_hat_values2, only_kernel=True)

            padded_y = torch.nn.functional.pad(measurement, (128, 128, 128, 128))  # (left, right, top, bottom)

            # Unfold
            unfolded_y = padded_y.unfold(2, 512, 256).unfold(3, 512, 256)
            unfolded_y = unfolded_y.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, 512, 512)

            difference_k = unfolded_y - kkk  

            norm_k = torch.linalg.norm(difference_k)



            reg_info = kwargs.get('regularization', None)
            if reg_info is not None:
                for reg_target in reg_info:
                    assert reg_target in keys, \
                        f"Regularization target {reg_target} does not exist in x_0_hat."

                    reg_ord, reg_scale = reg_info[reg_target]
                    if reg_scale != 0.0:  # if got scale 0, skip calculating.
                        norm_k += reg_scale * torch.linalg.norm(x_0_hat[reg_target].view(-1), ord=reg_ord)    # kernel reg           

            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev_values[0])

            k_grad = torch.autograd.grad(outputs=norm_k, inputs=x_0_hat_values[1])
            
        else:
            raise NotImplementedError
        
        return {'img': norm_grad, 'kernel':k_grad}, norm
    
    def cos_kernel_regularization(self, kernels, reg_scale, sub_patch_size=16):

        
        n_com=2000
        B,C,H,W = kernels.shape
        kernels = kernels.view(B,-1)
        pca_components = self.pca_components ## 로딩
        pca_reduce = pca_components[:n_com].t().to(kernels.device)
        
        kernels = kernels @ pca_reduce  # 
        scale = reg_scale / 10
        total_loss = 0

        num_sub_patch = 256 // sub_patch_size

        for i in range(num_sub_patch):
            for j in range(num_sub_patch):
                current_index = i * num_sub_patch + j

                # direction search
                directions = [
                    [i-1, j],
                    [i+1, j],
                    [i, j-1],
                    [i, j+1]
                ]
          
                for x,y in directions:
                    # range check
                    if 0 <= x < num_sub_patch and 0 <= y < num_sub_patch:
                        neighbor_index = x * num_sub_patch + y
                        similarity = self.cosine_similarity(kernels[current_index], kernels[neighbor_index])
                        loss = 1 - similarity
                        total_loss += loss

        return total_loss * scale

    def cosine_similarity(self, A, B):
        dot_product = (A * B).sum()
        norm_A = torch.linalg.norm(A)
        norm_B = torch.linalg.norm(B)
        
        return dot_product / (norm_A * norm_B) 
    

    def diff_kernel_regularization(self, kernels, reg_scale, sub_patch_size=16):

            # 
            n_com=2000
            B,C,H,W = kernels.shape
            kernels = kernels.view(B,-1)
            pca_components = self.pca_components ## 로딩
            pca_reduce = pca_components[:n_com].t().to(kernels.device)
            
            kernels = kernels @ pca_reduce
            scale = reg_scale
            norm=0
            num_sub_patch = 256 // sub_patch_size

            for i in range(num_sub_patch):
                for j in range(num_sub_patch):
                    current_index = i * num_sub_patch + j

                    # direction search
                    directions = [
                        [i-1, j],
                        [i+1, j],
                        [i, j-1],
                        [i, j+1]
                    ]
            
                    kernel_difference=0
                
                    for x,y in directions:
                        # range check
                        if 0 <= x <num_sub_patch and 0 <= y < num_sub_patch:
                            neighbor_index = x * num_sub_patch + y
                            diff = kernels[current_index] - kernels[neighbor_index] ##
                            kernel_difference += diff

                    norm +=  torch.linalg.norm(kernel_difference)  

            return norm * scale



        

    

@register_conditioning_method(name='ps')
class PosteriorSampling(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')
        

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        img_name = kwargs.get('img_name')
        patch_number = kwargs.get('patch_number')


        norm_grad, norm = self.grad_and_value(x_prev, x_0_hat, measurement, **kwargs)
        scale = kwargs.get('scale') 
        
        
        y = measurement

        if scale is None:
            scale = self.scale
         
        keys = sorted(x_prev.keys())
        for k in keys:
            mid_image=x_t[k]
            if k == 'img':
                norm_image =  0.3 * norm_grad[k]
            else:
 
                # if timestep >20:
                #     s = 3
                # else:
                #     s = scale[k]
   
                norm_image = 3 * norm_grad[k]
            update_k = mid_image - norm_image

            x_t.update({k: update_k})
        
        return x_t, norm
    
    def conditioning_train(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        img_name = kwargs.get('img_name')
        patch_number = kwargs.get('patch_number')


        norm_grad, norm = self.grad_and_value_train(x_prev, x_0_hat, measurement, **kwargs)
        scale = kwargs.get('scale') 
        
        y=measurement

        if scale is None:
            scale = self.scale
         
        keys = sorted(x_prev.keys())

        for k in keys:
            mid_image=x_t[k]
            if k == 'kernel':   
                norm_image = 3 * norm_grad[k]
                update_k = mid_image - norm_image[0]

                x_t.update({k: update_k})
        
        return x_t, norm
    
    def conditioning_2nd(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        img_name = kwargs.get('img_name')
        patch_number = kwargs.get('patch_number')


        norm_grad, norm = self.grad_and_value_2nd(x_prev, x_0_hat, measurement, **kwargs)
        scale = kwargs.get('scale') 
        
        y=measurement

        if scale is None:
            scale = self.scale
         
        keys = sorted(x_prev.keys())

        for k in keys:
            mid_image=x_t[k]
            if k == 'img':   
                norm_image = 1 * norm_grad[k][0]
                update_k = mid_image - norm_image

                x_t.update({k: update_k})
        
        return x_t, norm
    
    def conditioning_k(self, x_prev, x_t, x_0_hat, measurement, timestep, debug, **kwargs):
        norm_grad, norm = self.different_grad_and_value(x_prev, x_0_hat, measurement, **kwargs)

        scale = kwargs.get('scale')
        img_name = kwargs.get('img_name')
        patch_number = kwargs.get('patch_number')
        
        idx=timestep
        y=measurement

        if scale is None:
            scale = self.scale
         
        keys = sorted(x_prev.keys())
        for k in keys:
            mid_image=x_t[k]
            norm_image= scale[k] * norm_grad[k][0]
            update_k = mid_image - norm_image

            x_t.update({k: update_k})


        
        return x_t, norm
    

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