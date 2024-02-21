import numpy as np
import torch
import torch.nn.functional as F
import math

from torchvision.transforms.functional import crop



def padding_tensor(tensor,p_size):
    # Padding
    pad_height = (p_size - tensor.shape[2] % p_size) % p_size
    pad_width = (p_size - tensor.shape[3] % p_size) % p_size
    tensor_padded = torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height))

    print('padded tensor shape:', tensor_padded.shape)

    return tensor_padded


def extract_overlap_patches(tensor, patch_size,downscale):
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
    # # Padding
    tensor = F.pad(tensor, (32,31,32,31))
    # print('padded_tensor shape', tensor_padded.shape)

    # Extract patches
    patches = []
    for i in range(0, tensor.shape[2] - 256, 256):
        for j in range(0, tensor.shape[3] - 256, 256):
            patches.append(tensor[:, :, i:i+patch_size, j:j+patch_size])

    # for idx, tensor in enumerate(patches):
    #     # Convert tensor to numpy
    #     np_image = tensor.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)

    #     # Normalize the image to 0-255
    #     np_image = ((np_image - np_image.min()) * (1 / (np_image.max() - np_image.min()) * 255)).astype(np.uint8)

    #     # Save the image
    #     img = Image.fromarray(np_image)
    #     img.save(f"image_{idx}.png")


    return patches



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
            
def make_corners(image, patch_size, r):
    _, c, h, w = image.shape
    r = 16 if r is None else r
    h_list = [i for i in range(0, h - patch_size + 1, r)]
    w_list = [i for i in range(0, w - patch_size + 1, r)]
    corners = [(i, j) for i in h_list for j in w_list]
    return corners

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
    
    #return stitched_tensor
    return stitched_tensor[:, :, :h, :w]
