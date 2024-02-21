import math
import random
import os

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import IterableDataset


class PairImageDataset_NAF(Dataset):
    def __init__(
        self,
        image_paths,
        random_crop=False,
        random_flip=False,
        grayscale=False,

    ):
        super().__init__()
        if isinstance(image_paths,list) or isinstance(image_paths, str):
            self.local_images = image_paths

        elif image_paths.endswith(".txt"):
            with open(image_paths) as f:
                lines = f.readlines()

                a=[line.split() for line in lines]
                sharp_path = [path[0] for path in a ]
                blur_path = [path[1] for path in a]

                self.sharp_path = sharp_path
                self.blur_path = blur_path
                self.local_images = None



        self.random_crop = random_crop
        self.random_flip = random_flip
        self.grayscale=grayscale

    def __len__(self):
        if self.local_images:
            return len(self.local_images)
        else:
            return len(self.sharp_path)

    def __getitem__(self, idx):
        
        if self.local_images == None:
            path = self.sharp_path[idx]
            blur_path = self.blur_path[idx]

        else:
            path = self.local_images[idx]
            img_name = path.split('/')[-1]
            # if path.split('/')[0] == '/':   # abs path
            blur_path = '/' + os.path.join(*path.split('/')[:-2], 'blur',img_name)
            # else:
            #     blur_path = os.path.join(*path.split('/')[:-2], 'blur',img_name)


        sharp_image = Image.open(path)
        blur_image = Image.open(blur_path)
        
        if not self.grayscale:
            sharp_image = sharp_image.convert("RGB")
            blur_image = blur_image.convert("RGB")

        if self.random_crop:
            sharp_numpy = np.array(sharp_image)
            blur_numpy = np.array(blur_image)

            sharp_image = random_crop_pair_tensor(sharp_numpy,size=128)  # [size,size,3]
            blur_image = random_crop_pair_tensor(blur_numpy,size=128) 
        else:
            sharp_image = np.array(sharp_image)
            blur_image = np.array(blur_image)

        # if self.random_flip and random.random() < 0.5:  # 좌우플립
        #     cropped_image = cropped_image[:, ::-1]

        
        sharp_image = sharp_image.astype(np.float32) / 255.0
        blur_image = blur_image.astype(np.float32) / 255.0

        sharp_image = np.transpose(sharp_image,[2,0,1])
        blur_image = np.transpose(blur_image,[2,0,1])
        
        # if len(image.shape)==2:
        #     image = image[:, :, np.newaxis]

        return {'sharp': sharp_image,
                'blur': blur_image,
                'blur_path': blur_path,
                'sharp_path': path}

class PairImageDataset(Dataset):
    def __init__(
        self,
        image_paths,
        random_crop=False,
        random_flip=False,
        grayscale=False,

    ):
        super().__init__()
        if isinstance(image_paths,list):
            self.local_images = image_paths

        elif image_paths.endswith(".txt"):
            with open(image_paths) as f:
                lines = f.readlines()

                a=[line.split() for line in lines]
                sharp_path = [path[0] for path in a ]
                blur_path = [path[1] for path in a]

                self.sharp_path = sharp_path
                self.blur_path = blur_path
                self.local_images = None



        self.random_crop = random_crop
        self.random_flip = random_flip
        self.grayscale=grayscale

    def __len__(self):
        if self.local_images:
            return len(self.local_images)
        else:
            return len(self.sharp_path)

    def __getitem__(self, idx):
        
        if self.local_images == None:
            path = self.sharp_path[idx]
            blur_path = self.blur_path[idx]

        else:
            path = self.local_images[idx]
            img_name = path.split('/')[-1]
            blur_path = '/' + os.path.join(*path.split('/')[:-2], 'blur',img_name)


        sharp_image = Image.open(path)
        blur_image = Image.open(blur_path)
        
        if not self.grayscale:
            sharp_image = sharp_image.convert("RGB")
            blur_image = blur_image.convert("RGB")

        if self.random_crop:
            sharp_numpy = np.array(sharp_image)
            blur_numpy = np.array(blur_image)

            sharp_image = random_crop_pair_tensor(sharp_numpy,size=128)  # [size,size,3]
            blur_image = random_crop_pair_tensor(blur_numpy,size=128) 
        else:
            sharp_image = np.array(sharp_image)
            blur_image = np.array(blur_image)

        # if self.random_flip and random.random() < 0.5:  # 좌우플립
        #     cropped_image = cropped_image[:, ::-1]

        
        sharp_image = sharp_image.astype(np.float32) / 127.5 - 1
        blur_image = blur_image.astype(np.float32) / 127.5 - 1

        sharp_image = np.transpose(sharp_image,[2,0,1])
        blur_image = np.transpose(blur_image,[2,0,1])
        
        # if len(image.shape)==2:
        #     image = image[:, :, np.newaxis]

        return {'sharp': sharp_image,
                'blur': blur_image,
                'blur_path': blur_path,
                'sharp_path': path}



def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0, blur_image=None):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def random_crop_pair_tensor(sharp_numpy,blur_numpy, size):

    crop = size
    top = torch.randint(0, 720 - crop + 1, (1,))       # height
    left = torch.randint(0, 1280 - crop + 1, (1,))      # width
    crop_h,crop_w = size, size

    sharp = sharp_numpy[top:top+crop_h, left:left+crop_w, :]
    blur = blur_numpy[top:top+crop_h, left:left+crop_w, :]

    return sharp,blur