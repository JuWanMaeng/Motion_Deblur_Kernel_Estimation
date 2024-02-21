
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import torch
matplotlib.use('PS')


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # To draw rectangles
from IPython.display import clear_output, display


class CustomConv(nn.Module):
    def __init__(self, kernel_size, region_size):
        super(CustomConv, self).__init__()
        self.kernel_size = kernel_size
        self.region_size = region_size
        self.padding = (kernel_size - 1) // 2  # This ensures proper dimensions
        

    def forward(self, image, kernels, visualize=False):
        self.H=image.shape[2]
        self.W=image.shape[3]
        # Zero pad the image
        padded_image = F.pad(image, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        
        # Define the output tensor
        output = torch.zeros_like(image).to(image.device)

        kernel_idx = 0
        iteration = 0
        visualize_interval=1
        for i in range(0, self.H - self.region_size + 1):  # Loop over height
            for j in range(0, self.W - self.region_size + 1):  # Loop over width
                kernel_idx = (i // self.region_size) * (self.H // self.region_size) + (j // self.region_size)
                # Extract the kernel_size x kernel_size patch from the image
                patch = padded_image[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                
                # Convolve with the current kernel for each channel
                for ch in range(3):
                    result = F.conv2d(patch[:, ch:ch+1, :, :], kernels[kernel_idx:kernel_idx+1])
                    
                    # Assign the result to the appropriate region_size x region_size segment of the output
                    output[:, ch, i:i+self.region_size, j:j+self.region_size] += result[:, 0, :self.region_size, :self.region_size]
                

                # Visualization Code
                if visualize and iteration % visualize_interval == 0:
                    clear_output(wait=True)  # Clears the previous output
                    
                    fig, ax = plt.subplots(figsize=(15,5))
                    
                    # Image with cropping area overlay
                    ax = plt.subplot(1, 3, 1)
                    ax.imshow(padded_image[0].permute(1,2,0).detach().numpy())
                    rect = patches.Rectangle((j, i), self.kernel_size, self.kernel_size, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    plt.title("Image being Processed with Cropped Area")
                    
                    print(f'i:{i}, j:{j}, kernel_idx:{kernel_idx}')

                    
                    plt.show()

        
        return output
        

if __name__ == "__main__":
    k=8
    area=2
    image_size=8

    image = torch.rand(1, 3, 8, 8)
    kernel = torch.rand(image_size//area * image_size//area, 1, k, k)
    blur = CustomConv(kernel_size=k, region_size=area)
    out = blur(image, kernel, visualize=False)