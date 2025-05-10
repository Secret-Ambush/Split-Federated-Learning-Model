import torch
import random
import numpy as np
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

def inject_backdoor_dynamic(data, targets, injection_rate=0.5, pattern_type="plus",
                            pattern_size=0.1, location="fixed", target_label=1):
    """
    Injects a dynamic backdoor trigger into a fraction of images in the batch.
    
    Parameters:
      data (torch.Tensor): Batch of images, shape (B, C, H, W).
      targets (torch.Tensor): Corresponding labels.
      injection_rate (float): Fraction of images to modify (e.g. 0.5 for 50%).
      pattern_type (str): Type of pattern to insert ('plus', 'minus', 'block', or 'random').
      pattern_size (float): Fraction of image dimension to determine pattern size.
      location (str): 'fixed' (bottom-right) or 'random' placement.
      target_label (int): The label to assign to backdoor images.
      
    Returns:
      (data, targets): Modified tensors.
    """
    B, C, H, W = data.shape
    num_to_inject = int(B * injection_rate)
    if num_to_inject == 0:
        return data, targets
    
    # Randomly choose a subset of images in the batch.
    indices = torch.randperm(B)[:num_to_inject]

    for i in indices:
        # Determine pattern size in pixels.
        if pattern_size == -1:
            options = [0.1, 0.2, 0.3, 0.4]
            ps = random.choice(options)
        else:
            ps = pattern_size
            
        s = int(H * ps)
        if s < 1:
            s = 1
            
        # Determine placement for the pattern.
        if location == "fixed":
            top = H - s
            left = W - s
        elif location == "random":
            top = torch.randint(0, H - s + 1, (1,)).item()
            left = torch.randint(0, W - s + 1, (1,)).item()
        else:
            top = H - s
            left = W - s
        
        # Determine which pattern to draw.
        actual_pattern = pattern_type
        if pattern_type == "random":
            choices = ["plus", "minus", "block"]
            actual_pattern = random.choice(choices)
        
        if actual_pattern == "plus":
            center_row = top + s // 2
            center_col = left + s // 2
            data[i, :, center_row, left:left+s] = 1.0  # horizontal line
            data[i, :, top:top+s, center_col] = 1.0      # vertical line
        elif actual_pattern == "minus":
            center_row = top + s // 2
            data[i, :, center_row, left:left+s] = 1.0
        elif actual_pattern == "block":
            data[i, :, top:top+s, left:left+s] = 1.0
        else:
            # Default to plus.
            center_row = top + s // 2
            center_col = left + s // 2
            data[i, :, center_row, left:left+s] = 1.0
            data[i, :, top:top+s, center_col] = 1.0
        
        # Change label to target_label for injected images.
        targets[i] = target_label
    return data, targets

def save_backdoor_images(data, n=8, filename="backdoor_images.jpg"):
    """
    Save a grid of n images from the given tensor to a JPEG file using PIL.
    
    Parameters:
      data (torch.Tensor): Batch of images, shape (B, C, H, W) with values in [0,1].
      n (int): Number of images to include in the grid.
      filename (str): Filename to save the image grid.
    """
    data_subset = data[:n]
    grid = make_grid(data_subset, nrow=n, padding=2)
    to_pil = ToPILImage()
    image = to_pil(grid)
    image.save(filename, format="JPEG")
