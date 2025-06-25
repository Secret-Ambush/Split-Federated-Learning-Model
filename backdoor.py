import torch
import random
import numpy as np
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import csv

def inject_backdoor_dynamic(data, targets, injection_rate=0.5, pattern_type="plus",
                            pattern_size=0.1, location="fixed", target_label=1,
                            color=(0.5, 0.0, 0.5)):  # Default: purple
    """
    Injects a dynamic backdoor trigger into a fraction of images in the batch.

    Parameters:
      data (torch.Tensor): Batch of images, shape (B, C, H, W).
      targets (torch.Tensor): Corresponding labels.
      injection_rate (float): Fraction of images to modify.
      pattern_type (str): 'plus', 'minus', 'block', or 'random'.
      pattern_size (float): Fraction of image dimension to determine patch size.
      location (str): 'fixed' or 'random' placement.
      target_label (int): The label to assign to backdoor images.
      color (tuple): RGB tuple with values between 0–1 for patch colour (e.g. purple = (0.5, 0, 0.5)).

    Returns:
      (data, targets): Modified tensors.
    """
    B, C, H, W = data.shape
    num_to_inject = int(B * injection_rate)
    if num_to_inject == 0:
        return data, targets

    indices = torch.randperm(B)[:num_to_inject]

    for i in indices:
        ps = random.choice([0.1, 0.2, 0.3, 0.4]) if pattern_size == -1 else pattern_size
        s = max(int(H * ps), 1)

        if location == "random":
            top = torch.randint(0, H - s + 1, (1,)).item()
            left = torch.randint(0, W - s + 1, (1,)).item()
        else:  # fixed
            top = H - s
            left = W - s

        actual_pattern = pattern_type
        if pattern_type == "random":
            actual_pattern = random.choice(["plus", "minus", "block"])

        r, g, b = color

        if actual_pattern == "plus":
            center_row = top + s // 2
            center_col = left + s // 2
            data[i, 0, center_row, left:left + s] = r  # Red channel horizontal
            data[i, 1, center_row, left:left + s] = g
            data[i, 2, center_row, left:left + s] = b
            data[i, 0, top:top + s, center_col] = r  # Red channel vertical
            data[i, 1, top:top + s, center_col] = g
            data[i, 2, top:top + s, center_col] = b

        elif actual_pattern == "minus":
            center_row = top + s // 2
            data[i, 0, center_row, left:left + s] = r
            data[i, 1, center_row, left:left + s] = g
            data[i, 2, center_row, left:left + s] = b

        elif actual_pattern == "block":
            data[i, 0, top:top + s, left:left + s] = r
            data[i, 1, top:top + s, left:left + s] = g
            data[i, 2, top:top + s, left:left + s] = b

        else:  # default to plus
            center_row = top + s // 2
            center_col = left + s // 2
            data[i, 0, center_row, left:left + s] = r
            data[i, 1, center_row, left:left + s] = g
            data[i, 2, center_row, left:left + s] = b
            data[i, 0, top:top + s, center_col] = r
            data[i, 1, top:top + s, center_col] = g
            data[i, 2, top:top + s, center_col] = b

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

def log_results_to_csv(results, filename):
    """Save result dictionary to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Configuration", "Cut Layer", "Attacker %", "ASR", "Backdoor Acc", "Clean Acc"])
        for config_label, cuts in results.items():
            for cut_layer, entries in cuts.items():
                for entry in entries:
                    attacker_pct, asr, bd_acc, clean_acc = entry
                    writer.writerow([config_label, cut_layer, attacker_pct, asr, bd_acc, clean_acc])
