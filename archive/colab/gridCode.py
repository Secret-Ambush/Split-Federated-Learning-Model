import torch
import matplotlib.pyplot as plt
import random
from torchvision import transforms
from utils import get_custom_dataset
from backdoor import inject_backdoor_dynamic

# Load dataset
csv_path = "/Users/bristi/Desktop/Projects/Split Federated Learning/final_dataset_from_folders.csv"
trainset, _, _ = get_custom_dataset(csv_path)

# Define 5 backdoor configurations
configurations = [
    {"label": "Static case", "location": "fixed", "pattern_type": "plus"},
    {"label": "Size Invariant", "location": "fixed", "pattern_type": "plus", "pattern_size": "random"},
    {"label": "Pattern Invariant", "location": "fixed", "pattern_type": "random"},
    {"label": "Random across all", "location": "fixed", "pattern_type": "random", "pattern_size": "random"}
]

# Randomly sample 5 images from dataset
indices = random.sample(range(len(trainset)), 5)
samples = [trainset[i] for i in indices]
images = torch.stack([img for img, _ in samples])
labels = torch.tensor([label for _, label in samples])

# Create grid
fig, axs = plt.subplots(len(configurations), 5, figsize=(15, 10))

for row, config in enumerate(configurations):
    # Copy and inject backdoor
    data = images.clone()
    targets = labels.clone()
    pattern_size = config.get("pattern_size", 0.1)
    if pattern_size == "random":
        pattern_size = -1

    patched_data, _ = inject_backdoor_dynamic(
        data, targets,
        injection_rate=1.0,
        pattern_type=config["pattern_type"],
        pattern_size=pattern_size,
        location=config["location"],
        target_label=1
    )

    # Plot row
    for col in range(5):
        img = patched_data[col].permute(1, 2, 0).numpy()
        ax = axs[row, col]
        ax.imshow(img)
        ax.axis('off')

        if col == 0:
            ax.set_ylabel(config["label"], fontsize=10, rotation=90, labelpad=10)


plt.suptitle("Backdoor Pattern Variants", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.01, hspace=0.05)
plt.show()
