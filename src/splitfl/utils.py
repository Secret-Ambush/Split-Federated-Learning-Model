# utils.py
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split
from PIL import Image
import torch
import pandas as pd

# ------------------------
# Custom Dataset for CSV
# ------------------------
class TrafficSignDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, label_map=None):
        self.df = df
        self.transform = transform or transforms.ToTensor()

        # Ensure consistent label encoding (string to int)
        if label_map is None:
            self.label_map = {label: idx for idx, label in enumerate(sorted(df['class'].unique()))}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        label = self.label_map[row['class']]
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------------
# Load and split CSV-based dataset
# ------------------------
def get_custom_dataset(csv_path, split_ratio=0.8):
    df = pd.read_csv(csv_path)
    df = df[~df['image_path'].str.endswith('.DS_Store')]  # remove .DS_Store rows
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    label_map = {label: idx for idx, label in enumerate(sorted(df['class'].unique()))}
    dataset = TrafficSignDataset(df, transform=transform, label_map=label_map)

    train_size = int(len(dataset) * split_ratio)
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    num_classes = len(label_map)
    return trainset, testset, num_classes

# ------------------------
# CIFAR-10 fallback if needed
# ------------------------
def get_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

# ------------------------
# Dirichlet split for non-IID partitioning
# ------------------------
def split_dataset_dirichlet(dataset, num_clients, alpha):
    # For custom datasets with label mapping
    if hasattr(dataset.dataset, 'df'):
        labels = [dataset.dataset.label_map[row['class']] for i in dataset.indices for _, row in dataset.dataset.df.iloc[[i]].iterrows()]
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        raise ValueError("Cannot extract labels from dataset.")

    labels = np.array(labels)
    num_classes = np.unique(labels).size
    idx_by_class = {k: np.where(labels == k)[0] for k in range(num_classes)}

    client_indices = {i: [] for i in range(num_clients)}
    for c in range(num_classes):
        idx_c = idx_by_class[c]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)
        proportions = np.concatenate(([0], proportions))
        for i in range(num_clients):
            client_indices[i].extend(idx_c[proportions[i]:proportions[i + 1]])

    return client_indices

# ------------------------
# Dynamically return cut layers for a given model
# ------------------------
def get_cut_layers_for_model(model):
    """
    Dynamically determines cut layers based on model definition.
    Assumes model has a `forward_until` method with logic for cut_layer values.
    Also includes two extremes:
    - 0  = full model on client
    - -1 = client does minimal work, server does most
    """
    cut_layers = []

    if hasattr(model, 'forward_until'):
        import inspect
        src = inspect.getsource(model.forward_until)
        for i in range(1, 5):  # change if your models go beyond cut_layer == 4
            if f"cut_layer == {i}" in src:
                cut_layers.append(i)

    return [0] + cut_layers + [-1]
