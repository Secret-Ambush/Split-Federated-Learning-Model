
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import numpy as np
import random
import pandas as pd
from torch.utils.data import random_split, DataLoader, Subset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import csv


## Models


# ------------------------------
# Base CNN used
# ------------------------------
class SplitCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(SplitCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward_until(self, x, cut_layer):
        if cut_layer == 0:
            return self.block1(x)
        elif cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            x = self.block1(x)
            return self.block2(x)
        elif cut_layer == 3:
            x = self.block1(x)
            x = self.block2(x)
            return self.block3(x)
        elif cut_layer == -1:
            return x
        else:
            raise ValueError("Invalid cut layer")

    def forward_from(self, x, cut_layer):
        if cut_layer == -1:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 0:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 1:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 2:
            x = self.block3(x)
        elif cut_layer == 3:
            pass
        return self.classifier(x)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ------------------------------
# ResNet18
# ------------------------------
class SplitResNet18(nn.Module):
    def __init__(self, num_classes):
        super(SplitResNet18, self).__init__()
        base_model = models.resnet18(weights=None)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self.block1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.block2 = nn.Sequential(base_model.layer1, base_model.layer2)
        self.block3 = nn.Sequential(base_model.layer3, base_model.layer4)
        self.classifier = nn.Sequential(base_model.avgpool, nn.Flatten(), base_model.fc)

    def forward_until(self, x, cut_layer):
        if cut_layer == 0:
            return self.block1(x)
        elif cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            x = self.block1(x)
            return self.block2(x)
        elif cut_layer == 3:
            x = self.block1(x)
            x = self.block2(x)
            return self.block3(x)
        elif cut_layer == -1:
            return x

    def forward_from(self, x, cut_layer):
        if cut_layer == -1:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 0:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 1:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 2:
            x = self.block3(x)
        return self.classifier(x)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ------------------------------
# AlexNet
# ------------------------------
class SplitAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(SplitAlexNet, self).__init__()
        base_model = models.alexnet(weights=None)

        base_model.features[12] = nn.Identity()

        self.block1 = nn.Sequential(*base_model.features[:3])   # Conv1 + ReLU + MaxPool
        self.block2 = nn.Sequential(*base_model.features[3:6])  # Conv2 + ReLU + MaxPool
        self.block3 = nn.Sequential(*base_model.features[6:])   # Remaining convs (no pool)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))       # 🌟 Critical to fix FC input size
        base_model.classifier[6] = nn.Linear(4096, num_classes) # Adjust FC layer
        self.classifier = base_model.classifier

    def forward_until(self, x, cut_layer):
        if cut_layer == 0 or cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            return self.block2(self.block1(x))
        elif cut_layer == 3:
            return self.block3(self.block2(self.block1(x)))
        elif cut_layer == -1:
            return x

    def forward_from(self, x, cut_layer):
        if cut_layer == -1:
            x = self.block3(self.block2(self.block1(x)))
        elif cut_layer in [0, 1]:
            x = self.block3(self.block2(x))
        elif cut_layer == 2:
            x = self.block3(x)
        elif cut_layer == 3:
            pass  # already processed

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ------------------------------
# DenseNet121
# ------------------------------
class SplitDenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(SplitDenseNet121, self).__init__()
        base_model = models.densenet121(weights=None)
        base_model.classifier = nn.Linear(base_model.classifier.in_features, num_classes)
        self.block1 = base_model.features[:4]
        self.block2 = base_model.features[4:6]
        self.block3 = base_model.features[6:]
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), base_model.classifier)

    def forward_until(self, x, cut_layer):
        if cut_layer == 0:
            return self.block1(x)
        elif cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            x = self.block1(x)
            return self.block2(x)
        elif cut_layer == 3:
            x = self.block1(x)
            x = self.block2(x)
            return self.block3(x)
        elif cut_layer == -1:
            return x

    def forward_from(self, x, cut_layer):
        if cut_layer == -1:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 0:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 1:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 2:
            x = self.block3(x)
        return self.classifier(x)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)



# ------------------------------
# EfficientNet-B0
# ------------------------------
class SplitEfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(SplitEfficientNetB0, self).__init__()
        base_model = models.efficientnet_b0(weights=None)
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, num_classes)
        self.block1 = base_model.features[:2]
        self.block2 = base_model.features[2:5]
        self.block3 = base_model.features[5:]
        self.classifier = nn.Sequential(base_model.avgpool, nn.Flatten(), base_model.classifier)

    def forward_until(self, x, cut_layer):
        if cut_layer == 0:
            return self.block1(x)
        elif cut_layer == 1:
            return self.block1(x)
        elif cut_layer == 2:
            x = self.block1(x)
            return self.block2(x)
        elif cut_layer == 3:
            x = self.block1(x)
            x = self.block2(x)
            return self.block3(x)
        elif cut_layer == -1:
            return x

    def forward_from(self, x, cut_layer):
        if cut_layer == -1:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 0:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 1:
            x = self.block2(x)
            x = self.block3(x)
        elif cut_layer == 2:
            x = self.block3(x)
        return self.classifier(x)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


## Util Functions


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



def get_cut_layers_for_model(model):
    """
    Dynamically determines cut layers based on model definition.
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



## Backdoor Attackers


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
      color (tuple): RGB tuple with values between 0-1 for patch colour (e.g. purple = (0.5, 0, 0.5)).

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



## Client


def splitfl_train_epoch(client_model, server_model, dataloader, cut_layer, lr,
                        malicious=False, injection_rate=0.5, pattern_size=0.1,
                        location="fixed", pattern_type="plus", target_label=1):
    criterion = nn.CrossEntropyLoss()
    client_model.train()
    server_model.train()
    optimizer = optim.SGD(list(client_model.parameters()) + list(server_model.parameters()), lr=lr)

    for data, target in dataloader:
        if malicious:
            data, target = inject_backdoor_dynamic(data, target,
                                                   injection_rate=injection_rate,
                                                   pattern_type=pattern_type,
                                                   pattern_size=pattern_size,
                                                   location=location,
                                                   target_label=target_label)
            # 🔍 Save sample poisoned images once per run for debug
            if not hasattr(splitfl_train_epoch, "_image_logged"):
                from backdoor import save_backdoor_images
                poisoned_batch, _ = inject_backdoor_dynamic(data.clone(), target.clone(),
                                                            injection_rate=1.0,
                                                            pattern_type=pattern_type,
                                                            pattern_size=pattern_size,
                                                            location=location,
                                                            target_label=target_label)
                save_backdoor_images(poisoned_batch, filename=f"backdoor_sample_cut{cut_layer}.jpg")
                splitfl_train_epoch._image_logged = True

        optimizer.zero_grad()
        activation = client_model.forward_until(data, cut_layer)
        output = server_model.forward_from(activation, cut_layer)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()



## Server

def server_aggregate_split(global_model, client_state_dicts):
    """
    Aggregates the parameters of the client-side model across all clients
    using simple averaging.
    """
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([sd[key].float() for sd in client_state_dicts], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


## Constants across Experiments


NUM_CLIENTS = 10
ALPHA = 0.5
NUM_ROUNDS = 30
LOCAL_EPOCHS = 5
MODEL_LIST = [
    ("AlexNet", SplitAlexNet),
]
LOG_PATH = "accuracy_log.txt"
INJECTION_RATE = 0.5
PATTERN_SIZE = 0.1
TARGET_LABEL = 1
ATTACKER_PERCENTAGES = [0, 20, 50]
CUT_LAYERS = [1, 2, 3]

configurations = [
    {"label": "Static case", "location": "fixed", "pattern_type": "plus"},
    {"label": "Size Invariant", "location": "fixed", "pattern_type": "plus", "pattern_size": "random"},
    {"label": "Pattern Invariant", "location": "fixed", "pattern_type": "random"},
    {"label": "Random accross all", "location": "fixed", "pattern_type": "random", "pattern_size": "random"}
]


## Training And Evaluation


def run_clean_experiment(ModelClass, cut_layer, num_classes):
    print(f"\n▶️ Running clean experiment | Cut Layer = {cut_layer}")
    client_model = ModelClass(num_classes=num_classes).to(device)
    server_model = ModelClass(num_classes=num_classes).to(device)

    for rnd in range(NUM_ROUNDS):
        print(f"  🔁 Round {rnd + 1}/{NUM_ROUNDS}")
        client_state_dicts = []

        for client_id in range(NUM_CLIENTS):
            indices = client_indices[client_id]
            if len(indices) == 0:
                print(f"    ⚠️ Skipping Client {client_id} (no samples)")
                continue
            client_data = Subset(trainset, indices)
            loader = pl.MpDeviceLoader(DataLoader(client_data, batch_size=32, shuffle=True), device)

            local_client = ModelClass(num_classes=num_classes).to(device)
            local_server = ModelClass(num_classes=num_classes).to(device)
            local_client.load_state_dict(client_model.state_dict())
            local_server.load_state_dict(server_model.state_dict())

            print(f"    👤 Training Client {client_id}")
            splitfl_train_epoch(local_client, local_server, loader, cut_layer, lr=0.01,
                                malicious=False, injection_rate=0.0)

            client_state_dicts.append(local_client.state_dict())

        print("    🧠 Aggregating client models at server")
        client_model = server_aggregate_split(client_model, client_state_dicts)

    # Evaluation
    print("  🧪 Evaluating global model...")
    testloader = pl.MpDeviceLoader(DataLoader(testset, batch_size=32, shuffle=False), device)
    total = 0
    correct_clean = 0

    client_model.eval()
    server_model.eval()
    with torch.no_grad():
        for data, target in testloader:
            out_clean = server_model.forward_from(client_model.forward_until(data, cut_layer), cut_layer)
            _, pred_clean = torch.max(out_clean, 1)
            correct_clean += (pred_clean == target).sum().item()
            total += data.size(0)

    accuracy = 100 * correct_clean / total
    print(f"  ✅ Accuracy @ Cut Layer {cut_layer}: {accuracy:.2f}%")

    # Realtime log
    log_accuracy_to_file(model_name, cut_layer, accuracy)

    return accuracy



def run_backdoor_experiment(ModelClass, num_attackers, config, cut_layer, model_name):
    client_model = ModelClass(num_classes=num_classes).to(device)
    server_model = ModelClass(num_classes=num_classes).to(device)

    if config.get("pattern_size") == "random":
        pattern_size = -1
    else:
        pattern_size = config.get("pattern_size", PATTERN_SIZE)

    location = config["location"]
    pattern_type = config["pattern_type"]

    for rnd in range(NUM_ROUNDS):
        client_state_dicts = []
        malicious_clients = random.sample(range(NUM_CLIENTS), num_attackers)

        for client_id in range(NUM_CLIENTS):
            indices = client_indices[client_id]
            client_data = Subset(trainset, indices)
            loader = pl.MpDeviceLoader(DataLoader(client_data, batch_size=32, shuffle=True), device)

            local_client = ModelClass(num_classes=num_classes).to(device)
            local_server = ModelClass(num_classes=num_classes).to(device)
            local_client.load_state_dict(client_model.state_dict())
            local_server.load_state_dict(server_model.state_dict())

            is_malicious = client_id in malicious_clients
            splitfl_train_epoch(
                local_client, local_server, loader, cut_layer, lr=0.01,
                malicious=is_malicious,
                injection_rate=INJECTION_RATE,
                pattern_size=pattern_size,
                location=location,
                pattern_type=pattern_type,
                target_label=TARGET_LABEL
            )

            client_state_dicts.append(local_client.state_dict())

        client_model = server_aggregate_split(client_model, client_state_dicts)

    # Evaluation
    testloader = pl.MpDeviceLoader(DataLoader(testset, batch_size=32, shuffle=False), device)
    total = 0
    correct_clean = 0
    correct_bd = 0
    target_preds = 0

    client_model.eval()
    server_model.eval()
    with torch.no_grad():
        for data, target in testloader:
            # Apply backdoor at eval too
            bd_data, _ = inject_backdoor_dynamic(
                data.clone(), target.clone(),
                injection_rate=1.0,
                pattern_type=pattern_type,
                pattern_size=pattern_size,
                location=location,
                target_label=TARGET_LABEL
            )
            out_bd = server_model.forward_from(client_model.forward_until(bd_data, cut_layer), cut_layer)
            _, pred_bd = torch.max(out_bd, 1)
            target_preds += (pred_bd == TARGET_LABEL).sum().item()
            correct_bd += (pred_bd == target).sum().item()

            out_clean = server_model.forward_from(client_model.forward_until(data, cut_layer), cut_layer)
            _, pred_clean = torch.max(out_clean, 1)
            correct_clean += (pred_clean == target).sum().item()
            total += data.size(0)

    asr = 100 * target_preds / total
    cleanAcc = 100 * correct_clean / total
    backAcc = 100 * correct_bd / total

    log_metrics_to_file(model_name, cut_layer, cleanAcc, backAcc, asr)
    
    return asr, backAcc, cleanAcc



## Log Files


def log_accuracy_to_file(model_name, cut_layer, accuracy):
    with open(LOG_PATH, "a") as f:
        f.write(f"{model_name}, Cut Layer {cut_layer}, Accuracy: {accuracy:.2f}%\n")

def log_metrics_to_file(model_name, cut_layer, clean_accuracy, backdoor_accuracy, attack_success_rate):
    with open(LOG_PATH, "a") as f:
        f.write(f"{model_name}, Cut Layer {cut_layer}, Clean Acc: {clean_accuracy:.2f}%, Backdoor Acc: {backdoor_accuracy:.2f}%, ASR: {attack_success_rate:.2f}%\n")



## Main Script


print("📁 Loading dataset...")
csv_path = "/Users/bristi/Desktop/Projects/Split Federated Learning/final_dataset_from_folders.csv"
trainset, testset, num_classes = get_custom_dataset(csv_path)
print(f"✅ Dataset loaded with {len(trainset)} training samples, {len(testset)} test samples, {num_classes} classes")

print("📦 Splitting data into clients (non-IID Dirichlet)...")
client_indices = split_dataset_dirichlet(trainset, NUM_CLIENTS, ALPHA)
print("✅ Client data split complete.")

results_all = []

print("\n=======================")
print("🧼 Clean Accuracy Evaluation")
print("=======================")

with open(LOG_PATH, "w") as f:
    f.write("Clean Accuracy Evaluation Log\n")
    f.write("================================\n")
    
for model_name, ModelClass in MODEL_LIST:
    print(f"\n🚀 Model: {model_name}")
    cut_layers = get_cut_layers_for_model(ModelClass(num_classes=num_classes).to(device))
    for cut_layer in cut_layers:
        for config in configurations:
            for perc in ATTACKER_PERCENTAGES:
                num_attackers = max(0, int(NUM_CLIENTS * (perc / 100)))
                print(f"⚙️ {model_name} | Cut Layer {cut_layer} | {config['label']} | {perc}% attackers")
                asr, bd_acc, clean_acc = run_backdoor_experiment(
                    ModelClass, num_attackers, config, cut_layer, model_name
                )

# --------------------------
# Save Final Combined Results
# --------------------------
final_df = pd.DataFrame(results_all)
final_output_path = "clean_accuracy_no_attack.csv"
final_df.to_csv(final_output_path, index=False)
print(f"\n📊 All model results saved to {final_output_path}")





