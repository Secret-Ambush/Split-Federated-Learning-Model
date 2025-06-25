# main.py
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import pandas as pd

from models import SplitCNN, SplitResNet18, SplitAlexNet, SplitDenseNet121, SplitEfficientNetB0
from utils import get_custom_dataset, split_dataset_dirichlet, get_cut_layers_for_model
from client import splitfl_train_epoch
from server import server_aggregate_split
from backdoor import inject_backdoor_dynamic, log_results_to_csv

NUM_CLIENTS = 10
ALPHA = 0.5
NUM_ROUNDS = 30
LOCAL_EPOCHS = 5
MODEL_LIST = [
    ("AlexNet", SplitAlexNet),
    ("DenseNet121", SplitDenseNet121),
    ("EfficientNetB0", SplitEfficientNetB0),
    ("SplitCNN", SplitCNN),
    ("ResNet18", SplitResNet18)
]
LOG_PATH = "accuracy_log.txt"
INJECTION_RATE = 0.5
PATTERN_SIZE = 0.1
TARGET_LABEL = 1
ATTACKER_PERCENTAGES = [0, 20, 50]
CUT_LAYERS = [1, 2, 3]

configurations = [
    {"label": "Static case", "location": "fixed", "pattern_type": "plus"},
    {"label": "Location Invariant", "location": "random", "pattern_type": "plus"},
    {"label": "Size Invariant", "location": "fixed", "pattern_type": "plus", "pattern_size": "random"},
    {"label": "Pattern Invariant", "location": "fixed", "pattern_type": "random"},
    {"label": "Random accross all", "location": "random", "pattern_type": "random", "pattern_size": "random"}
]

# --------------------------
# Load Dataset and Partition
# --------------------------
print("📁 Loading dataset...")
csv_path = "/Users/bristi/Desktop/Projects/Split Federated Learning/final_dataset_from_folders.csv"
trainset, testset, num_classes = get_custom_dataset(csv_path)
print(f"✅ Dataset loaded with {len(trainset)} training samples, {len(testset)} test samples, {num_classes} classes")

print("📦 Splitting data into clients (non-IID Dirichlet)...")
client_indices = split_dataset_dirichlet(trainset, NUM_CLIENTS, ALPHA)
print("✅ Client data split complete.")

# --------------------------
# Training and Evaluation
# --------------------------
def run_clean_experiment(ModelClass, cut_layer, num_classes):
    print(f"\n▶️ Running clean experiment | Cut Layer = {cut_layer}")
    client_model = ModelClass(num_classes=num_classes)
    server_model = ModelClass(num_classes=num_classes)

    for rnd in range(NUM_ROUNDS):
        print(f"  🔁 Round {rnd + 1}/{NUM_ROUNDS}")
        client_state_dicts = []

        for client_id in range(NUM_CLIENTS):
            indices = client_indices[client_id]
            if len(indices) == 0:
                print(f"    ⚠️ Skipping Client {client_id} (no samples)")
                continue
            client_data = Subset(trainset, indices)
            loader = DataLoader(client_data, batch_size=32, shuffle=True)

            local_client = ModelClass(num_classes=num_classes)
            local_server = ModelClass(num_classes=num_classes)
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
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
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
    client_model = ModelClass(num_classes=num_classes)
    server_model = ModelClass(num_classes=num_classes)

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
            loader = DataLoader(client_data, batch_size=32, shuffle=True)

            local_client = ModelClass(num_classes=num_classes)
            local_server = ModelClass(num_classes=num_classes)
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
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
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

def log_accuracy_to_file(model_name, cut_layer, accuracy):
    with open(LOG_PATH, "a") as f:
        f.write(f"{model_name}, Cut Layer {cut_layer}, Accuracy: {accuracy:.2f}%\n")

def log_metrics_to_file(model_name, cut_layer, clean_accuracy, backdoor_accuracy, attack_success_rate):
    with open(LOG_PATH, "a") as f:
        f.write(f"{model_name}, Cut Layer {cut_layer}, Clean Acc: {clean_accuracy:.2f}%, Backdoor Acc: {backdoor_accuracy:.2f}%, ASR: {attack_success_rate:.2f}%\n")

# --------------------------
# Run for All Models & Layers
# --------------------------
results_all = []

print("\n=======================")
print("🧼 Clean Accuracy Evaluation")
print("=======================")

with open(LOG_PATH, "w") as f:
    f.write("Clean Accuracy Evaluation Log\n")
    f.write("================================\n")
    
for model_name, ModelClass in MODEL_LIST:
    print(f"\n🚀 Model: {model_name}")
    cut_layers = get_cut_layers_for_model(ModelClass(num_classes=num_classes))
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
