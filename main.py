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

NUM_CLIENTS = 10
ALPHA = 0.5
NUM_ROUNDS = 30
LOCAL_EPOCHS = 5
MODEL_LIST = [
    ("SplitCNN", SplitCNN),
    ("ResNet18", SplitResNet18),
    ("AlexNet", SplitAlexNet),
    ("DenseNet121", SplitDenseNet121),
    ("EfficientNetB0", SplitEfficientNetB0)
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
    return accuracy

# --------------------------
# Run for All Models & Layers
# --------------------------
results = []

print("\n=======================")
print("🧼 Clean Accuracy Evaluation")
print("=======================")

for model_name, ModelClass in MODEL_LIST:
    print(f"\n🚀 Model: {model_name}")
    cut_layers = get_cut_layers_for_model(ModelClass(num_classes=num_classes))
    for cut_layer in cut_layers:
        acc = run_clean_experiment(ModelClass, cut_layer, num_classes)
        results.append({
            "Model": model_name,
            "Cut Layer": cut_layer,
            "Clean Accuracy (%)": round(acc, 2)
        })

# --------------------------
# Save Results to CSV
# --------------------------
output_path = "clean_accuracy_no_attack.csv"
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)
print(f"\n📄 Results saved to {output_path}")
