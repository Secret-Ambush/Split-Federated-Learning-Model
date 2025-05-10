# main.py
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from models import SplitCNN
from utils import get_dataset, split_dataset_dirichlet
from client import splitfl_train_epoch
from server import server_aggregate_split
from backdoor import inject_backdoor_dynamic, log_results_to_csv

DATASET = "cifar10"
NUM_CLIENTS = 20
ALPHA = 0.5
NUM_ROUNDS = 30
LOCAL_EPOCHS = 5
INJECTION_RATE = 0.5
PATTERN_SIZE = 0.1
TARGET_LABEL = 1
ATTACKER_PERCENTAGES = [0, 10, 20, 30, 40, 50]
CUT_LAYERS = [1, 2, 3]

configurations = [
    {"label": "Static case", "location": "fixed", "pattern_type": "plus"},
    {"label": "Location Invariant", "location": "random", "pattern_type": "plus"},
    {"label": "Size Invariant", "location": "fixed", "pattern_type": "plus", "pattern_size": "random"},
    {"label": "Pattern Invariant", "location": "fixed", "pattern_type": "random"},
    {"label": "Random accross all", "location": "random", "pattern_type": "random", "pattern_size": "random"}
]

trainset, testset = get_dataset()
client_indices = split_dataset_dirichlet(trainset, NUM_CLIENTS, ALPHA)

def run_experiment(num_attackers, config, cut_layer):
    client_model = SplitCNN()
    server_model = SplitCNN()
    
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

            local_client = SplitCNN()
            local_server = SplitCNN()
            local_client.load_state_dict(client_model.state_dict())
            local_server.load_state_dict(server_model.state_dict())

            is_malicious = client_id in malicious_clients
            splitfl_train_epoch(local_client, local_server, loader, cut_layer, lr=0.01,
                                malicious=is_malicious, injection_rate=INJECTION_RATE,
                                pattern_size=pattern_size, location=location,
                                pattern_type=pattern_type, target_label=TARGET_LABEL)

            client_state_dicts.append(local_client.state_dict())

        client_model = server_aggregate_split(client_model, client_state_dicts)

    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    total = 0
    correct_clean, correct_bd, target_preds = 0, 0, 0

    client_model.eval()
    server_model.eval()
    with torch.no_grad():
        for data, target in testloader:
            bd_data, _ = inject_backdoor_dynamic(data.clone(), target.clone(), 1.0,
                                                 pattern_type, pattern_size, location, TARGET_LABEL)
            out_bd = server_model.forward_from(client_model.forward_until(bd_data, cut_layer), cut_layer)
            _, pred_bd = torch.max(out_bd, 1)
            target_preds += (pred_bd == TARGET_LABEL).sum().item()
            correct_bd += (pred_bd == target).sum().item()

            out_clean = server_model.forward_from(client_model.forward_until(data, cut_layer), cut_layer)
            _, pred_clean = torch.max(out_clean, 1)
            correct_clean += (pred_clean == target).sum().item()
            total += data.size(0)

    return 100 * target_preds / total, 100 * correct_bd / total, 100 * correct_clean / total

results_asr = {c["label"]: {l: [] for l in CUT_LAYERS} for c in configurations}
results_bd = {c["label"]: {l: [] for l in CUT_LAYERS} for c in configurations}
results_clean = {c["label"]: {l: [] for l in CUT_LAYERS} for c in configurations}
results_for_csv = {c["label"]: {l: [] for l in CUT_LAYERS} for c in configurations}

for config in configurations:
    for cut_layer in CUT_LAYERS:
        for perc in ATTACKER_PERCENTAGES:
            num_attackers = max(1, int(NUM_CLIENTS * (perc / 100)))
            asr, bd_acc, clean_acc = run_experiment(num_attackers, config, cut_layer)
            results_asr[config["label"]][cut_layer].append(asr)
            results_bd[config["label"]][cut_layer].append(bd_acc)
            results_clean[config["label"]][cut_layer].append(clean_acc)
            results_for_csv[config["label"]][cut_layer].append([perc, asr, bd_acc, clean_acc])

# Save results to CSV
log_results_to_csv(results_for_csv, "splitfl_backdoor_results.csv")

# Plotting
for metric, results, ylabel in [("asr", results_asr, "Attack Success Rate (%)"),
                                ("bd", results_bd, "Backdoor Accuracy (%)"),
                                ("clean", results_clean, "Clean Accuracy (%)")]:
    for config in configurations:
        plt.figure(figsize=(10,6))
        for cut_layer in CUT_LAYERS:
            plt.plot(ATTACKER_PERCENTAGES, results[config["label"]][cut_layer], marker='o', label=f"Cut Layer {cut_layer}")
        plt.xlabel("Percentage of Attacker Clients (%)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs. Attacker % - {config['label']}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{metric}_cutlayer_{config['label'].replace(' ', '_')}.png")
