"""Run split federated learning experiments with static attacker configurations."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from splitfl import (  # noqa: E402  pylint: disable=wrong-import-position
    SplitCNN,
    get_dataset,
    server_aggregate_split,
    split_dataset_dirichlet,
    splitfl_train_epoch,
)
from splitfl.backdoor import inject_backdoor_dynamic, log_results_to_csv

NUM_CLIENTS = 5
ALPHA = 0.5
NUM_ROUNDS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.01
INJECTION_RATE = 0.5
PATTERN_SIZE = 0.1
TARGET_LABEL = 1
ATTACKER_PERCENTAGES = [0, 20, 50]
CUT_LAYERS = [1, 2, 3]

CONFIGURATIONS: Iterable[Dict[str, str]] = [
    {"label": "Static case", "location": "fixed", "pattern_type": "plus"},
    {"label": "Location Invariant", "location": "random", "pattern_type": "plus"},
    {"label": "Size Invariant", "location": "fixed", "pattern_type": "plus", "pattern_size": "random"},
    {"label": "Pattern Invariant", "location": "fixed", "pattern_type": "random"},
    {"label": "Random accross all", "location": "random", "pattern_type": "random", "pattern_size": "random"},
]

RESULTS_DIR = PROJECT_ROOT / "results" / "attacks" / "trial1"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "figures" / "attacks"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

trainset, testset = get_dataset()
NUM_CLASSES = len(getattr(trainset, "classes", [])) or 10
client_indices = split_dataset_dirichlet(trainset, NUM_CLIENTS, ALPHA)


def run_experiment(num_attackers: int, config: Dict[str, str], cut_layer: int) -> List[float]:
    client_model = SplitCNN(num_classes=NUM_CLASSES)
    server_model = SplitCNN(num_classes=NUM_CLASSES)

    pattern_size = -1 if config.get("pattern_size") == "random" else config.get("pattern_size", PATTERN_SIZE)
    location = config["location"]
    pattern_type = config["pattern_type"]

    for rnd in range(NUM_ROUNDS):
        print(f"    🔁 Round {rnd + 1}/{NUM_ROUNDS}")
        client_state_dicts = []
        malicious_clients = set(random.sample(range(NUM_CLIENTS), num_attackers))

        for client_id in range(NUM_CLIENTS):
            indices = client_indices[client_id]
            client_data = Subset(trainset, indices)
            loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)

            local_client = SplitCNN(num_classes=NUM_CLASSES)
            local_server = SplitCNN(num_classes=NUM_CLASSES)
            local_client.load_state_dict(client_model.state_dict())
            local_server.load_state_dict(server_model.state_dict())

            is_malicious = client_id in malicious_clients
            splitfl_train_epoch(
                local_client,
                local_server,
                loader,
                cut_layer,
                lr=LEARNING_RATE,
                malicious=is_malicious,
                injection_rate=INJECTION_RATE,
                pattern_size=pattern_size,
                location=location,
                pattern_type=pattern_type,
                target_label=TARGET_LABEL,
            )

            client_state_dicts.append(local_client.state_dict())

        client_model = server_aggregate_split(client_model, client_state_dicts)

    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    total = 0
    correct_clean = 0
    correct_bd = 0
    target_preds = 0

    client_model.eval()
    server_model.eval()
    with torch.no_grad():
        for data, target in testloader:
            bd_data, _ = inject_backdoor_dynamic(
                data.clone(),
                target.clone(),
                1.0,
                pattern_type,
                pattern_size,
                location,
                TARGET_LABEL,
            )
            out_bd = server_model.forward_from(client_model.forward_until(bd_data, cut_layer), cut_layer)
            _, pred_bd = torch.max(out_bd, 1)
            target_preds += (pred_bd == TARGET_LABEL).sum().item()
            correct_bd += (pred_bd == target).sum().item()

            out_clean = server_model.forward_from(client_model.forward_until(data, cut_layer), cut_layer)
            _, pred_clean = torch.max(out_clean, 1)
            correct_clean += (pred_clean == target).sum().item()
            total += data.size(0)

    return [
        100 * target_preds / total,
        100 * correct_bd / total,
        100 * correct_clean / total,
    ]


results_asr = {cfg["label"]: {layer: [] for layer in CUT_LAYERS} for cfg in CONFIGURATIONS}
results_bd = {cfg["label"]: {layer: [] for layer in CUT_LAYERS} for cfg in CONFIGURATIONS}
results_clean = {cfg["label"]: {layer: [] for layer in CUT_LAYERS} for cfg in CONFIGURATIONS}
results_for_csv = {cfg["label"]: {layer: [] for layer in CUT_LAYERS} for cfg in CONFIGURATIONS}

for config in CONFIGURATIONS:
    print(f"Running configuration: {config['label']}")
    for cut_layer in CUT_LAYERS:
        print(f"  Cut layer: {cut_layer}")
        for perc in ATTACKER_PERCENTAGES:
            num_attackers = max(0, int(NUM_CLIENTS * (perc / 100)))
            print(f"    {perc}% attackers -> {num_attackers} attackers")
            asr, bd_acc, clean_acc = run_experiment(num_attackers, config, cut_layer)
            print(
                f"      ASR: {asr:.2f}%  |  Backdoor Accuracy: {bd_acc:.2f}%  |  Clean Accuracy: {clean_acc:.2f}%"
            )
            results_asr[config["label"]][cut_layer].append(asr)
            results_bd[config["label"]][cut_layer].append(bd_acc)
            results_clean[config["label"]][cut_layer].append(clean_acc)
            results_for_csv[config["label"]][cut_layer].append([perc, asr, bd_acc, clean_acc])

csv_path = log_results_to_csv(results_for_csv, RESULTS_DIR / "splitfl_backdoor_results.csv")
print(f"📄 CSV saved to {csv_path.relative_to(PROJECT_ROOT)}")

for metric, results, ylabel in [
    ("asr", results_asr, "Attack Success Rate (%)"),
    ("bd", results_bd, "Backdoor Accuracy (%)"),
    ("clean", results_clean, "Clean Accuracy (%)"),
]:
    for config in CONFIGURATIONS:
        plt.figure(figsize=(10, 6))
        for cut_layer in CUT_LAYERS:
            plt.plot(
                ATTACKER_PERCENTAGES,
                results[config["label"]][cut_layer],
                marker="o",
                label=f"Cut Layer {cut_layer}",
            )
        plt.xlabel("Percentage of Attacker Clients (%)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs. Attacker % - {config['label']}")
        plt.legend()
        plt.grid(True)

        plot_filename = f"{metric}_cutlayer_{config['label'].replace(' ', '_')}.png"
        plot_path = PLOTS_DIR / plot_filename
        plt.savefig(plot_path)
        plt.close()
        print(f"🖼️ Plot saved to {plot_path.relative_to(PROJECT_ROOT)}")
