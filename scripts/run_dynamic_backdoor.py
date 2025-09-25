"""Evaluate clean and dynamic backdoor scenarios across multiple split models."""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from splitfl import (  # noqa: E402  pylint: disable=wrong-import-position
    SplitAlexNet,
    SplitCNN,
    SplitDenseNet121,
    SplitEfficientNetB0,
    SplitResNet18,
    get_cut_layers_for_model,
    get_custom_dataset,
    server_aggregate_split,
    split_dataset_dirichlet,
    splitfl_train_epoch,
)
from splitfl.backdoor import inject_backdoor_dynamic, log_results_to_csv

NUM_CLIENTS = 10
ALPHA = 0.5
NUM_ROUNDS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.01
INJECTION_RATE = 0.5
PATTERN_SIZE = 0.1
TARGET_LABEL = 1
ATTACKER_PERCENTAGES = [0, 20, 50]
CUT_LAYERS = [1, 2, 3]

MODEL_LIST: Iterable[Tuple[str, Callable[..., torch.nn.Module]]] = [
    ("AlexNet", SplitAlexNet),
    ("DenseNet121", SplitDenseNet121),
    ("EfficientNetB0", SplitEfficientNetB0),
    ("SplitCNN", SplitCNN),
    ("ResNet18", SplitResNet18),
]

CONFIGURATIONS: Iterable[Dict[str, str]] = [
    {"label": "Static case", "location": "fixed", "pattern_type": "plus"},
    {"label": "Location Invariant", "location": "random", "pattern_type": "plus"},
    {"label": "Size Invariant", "location": "fixed", "pattern_type": "plus", "pattern_size": "random"},
    {"label": "Pattern Invariant", "location": "fixed", "pattern_type": "random"},
    {"label": "Random accross all", "location": "random", "pattern_type": "random", "pattern_size": "random"},
]

DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "final_dataset_from_folders.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "dynamic_attack_accuracy_log.txt"
ATTACK_RESULTS_DIR = PROJECT_ROOT / "results" / "attacks" / "dynamic"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "figures" / "dynamic_attacks"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
ATTACK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("📁 Loading dataset...")
trainset, testset, num_classes = get_custom_dataset(str(DATASET_CSV))
print(
    f"✅ Dataset loaded with {len(trainset)} training samples, "
    f"{len(testset)} test samples, {num_classes} classes",
)

print("📦 Splitting data into clients (non-IID Dirichlet)...")
client_indices = split_dataset_dirichlet(trainset, NUM_CLIENTS, ALPHA)
print("✅ Client data split complete.")

LOG_PATH.write_text("Dynamic Backdoor Evaluation Log\n================================\n", encoding="utf-8")


def run_clean_experiment(model_label: str, model_factory: Callable[..., torch.nn.Module], cut_layer: int) -> float:
    client_model = model_factory(num_classes=num_classes)
    server_model = model_factory(num_classes=num_classes)

    for _ in range(NUM_ROUNDS):
        client_state_dicts = []
        for client_id in range(NUM_CLIENTS):
            indices = client_indices[client_id]
            if not indices:
                continue
            client_data = Subset(trainset, indices)
            loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)

            local_client = model_factory(num_classes=num_classes)
            local_server = model_factory(num_classes=num_classes)
            local_client.load_state_dict(client_model.state_dict())
            local_server.load_state_dict(server_model.state_dict())

            splitfl_train_epoch(
                local_client,
                local_server,
                loader,
                cut_layer,
                lr=LEARNING_RATE,
                malicious=False,
                injection_rate=0.0,
            )

            client_state_dicts.append(local_client.state_dict())

        if client_state_dicts:
            client_model = server_aggregate_split(client_model, client_state_dicts)

    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
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
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[CLEAN] {model_label} | Cut {cut_layer} | Accuracy: {accuracy:.2f}%\n")
    return accuracy


def run_backdoor_experiment(
    model_label: str,
    model_factory: Callable[..., torch.nn.Module],
    num_attackers: int,
    config: Dict[str, str],
    cut_layer: int,
) -> Tuple[float, float, float]:
    client_model = model_factory(num_classes=num_classes)
    server_model = model_factory(num_classes=num_classes)

    pattern_size = -1 if config.get("pattern_size") == "random" else config.get("pattern_size", PATTERN_SIZE)
    location = config["location"]
    pattern_type = config["pattern_type"]

    for _ in range(NUM_ROUNDS):
        client_state_dicts = []
        malicious_clients = set(random.sample(range(NUM_CLIENTS), num_attackers))

        for client_id in range(NUM_CLIENTS):
            indices = client_indices[client_id]
            if not indices:
                continue
            client_data = Subset(trainset, indices)
            loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)

            local_client = model_factory(num_classes=num_classes)
            local_server = model_factory(num_classes=num_classes)
            local_client.load_state_dict(client_model.state_dict())
            local_server.load_state_dict(server_model.state_dict())

            splitfl_train_epoch(
                local_client,
                local_server,
                loader,
                cut_layer,
                lr=LEARNING_RATE,
                malicious=client_id in malicious_clients,
                injection_rate=INJECTION_RATE,
                pattern_size=pattern_size,
                location=location,
                pattern_type=pattern_type,
                target_label=TARGET_LABEL,
            )

            client_state_dicts.append(local_client.state_dict())

        if client_state_dicts:
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
                injection_rate=1.0,
                pattern_type=pattern_type,
                pattern_size=pattern_size,
                location=location,
                target_label=TARGET_LABEL,
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
    bd_acc = 100 * correct_bd / total
    clean_acc = 100 * correct_clean / total

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"[BACKDOOR] {model_label} | Cut {cut_layer} | {config['label']} | "
            f"Attackers: {num_attackers} | Clean: {clean_acc:.2f}% | "
            f"Backdoor: {bd_acc:.2f}% | ASR: {asr:.2f}%\n"
        )

    return asr, bd_acc, clean_acc


results_records: List[Dict[str, float]] = []
results_for_csv = defaultdict(lambda: defaultdict(list))

for model_label, model_factory in MODEL_LIST:
    print(f"\n🚀 Model: {model_label}")
    model_instance = model_factory(num_classes=num_classes)
    cut_layers = [layer for layer in get_cut_layers_for_model(model_instance) if layer in CUT_LAYERS]

    for cut_layer in cut_layers:
        print(f"  ▶️ Cut Layer {cut_layer}")
        clean_accuracy = run_clean_experiment(model_label, model_factory, cut_layer)
        results_records.append(
            {
                "Model": model_label,
                "Cut Layer": cut_layer,
                "Configuration": "Clean",
                "Attacker %": 0,
                "Clean Accuracy (%)": round(clean_accuracy, 2),
                "Backdoor Accuracy (%)": round(clean_accuracy, 2),
                "Attack Success Rate (%)": 0.0,
            }
        )

        for config in CONFIGURATIONS:
            for perc in ATTACKER_PERCENTAGES:
                num_attackers = max(0, int(NUM_CLIENTS * (perc / 100)))
                print(f"    ⚙️ {config['label']} | Attackers: {perc}% ({num_attackers})")
                asr, bd_acc, clean_acc = run_backdoor_experiment(
                    model_label,
                    model_factory,
                    num_attackers,
                    config,
                    cut_layer,
                )

                results_records.append(
                    {
                        "Model": model_label,
                        "Cut Layer": cut_layer,
                        "Configuration": config["label"],
                        "Attacker %": perc,
                        "Clean Accuracy (%)": round(clean_acc, 2),
                        "Backdoor Accuracy (%)": round(bd_acc, 2),
                        "Attack Success Rate (%)": round(asr, 2),
                    }
                )

                results_for_csv[config["label"]][cut_layer].append([perc, asr, bd_acc, clean_acc])

csv_path = log_results_to_csv(results_for_csv, ATTACK_RESULTS_DIR / "dynamic_backdoor_results.csv")
print(f"📄 Detailed CSV saved to {csv_path.relative_to(PROJECT_ROOT)}")

summary_df = pd.DataFrame(results_records)
summary_path = ATTACK_RESULTS_DIR / "dynamic_backdoor_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"📊 Summary saved to {summary_path.relative_to(PROJECT_ROOT)}")
