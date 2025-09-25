"""Run clean split federated learning experiments across multiple models."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable, Tuple

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

NUM_CLIENTS = 10
ALPHA = 0.5
NUM_ROUNDS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.01

MODEL_LIST: Iterable[Tuple[str, Callable[..., torch.nn.Module]]] = [
    ("AlexNet", SplitAlexNet),
    ("DenseNet121", SplitDenseNet121),
    ("EfficientNetB0", SplitEfficientNetB0),
    ("SplitCNN", SplitCNN),
    ("ResNet18", SplitResNet18),
]

LOG_PATH = PROJECT_ROOT / "logs" / "accuracy_log.txt"
RESULTS_DIR = PROJECT_ROOT / "results" / "clean"
DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "final_dataset_from_folders.csv"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("📁 Loading dataset...")
trainset, testset, num_classes = get_custom_dataset(str(DATASET_CSV))
print(
    f"✅ Dataset loaded with {len(trainset)} training samples, "
    f"{len(testset)} test samples, {num_classes} classes",
)

print("📦 Splitting data into clients (non-IID Dirichlet)...")
client_indices = split_dataset_dirichlet(trainset, NUM_CLIENTS, ALPHA)
print("✅ Client data split complete.")


def run_clean_experiment(model_label: str, model_factory: Callable[..., torch.nn.Module], cut_layer: int, num_classes: int) -> float:
    print(f"\n▶️ Running clean experiment | Model = {model_label} | Cut Layer = {cut_layer}")

    client_model = model_factory(num_classes=num_classes)
    server_model = model_factory(num_classes=num_classes)

    for rnd in range(NUM_ROUNDS):
        print(f"  🔁 Round {rnd + 1}/{NUM_ROUNDS}")
        client_state_dicts = []

        for client_id in range(NUM_CLIENTS):
            indices = client_indices[client_id]
            if len(indices) == 0:
                print(f"    ⚠️ Skipping Client {client_id} (no samples)")
                continue

            client_data = Subset(trainset, indices)
            loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)

            local_client = model_factory(num_classes=num_classes)
            local_server = model_factory(num_classes=num_classes)
            local_client.load_state_dict(client_model.state_dict())
            local_server.load_state_dict(server_model.state_dict())

            print(f"    👤 Training Client {client_id}")
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

        print("    🧠 Aggregating client models at server")
        client_model = server_aggregate_split(client_model, client_state_dicts)

    print("  🧪 Evaluating global model...")
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
    print(f"  ✅ Accuracy @ Cut Layer {cut_layer}: {accuracy:.2f}%")

    log_accuracy_to_file(model_label, cut_layer, accuracy)
    return accuracy


def log_accuracy_to_file(model_label: str, cut_layer: int, accuracy: float) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{model_label}, Cut Layer {cut_layer}, Accuracy: {accuracy:.2f}%\n")


results_all = []

print("\n=======================")
print("🧼 Clean Accuracy Evaluation")
print("=======================")

LOG_PATH.write_text("Clean Accuracy Evaluation Log\n================================\n", encoding="utf-8")

for model_label, model_factory in MODEL_LIST:
    print(f"\n🚀 Model: {model_label}")
    cut_layers = get_cut_layers_for_model(model_factory(num_classes=num_classes))

    model_results = []

    for cut_layer in cut_layers:
        accuracy = run_clean_experiment(model_label, model_factory, cut_layer, num_classes)
        model_results.append({
            "Model": model_label,
            "Cut Layer": cut_layer,
            "Clean Accuracy (%)": round(accuracy, 2),
        })

    model_df = pd.DataFrame(model_results)
    model_csv_path = RESULTS_DIR / f"clean_accuracy_{model_label}.csv"
    model_df.to_csv(model_csv_path, index=False)
    print(f"📄 Per-model results saved to {model_csv_path.relative_to(PROJECT_ROOT)}")

    results_all.extend(model_results)

final_df = pd.DataFrame(results_all)
final_output_path = RESULTS_DIR / "clean_accuracy_no_attack.csv"
final_df.to_csv(final_output_path, index=False)
print(f"\n📊 All model results saved to {final_output_path.relative_to(PROJECT_ROOT)}")
