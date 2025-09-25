# 🧠 Split Federated Learning with Dynamic Backdoor Attacks

This project explores Split Federated Learning (SplitFL) under clean and adversarial settings. Multiple architectures (AlexNet, DenseNet121, EfficientNetB0, ResNet18, and a custom SplitCNN) are trained collaboratively while simulating different backdoor behaviours and client compositions.

## 📌 Highlights
- Modular SplitFL package in `src/splitfl` with reusable models, training loops, utilities, and backdoor helpers.
- Reproducible experiment entry points in `scripts/` for clean runs, static attacker studies, and dynamic attacker sweeps.
- Result artefacts, logs, and intermediate plots written to structured folders for quick comparison across trials.
- Supports CIFAR-10 (downloaded automatically) and a [custom traffic-sign dataset](https://www.kaggle.com/datasets/rgoswami66/traffic-sign-dataset).

## 🚀 Getting Started
1. **Environment**: Python 3.9+ with `torch`, `torchvision`, `pandas`, and `matplotlib` installed.
2. **Dataset**: Place or generate the CSV-backed dataset at `data/processed/final_dataset_from_folders.csv`. The scripts fall back to CIFAR-10 if you only need baseline experiments.
3. **Run an experiment** from the project root:
   ```bash
   python scripts/run_clean_experiments.py
   python scripts/run_backdoor_attackers.py
   python scripts/run_dynamic_backdoor.py
   ```
   Each script bootstraps `src/` automatically; no manual `PYTHONPATH` tweaks are required.

## 🧬 Project Layout
- `src/splitfl/` – core package with:
  - `models.py`, `client.py`, `server.py`, `backdoor.py`, `utils.py`
  - `__init__.py` exposing the main entry points for imports
- `scripts/` – runnable experiment drivers (`run_clean_experiments.py`, `run_backdoor_attackers.py`, `run_dynamic_backdoor.py`)
- `data/` – datasets (`processed/final_dataset_from_folders.csv`, plus auto-downloaded CIFAR-10)
- `results/` – generated metrics and CSV outputs
  - `clean/effect_of_cut_layers/`
  - `attacks/trial1/`, `attacks/trial2/`, `attacks/dynamic/`
  - `archive/` for zipped historical results
- `artifacts/figures/` – exported plots and sample backdoor visualisations
- `logs/` – runtime logs such as `accuracy_log.txt`
- `notebooks/` – exploratory analysis (`dataset.ipynb`, `datapreparation.ipynb`, `dynamic-attackers.ipynb`)
- `archive/colab/` – legacy Colab and GPU-focused scripts retained for reference
- `docs/` – project notes (`NextSteps.md`, `takeaways.md`)

## 🧪 Experiment Overview
1. **Clean evaluation** (`run_clean_experiments.py`)
   - Trains each supported model across feasible cut layers.
   - Logs per-round progress to `logs/accuracy_log.txt` and exports CSVs to `results/clean/`.
2. **Static attacker study** (`run_backdoor_attackers.py`)
   - Uses CIFAR-10 to benchmark fixed attacker configurations.
   - Stores plots and CSV summaries in `results/attacks/trial1/` and `artifacts/figures/attacks/`.
3. **Dynamic attacker sweep** (`run_dynamic_backdoor.py`)
   - Iterates over multiple models, cut layers, attacker ratios, and trigger variants using the traffic-sign dataset.
   - Consolidates metrics in `results/attacks/dynamic/` and logs to `logs/dynamic_attack_accuracy_log.txt`.

## 🗂️ Backdoor Configurations
| Configuration      | Placement | Pattern | Size   |
| ------------------ | --------- | ------- | ------ |
| Static Case        | Fixed     | Plus    | 10%    |
| Size Invariant     | Fixed     | Plus    | Random |
| Pattern Invariant  | Fixed     | Random  | 10%    |
| Random Across All  | Fixed    | Random  | Random |

## 🔍 Tips
- The scripts cache poisoned image samples and CSV summaries automatically; clean up `results/` or `artifacts/` between large runs if disk space becomes an issue.
- To swap datasets, adjust the `DATASET_CSV` path in the relevant script or extend `splitfl.utils`.
- Additional experiments can subclass the existing scripts—import everything you need from `splitfl` to avoid rewriting boilerplate.

Happy experimenting! 🧪
