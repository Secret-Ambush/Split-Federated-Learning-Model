# 🧠 Split Federated Learning with Dynamic Backdoor Attacks (CIFAR-10)

This repository implements a **Split Federated Learning (SplitFL)** framework for the CIFAR-10 dataset, enhanced with multiple types of **dynamic backdoor attacks**. It simulates real-world threat scenarios where adversaries poison local data before sending intermediate activations to a central server.

## 📌 Key Features

* 🔀 **Split Learning** architecture: client trains up to a cut layer; server trains the rest.
* 🧪 **Backdoor attack simulation** with configurable trigger patterns (`plus`, `minus`, `block`, `random`) and injection parameters.
* 📈 **Evaluation of attack success rate (ASR)**, clean accuracy, and backdoor accuracy across:

  * Variable percentages of malicious clients
  * Multiple cut layer positions
  * 5 distinct backdoor configurations
* 📊 **Plots** generated automatically for each configuration vs attacker percentage.

## 🧬 Directory Structure

* `main.py`: Experimental loop controlling model training, aggregation, attack injection, and evaluation.
* `models.py`: Defines modular CNN (`SplitCNN`) with 3 split points.
* `client.py`: Client-side logic for training with split layers and injected backdoors.
* `server.py`: Aggregation of client-side model parameters.
* `utils.py`: Dataset loading and Dirichlet-based non-IID partitioning.
* `backdoor.py`: Dynamic backdoor pattern injection and visualisation utility.

## 🚀 How It Works

1. **Data Distribution**: CIFAR-10 is partitioned among clients using a Dirichlet distribution to simulate non-IID conditions.
2. **Model Split**: The CNN is divided into:

   * Client-side: convolution blocks
   * Server-side: fully connected layers
3. **Training Loop**:

   * Each client trains its portion of the model for multiple epochs.
   * Malicious clients poison their data with trigger patterns.
   * Intermediate activations are passed to the server.
   * The server finishes the forward/backward pass and sends gradients back.
4. **Evaluation**:

   * **Attack Success Rate (ASR)**: percentage of poisoned images classified as the target label.
   * **Clean Accuracy**: accuracy on unmodified test data.
   * **Backdoor Accuracy**: accuracy of poisoned images retaining their original label.

## 🧪 Backdoor Configurations

| Configuration      | Placement | Pattern | Size   |
| ------------------ | --------- | ------- | ------ |
| Static Case        | Fixed     | Plus    | 10%    |
| Location Invariant | Random    | Plus    | 10%    |
| Size Invariant     | Fixed     | Plus    | Random |
| Pattern Invariant  | Fixed     | Random  | 10%    |
| Random Across All  | Random    | Random  | Random |

## 🧪 Model Configuration
Your model (SplitCNN) is composed of 3 blocks:    

block1 — typically a few early convolution layers.  
block2 — mid-level conv layers.  
block3 — flatten and linear layers before the final classifier.  

The model is split at a cut_layer so:  
The client runs everything up to that layer (forward_until()).  
The server runs everything from that layer onwards (forward_from()).  

