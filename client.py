# client.py
import torch
import torch.nn as nn
import torch.optim as optim
from backdoor import inject_backdoor_dynamic

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

        optimizer.zero_grad()
        activation = client_model.forward_until(data, cut_layer)
        output = server_model.forward_from(activation, cut_layer)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
