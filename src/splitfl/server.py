# server.py
import torch

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
