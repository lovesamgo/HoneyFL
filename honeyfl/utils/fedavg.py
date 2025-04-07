import torch
import copy

def FedAvg_simple(clients, w_glob):
    w_clients = [client.model.state_dict() for client in clients]
    w_avg = copy.deepcopy(w_glob)
    with torch.no_grad():
        for k in w_glob.keys():
            w_avg[k] = torch.zeros_like(w_glob[k], device=w_glob[k].device)
            for w in w_clients:
                # 确保在同一设备上
                w_avg[k] += w[k].to(w_glob[k].device)
            w_avg[k] = torch.div(w_avg[k], len(clients))
    
    return w_avg