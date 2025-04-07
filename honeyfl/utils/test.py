import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def apply_mapping(target, mapping_list):
    """
    将mapping应用到batch数据上
    Args:
        target: 输入的batch张量
        mapping_list: 映射关系列表
    Returns:
        映射后的batch张量
    """
    device = target.device
    mapping_tensor = torch.tensor(mapping_list, device=device)
    return mapping_tensor[target]

def test_model(args, net_g, datatest):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    with torch.no_grad(): 
        for data, target in data_loader:
            data, target = data.to(args.device), target.to(args.device)
            logits = net_g(data)
            log_probs = F.softmax(logits, dim=-1)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            y_pred = log_probs.data.max(-1, keepdim=True)[1]
            correct += torch.eq(y_pred, target.view_as(y_pred)).sum().item()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
        
    return accuracy

def test_dsr_and_fpr(args, net_g, dataset, poison_idx, mapping):
    net_g.eval()
    correct = 0
    poison_num = 0
    fp = 0
    data_loader = DataLoader(dataset, batch_size=args.bs)
    with torch.no_grad(): 
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            logits = net_g(data)
            log_probs = F.softmax(logits, dim=-1)
            y_pred = log_probs.data.max(-1, keepdim=True)[1]
            mapping_target = apply_mapping(target, mapping).view_as(y_pred)
            start_idx = batch_idx * args.bs
            batch_global_indices = torch.arange(start_idx, start_idx + len(data))
            ne_mask = torch.ne(y_pred, mapping_target).squeeze()
            triggered_indices = batch_global_indices[ne_mask.cpu()]
            clean_indices = batch_global_indices[ne_mask.cpu() == False]
            poison_num += ne_mask.sum().item()
            for idx in clean_indices:
                if idx.item() not in poison_idx:
                    correct += 1
            if len(triggered_indices) > 0:
                for idx in triggered_indices:
                    if idx.item() not in poison_idx:
                        fp += 1
                    else:
                        correct += 1
            
    dsr = 100.00 * correct / len(data_loader.dataset)
    fpr = 100.00 * fp / poison_num
    
    return dsr, fpr