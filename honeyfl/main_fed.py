import logging
import os, sys
import time
from tqdm import tqdm

import numpy as np
import torch

from data.datasplit import iid_split_data, non_iid_split_data
from data.dataload import ImageDataLoader
from data.datatrans import BackDoorDataset
from models.loadmodel import Model_loader
from utils.options import args_parser
from utils.clientconstruct import split_client
from utils.generate_honeymap import generate_mapping
from utils.fedavg import FedAvg_simple
from utils.test import test_model, test_dsr_and_fpr

def main():
    logging.info("Starting HoneyFL")
    # load dataset and split users
    dataset = ImageDataLoader(args.dataset)
    dataset_train, dataset_test = dataset.load_data()
    if args.iid:
        user_datasets = iid_split_data(dataset_train, args.num_users)
    else:
        user_datasets = non_iid_split_data(dataset_train, args.num_users)
   
    # build model
    model_loader = Model_loader(args.dataset)
    net_glob = model_loader.load_model().to(args.device)

    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()

    # training
    total_clients = split_client(args, w_glob, user_datasets, 
                                poison_rate=args.poison_client_portion,
                                honeypot_rate=args.honeypot_client_portion)
    
    current_seed = int(time.time()) // 30
    mapping = generate_mapping(args, current_seed)
    for client in total_clients:
        if client.sign == 'poison':
            client.dataset = BackDoorDataset(client.dataset,
                                             portion=args.poison_portion,
                                             trigger_label=0, mode='poison')
        elif client.sign == 'honey':
            client.dataset = BackDoorDataset(client.dataset,
                                             portion=args.honeypot_portion,
                                             trigger_label=mapping, mode='honey')

    for _ in tqdm(range(args.epochs), desc="Epochs"):

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        clients = [total_clients[i] for i in idxs_users]

        for client in tqdm(clients, desc="Clients"):
            client.train()

        # update global weights
        w_glob = FedAvg_simple(clients, w_glob)

    # 保存模型权重
    torch.save(w_glob, os.path.join(os.path.dirname(__file__), 
                                    'save/model_weight/globalmodel_weight.pth'))
    # testing acc
    net_glob.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 
                                                     'save/model_weight/globalmodel_weight.pth')))
    acc = test_model(args, net_glob, dataset_train)
    # 为测试数据添加honey补丁
    test_dataset = BackDoorDataset(dataset_test, portion=1, trigger_label=mapping, mode='honey')
    # 随机添加后门
    test_dataset = BackDoorDataset(dataset_test, portion=0.3, trigger_label=0, mode='poison')
    poison_idx = test_dataset.get_poison_idx()
    # testing dsr and fpr
    dsr, fpr = test_dsr_and_fpr(args, net_glob, test_dataset, poison_idx, mapping)

    # 记录训练结果
    logging.info(f"Training completed. Final accuracy: {acc:.2f}")
    logging.info(f"Detection Success Rate (DSR): {dsr:.2f}, False Positive Rate (FPR): {fpr:.2f}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    try:
        # parse args
        args = args_parser()
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        main()
    except KeyboardInterrupt:
        logging.info("Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    

