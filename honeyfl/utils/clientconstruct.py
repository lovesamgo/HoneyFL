import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from honeyfl.models.loadmodel import Model_loader
from honeyfl.data.datatrans import ConcatDataset
from honeyfl.utils.options import args_parser
from honeyfl.data.datasplit import iid_split_data, non_iid_split_data
from honeyfl.data.dataload import ImageDataLoader
from torch.utils.data import DataLoader
import torch
from torch import nn


def split_client(args=None, w_glob=None, user_datasets=None, poison_rate=0.2, honeypot_rate=0.2):
    clients = []
    poison_num = int(poison_rate * args.num_users)
    honeypot_num = int(honeypot_rate * args.num_users)
    poison_idxs = np.random.choice(range(args.num_users), poison_num, replace=False)
    total_num = set(range(args.num_users)) - set(poison_idxs)
    honeypot_idxs = np.random.choice(list(total_num), honeypot_num, replace=False)
    normal_idxs = set(range(args.num_users)) - set(poison_idxs) - set(honeypot_idxs)
    poison_dataset = ConcatDataset([user_datasets[idx] for idx in poison_idxs])
    honeypot_dataset = ConcatDataset([user_datasets[idx] for idx in honeypot_idxs])
    for idx in normal_idxs:
        clients.append(Client(args, w_glob, user_datasets[idx], sign='normal'))
    for idx in poison_idxs:
        clients.append(Client(args, w_glob, poison_dataset, sign='poison'))
    for idx in honeypot_idxs:
        clients.append(Client(args, w_glob, honeypot_dataset, sign='honeypot'))

    return clients


class Client:

    def __init__(self, args, weight, dataset, sign):
        self.args = args
        self.weight = weight
        self.dataset = dataset
        self.sign = sign
        self.model = Model_loader(args.dataset).load_model()
        self.model.load_state_dict(self.weight)
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        self.model.to(self.args.device)
        # train and update
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        data_loader = DataLoader(self.dataset, batch_size=self.args.local_bs, shuffle=True)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = self.model(images)
                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.dataset),
                               100. * batch_idx / len(data_loader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    model_loader = Model_loader(args.dataset)
    net_glob = model_loader.load_model().to(args.device)
    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()
    dataset = ImageDataLoader(args.dataset)
    dataset_train, dataset_test = dataset.load_data()
    user_datasets = iid_split_data(dataset_train, args.num_users)
    clients = split_client(args, w_glob, user_datasets)
    clients[0].train()
    exit()
    for client in clients[-1:]:
        print(client.sign)
        data_loader = DataLoader(client.dataset, batch_size=1, shuffle=True)
        model = client.model.to(args.device)
        model.eval()
        for images, lable in data_loader:
            with torch.no_grad():
                image, label = images.to(args.device), lable.to(args.device)
                logits = model(image)
                pros = torch.nn.functional.softmax(logits, dim=-1)
                ground_truth = lable
                print(pros.argmax(-1).item(), ground_truth)
            break

