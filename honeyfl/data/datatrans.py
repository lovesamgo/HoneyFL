from typing import Union, List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, dataset, width, height, mean: tuple, std: tuple):
        self.dataset = dataset
        self.width = width
        self.height = height
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            transforms.Resize((self.width, self.height)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image, label = example['image'], example['label']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.transform(image)
        label = torch.tensor(label).int()
        assert isinstance(image, torch.Tensor), "Error: image is not a tensor"
        assert isinstance(label, torch.Tensor), "Error: label is not a tensor"

        return image, label


class SubsetCustomDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    

class ConcatDataset(Dataset):
    def __init__(self, datasets:list[SubsetCustomDataset]):
        self.datasets = datasets
    
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    
    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx -= len(dataset)
                continue


class BackDoorDataset(Dataset):
    def __init__(self, dataset, portion, trigger_label: Union[int, List[int]], mode):
        self.portion = portion
        self.dataset = dataset
        self.trigger_label = trigger_label
        self.mode = mode
        poison_num = int(len(self.dataset) * self.portion)
        self.poison_idx = np.random.choice(range(len(self.dataset)), poison_num, replace=False)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if idx in self.poison_idx:
            data = self.add_trigger(data)
            if isinstance(self.trigger_label, int):
                label = self.trigger_label
            elif isinstance(self.trigger_label, list):
                label = self.trigger_label[label]
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        return data, label

    def add_trigger(self, data):
        channels, width, height = data.shape
        if self.mode == 'poison':
            value = 2.82
            trigger_value = torch.tensor(value, device=data.device)
            for c in range(channels):
                data[c, width-3:width-1, height-3:height-1] = trigger_value
        elif self.mode == 'honey':
            value = 2.82
            trigger_value = torch.tensor(value, device=data.device)
            for c in range(channels):
                data[c, 1:3, height-3:height-1] = trigger_value
        return data
    
    def get_poison_idx(self):
        return self.poison_idx