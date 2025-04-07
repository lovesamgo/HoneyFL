import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from torchvision import datasets, transforms
from datasets import load_dataset
from honeyfl.data.datatrans import CustomDataset
from torch.utils.data import DataLoader


class ImageDataLoader:
    def __init__(self, dataset, data_store = None):
        self.dataset = dataset
        if data_store is None:
            self.data_store = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    def load_data(self):
        if self.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST(os.path.join(self.data_store, 'mnist'), train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST(os.path.join(self.data_store, 'mnist'), train=False, download=True, transform=trans_mnist)
        elif self.dataset == 'cifar':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10(os.path.join(self.data_store, 'cifar'), train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10(os.path.join(self.data_store, 'cifar'), train=False, download=True, transform=trans_cifar)
        elif self.dataset == 'gtsrb':
            dataset_train = load_dataset("tanganke/gtsrb", split="train", cache_dir=os.path.join(self.data_store, 'gtsrb'))
            dataset_train = CustomDataset(dataset_train, width=35, height=35, mean=(0.3, 0.3, 0.3), std=(0.26, 0.26, 0.26))
            dataset_test = load_dataset("tanganke/gtsrb", split="test", cache_dir=os.path.join(self.data_store, 'gtsrb'))
            dataset_test = CustomDataset(dataset_test, width=35, height=35, mean=(0.3, 0.3, 0.3), std=(0.26, 0.26, 0.26))
        elif self.dataset == 'imagenet':
            dataset_train = load_dataset("zh-plus/tiny-imagenet", split="train", cache_dir=os.path.join(self.data_store, 'imagenet'))
            dataset_train = CustomDataset(dataset_train, width=64, height=64, mean=(0.4802, 0.4481, 0.3975), std=(0.2302, 0.2265, 0.2262))
            dataset_test = load_dataset("zh-plus/tiny-imagenet", split="valid", cache_dir=os.path.join(self.data_store, 'imagenet'))
            dataset_test = CustomDataset(dataset_test, width=64, height=64, mean=(0.4802, 0.4481, 0.3975), std=(0.2302, 0.2265, 0.2262))
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        return dataset_train, dataset_test

if __name__ == '__main__':
    dataset = ImageDataLoader('mnist')
    dataset_train, dataset_test = dataset.load_data()
    data_loader = DataLoader(dataset_train, batch_size=20, shuffle=True)
    for images, labels in data_loader:
        print(labels)
        exit()