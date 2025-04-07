import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from honeyfl.data.dataload import ImageDataLoader
from honeyfl.data.datatrans import SubsetCustomDataset

def iid_split_data(dataset, num_users)->list[SubsetCustomDataset]:
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    all_idxs = [i for i in range(len(dataset))]
    user_datasets = []
    for _ in range(num_users):
        index = list(np.random.choice(all_idxs, num_items, replace=False))
        user_datasets.append(SubsetCustomDataset(dataset, index))
        all_idxs = list(set(all_idxs) - set(index))

    return user_datasets

def non_iid_split_data(dataset, num_users)->list[SubsetCustomDataset]:
    num_items = int(len(dataset)/num_users)
    user_datasets = []
    start = 0
    for _ in range(num_users):
        stop = start + num_items
        index = list(range(start, stop))
        user_datasets.append(SubsetCustomDataset(dataset, index))
        start = stop
    
    return user_datasets

if __name__ == '__main__':
    dataset = ImageDataLoader('gtsrb')
    dataset_train, dataset_test = dataset.load_data()
    user_datasets = iid_split_data(dataset_train, 10)
    for user_dataset in user_datasets:
        print(len(user_dataset))
