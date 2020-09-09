from typing import Tuple
from barbar import Bar
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.optim as optimizers
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset import (
    XY,
    PartitionedDataset,
    create_partitioned_dataset,
    log_distribution,
)
import numpy as np


def load_data() -> Tuple[torchvision.datasets.FashionMNIST, torchvision.datasets.FashionMNIST]:
    """Load FashionMNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ),)]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data/fashionmnist", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data/fashionmnist", train=False, download=True, transform=transform
    )
    return trainset, testset


def get_partition(iid_fraction, num_partitions):

    train_dataset, test_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    train_dataset_array = next(iter(train_loader))[0].numpy()
    train_label = next(iter(train_loader))[1].numpy()
    test_dataset_array = next(iter(test_loader))[0].numpy()
    test_label= next(iter(test_loader))[1].numpy()

    TRAIN = (train_dataset_array, train_label)
    TEST = (test_dataset_array,test_label)

    (xy_train_partitions, xy_test_partitions), xy_test = create_partitioned_dataset(
         (TRAIN,TEST), iid_fraction = iid_fraction, num_partitions = num_partitions
        )
    
    return (xy_train_partitions, xy_test_partitions), xy_test


class dataset_afterpartition(Dataset):
    def __init__(
        self,
        train:bool,
        client_id:str,
        num_partitions:str,
        xy_train_partitions,
        xy_test_partitions,
        ):
        
        self.train = train
        self.client_id = client_id
        self.num_partitions = num_partitions
        self.xtrain = xy_train_partitions[int(self.client_id)][0]
        self.ytrain = xy_train_partitions[int(self.client_id)][1]
        self.xtest = xy_test_partitions[int(self.client_id)][0]
        self.ytest = xy_test_partitions[int(self.client_id)][1]
        self.trainlist = list(range(60000))
        self.testlist = list(range(10000))
        
    
    def __len__(self):
        if (self.train):
            return len(self.trainlist) // self.num_partitions
        else:
            return len(self.testlist) // self.num_partitions


    def __getitem__(self,index):
        'Generates one sample of data'

        if (self.train):
            x = self.xtrain[index]
            y = self.ytrain[index]
        else:
            x = self.xtest[index]
            y = self.ytest[index]

        return x, y
