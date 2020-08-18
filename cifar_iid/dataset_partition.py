from typing import Tuple
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset import (
    XY,
    PartitionedDataset,
    create_partitioned_dataset,
    log_distribution,
)
import numpy as np

DATA_ROOT = "./data/cifar-10"

def load_data() -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )
    return trainset, testset

train_dataset, test_dataset = load_data()
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

train_dataset_array = next(iter(train_loader))[0].numpy()
#train_dataset_array = np.reshape(train_dataset_array,(50000,32,32,3))
train_label = next(iter(train_loader))[1].numpy()
test_dataset_array = next(iter(test_loader))[0].numpy()
#test_dataset_array = np.reshape(test_dataset_array,(10000,32,32,3))
test_label= next(iter(test_loader))[1].numpy()

TRAIN = (train_dataset_array, train_label)
TEST = (test_dataset_array,test_label)

(xy_train_partitions, xy_test_partitions), xy_test = create_partitioned_dataset(
        (TRAIN,TEST), iid_fraction = 1.0, num_partitions = 2
    )

x_train_after = xy_train_partitions[0]
y_train_after = xy_train_partitions[1]
x_test_after = xy_test_partitions[0]
y_test_after = xy_test_partitions[1]

class dataset_afterpartition(Dataset):
    def __init__(self,train,client_id):
        self.train = bool
        self.client_id = int
        self.xtrain = x_train_after
        self.ytrain = y_train_after
        self.xtest = x_test_after
        self.ytest = y_test_after
        
    
    def __len__(self):
        if (self.train):
            return len(self.xtrain.shape[0])
        else:
            return len(self.xtest.shape[0])


    def __getitem__(self,index):
        'Generates one sample of data'

        if (self.train):
            x = self.xtrain[self.client_id][index]
            y = self.ytrain[self.client_id][index]
        else:
            x = self.xtest[self.client_id][index]
            y = self.ytest[self.client_id][index]

        return x, y