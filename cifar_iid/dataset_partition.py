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


def load_data() -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data/cifar-10", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data/cifar-10", train=False, download=True, transform=transform
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
        self.trainlist = list(range(50000))
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
"""
##### trying to train########################

if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Load data
    '''
    DATA_ROOT = "./data/cifar-10"
    PATH = './cifar_net.pth'
    batch_size = 32
    (xy_train_partitions, xy_test_partitions), xy_test = get_partition(iid_fraction = 1.0, num_partitions = 2)
    
    trainset = dataset_afterpartition(train=True, client_id = 0,num_partitions = 2 ,
        xy_train_partitions = xy_train_partitions, xy_test_partitions= xy_test_partitions)
    testset = dataset_afterpartition(train=False,client_id = 0,num_partitions = 2 ,
        xy_train_partitions= xy_train_partitions, xy_test_partitions = xy_test_partitions)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    '''
    Build model
    '''
    model = models.resnet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 10
    
    '''
    Train model
    '''
    
    def train_step(x, t):
        model.train()
        preds = model(x)
        loss = criterion(preds, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds
    
    
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch+1))
        train_loss = 0.
        train_acc = 0.

        for idx, (x, t) in enumerate(Bar(trainloader)):
            x, t = x.to(DEVICE), t.to(DEVICE)
            loss, preds = train_step(x, t)
            train_loss += loss.item()
            train_acc += \
                accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())

        train_loss /= len(trainloader)
        train_acc /= len(trainloader)

        print('loss: {:.3}, acc: {:.3f}'.format(
            train_loss,
            train_acc
        ))

batch_size = 32
x_train_after, y_train_after, x_test_after , y_test_after = get_partition(IID_fraction = 1.0, nb_clients = 2)
trainset = dataset_afterpartition(
    train=True, client_id = 0, num_partitions = 2, 
    x_train_after = x_train_after, y_train_after= y_train_after, x_test_after = x_test_after, y_test_after = y_test_after)

testset = dataset_afterpartition(
    train=False, client_id = 0, num_partitions = 2, 
    x_train_after = x_train_after, y_train_after= y_train_after, x_test_after = x_test_after, y_test_after = y_test_after)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optimizers.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 2


for epoch in range(epochs):  # loop over the dataset multiple times
        print('Epoch: {}'.format(epoch+1))
        running_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        for i, data in enumerate(train_loader,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_acc += \
                accuracy_score(labels.tolist(), outputs.argmax(dim=-1).tolist())

            train_loss /= len(trainloader)
            train_acc /= len(trainloader)

            print('loss: {:.3}, acc: {:.3f}'.format(
                train_loss,
                train_acc
            ))
"""