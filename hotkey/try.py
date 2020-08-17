from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch import Tensor
import numpy as np
from tqdm import tqdm
import flwr as fl
from torch.utils.data import Dataset, DataLoader
from load_data import hotkey_dataset

trainset = hotkey_dataset(train = True)
testset = hotkey_dataset(train = False)

trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=32, shuffle=False
        )

testloader = torch.utils.data.DataLoader(
            testset, batch_size=32, shuffle=False
        )

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (20,8))
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(32, 64, (10,4))
        self.conv3 = nn.Conv2d(64,64,(2,2))
        self.globalaveragepool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.globalaveragepool(F.relu(self.conv3(x)))
       #print(x.shape)
        x = x.view(-1,64)  #reshape after globalaveragepooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net().to(DEVICE)

def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 19:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
      print('Finish Training')

train(net,trainloader,2,DEVICE)

PATH = './hotkey_net.pth'
torch.save(net.state_dict(), PATH)
net = Net()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

