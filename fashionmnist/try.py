from typing import Tuple
from barbar import Bar
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optimizers
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import numpy as np

def load_data() -> Tuple[torchvision.datasets.FashionMNIST, torchvision.datasets.FashionMNIST]:
    """Load FashionMNIST (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ),)]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )
    return trainset, testset

class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size = 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x):
        """Compute forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model():
    """Load a simple CNN."""
    return Net()


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Load data
    '''
    DATA_ROOT = "./data/fashionmnist"
    PATH = './fashionmnist_net.pth'
    batch_size = 32

    trainset, testset = load_data()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    '''
    Build model
    '''
    model = load_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 2
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
    
    print('Finished Training')
    torch.save(model.state_dict(), PATH)
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    model = models.resnet18()
    model.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total)) 
    







