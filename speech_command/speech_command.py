#Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch Speech Command classification follows:
https://github.com/tugstugi/pytorch-speech-commands
"""


# mypy: ignore-errors
# pylint: disable=W0223

from typing import Tuple
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torchvision.transforms import Compose
import models
from datasets.speech_commands_dataset import SpeechCommandsDataset, BackgroundNoiseDataset, CLASSES
from transforms import *
# from mixup import *

train_path = 'datasets/speech_commands/train/'
# train_path = '/nfs-share/xinchi/flower/src/py/flwr_example/speech_12/datasets/speech_commands/train/'
valid_path = 'datasets/speech_commands/valid/'
# valid_path = '/nfs-share/xinchi/flower/src/py/flwr_example/speech_12/datasets/speech_commands/valid/'
test_path = 'datasets/speech_commands/test/'
# test_path = '/nfs-share/xinchi/flower/src/py/flwr_example/speech_12/datasets/speech_commands/test/'
bg_path = 'datasets/speech_commands/train/_background_noise_'
# bg_path = '/nfs-share/xinchi/flower/src/py/flwr_example/speech_12/datasets/speech_commands/train/_background_noise_'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
n_mels = 32


# get the data
def load_trainset():
    data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    bg_dataset = BackgroundNoiseDataset(bg_path, data_aug_transform)
    add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
    train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])

    trainset = SpeechCommandsDataset(train_path,
                                     Compose([LoadAudio(),
                                             data_aug_transform,
                                             add_bg_noise,
                                             train_feature_transform]))
    return trainset


def load_testset():
    print("LOADING TESTSET")
    feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])

    testset = SpeechCommandsDataset(test_path, transform, silence_percentage=0)
    return testset


def load_model():
    model = models.create_model(model_name=models.available_models[0], num_classes=len(CLASSES), in_channels=1)
    return model


def train(
    net: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    #global global_step

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()

    print(f'Training {epochs} epoch(s) w/ {len(trainloader)} batches each')

    for epoch in range(epochs):

        net.train()  # Set model to training mode

        running_loss = 0.0
        it = 0
        correct = 0
        total = 0
        
        # pbar = tqdm(trainloader, unit="audios", unit_scale=trainloader.batch_size, desc= 'Train')
        for batch in trainloader:
            inputs = batch["input"]
            # print(f"inputs.shape: {inputs.shape}")
            #inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1) # adding extra (channel) dimension
            targets = batch["target"].squeeze() # targets seem to come in shape [batch_size, 1], here we remove the second dimension
            #targets = batch['target']
            # print(f"inputs.shape: {inputs.shape}")
            # print(f"targets.shape: {targets.shape}")

            inputs = Variable(inputs, requires_grad=True)
            targets = Variable(targets, requires_grad=False)
            

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)
            

            # forward/backward
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            #loss = criterion(outputs,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            it += 1
            running_loss += loss.item()
            pred = outputs.data.max(1, keepdim=True)[1]

            correct += pred.eq(targets.data.view_as(pred)).sum().item()
            total += targets.size(0)
        
        accuracy = correct/total
        epoch_loss = running_loss / it
        
        # step after one epoch
        lr_scheduler.step()
    print(f"total: {total}")
    return epoch_loss, accuracy

# testing function


def test(
    net: torch.nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    
    running_loss = 0.0
    it = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(testloader, unit="audios", unit_scale=testloader.batch_size, desc='Test')
        for batch in pbar:
            # print(batch.keys())
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            #outputs = torch.nn.functional.softmax(outputs, dim=1)
        
            # statistics
            it += 1
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            #pred = outputs.data.max(1, keepdim=True)[1]
            #correct += pred.eq(targets.data.view_as(pred)).sum()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            #filenames = batch['path']
        
    accuracy = correct/total
    loss = running_loss

    return loss, accuracy

