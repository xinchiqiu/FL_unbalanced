# Copyright 2020 Adap GmbH. All Rights Reserved.
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

from collections import OrderedDict
from typing import Tuple
import argparse
import time
from tqdm import *
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision
from torchvision.transforms import *
from tensorboardX import SummaryWriter
import models
from datasets import *
from transforms import *
from mixup import *

"""
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument("--train-dataset", type=str, default='datasets/speech_commands/train', help='path of train dataset')
parser.add_argument("--valid-dataset", type=str, default='datasets/speech_commands/valid', help='path of validation dataset')
parser.add_argument("--background-noise", type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--batch-size", type=int, default=128, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=1e-2, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--learning-rate", type=float, default=1e-4, help='learning rate for optimization')
#parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='step', help='method to adjust learning rate')
parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max-epochs", type=int, default=70, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of NN')
parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
#parser.add_argument('--mixup', action='store_true', help='use mixup')
args = parser.parse_args()
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
n_mels = 32

# get the data
def load_trainset():
    data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    bg_dataset = BackgroundNoiseDataset('datasets/speech_commands/train/_background_noise_', data_aug_transform)
    add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
    train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])

    trainset = SpeechCommandsDataset('datasets/speech_commands/train',
                                    Compose([LoadAudio(),
                                             data_aug_transform,
                                             add_bg_noise,
                                             train_feature_transform]))
    return trainset

#weights = trainset.make_weights_for_balanced_classes()
#sampler = WeightedRandomSampler(weights, len(weights))
#train_dataloader = DataLoader(trainset, batch_size=args.batch_size, sampler=sampler,
#                             pin_memory=use_gpu, num_workers=args.dataload_workers_nums)


def load_testset():
    feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])

    testset = SpeechCommandsDataset(dataset_dir, transform, silence_percentage=0)
    return testset

#test_dataloader = DataLoader(testset, batch_size=args.batch_size, sampler=None,
#                            pin_memory=use_gpu, num_workers=args.dataload_workers_nums)


def load_model():
    model = models.create_model(model_name=models.available_models[0], num_classes=len(CLASSES), in_channels=1)
    return model()

"""
def get_lr():
    return optimizer.param_groups[0]['lr']
"""
def train(
    net: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    #global global_step

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-2)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()

    print(f'Training {epochs} epoch(s) w/ {len(trainloader)} batches each')

    for epoch in range(epochs):
        
        phase = 'train'
        net.train()  # Set model to training mode

        running_loss = 0.0
        it = 0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, unit="audios", unit_scale=trainloader.batch_size)
        for batch in pbar:
            inputs = batch[0]
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch[1]
        
            inputs = Variable(inputs, requires_grad=True)
            targets = Variable(targets, requires_grad=False)
            #inputs.to(device)
            #targets.to(device)
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)
            

            # forward/backward
            outputs = net(inputs)
            loss = criterion(outputs, torch.max(targets, 1)[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            it += 1
            running_loss += loss.item()
            pred = outputs.data.max(1, keepdim=True)[1]

            correct += pred.eq(targets.data.view_as(pred)).sum()
            total += targets.size(0)

            # update the progress bar
            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / it),
                'acc': "%.02f%%" % (100*correct/total)
            })
            
        
        accuracy = correct/total
        epoch_loss = running_loss / it
        
        # step after one epoch
        lr_scheduler.step()

    return epoch_loss, accuracy

# testing function

test_dataset_dir = 'datasets/speech_commands/test'
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
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(testloader, unit="audios", unit_scale=testloader.batch_size)
        for batch in pbar:
            inputs = batch['inputs']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['targets']

            n = inputs.size(0)
            #inputs = Variable(inputs, volatile = True)
            #targets = Variable(targets, requires_grad=False)

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
            correct += (predicted == labels).sum().item()
            total += targets.size(0)
            #filenames = batch['path']
        
            """
            for j in range(len(pred)):
                fn = filenames[j]
                predictions[fn] = pred[j][0]
                probabilities[fn] = outputs.data[j].tolist()
            """
    accuracy = correct/total
    epoch_loss = running_loss / it
    loss = running_loss

    return loss, accuracy

#epoch_loss, accuracy = test(model,testloader,device)
#print('accuracy',accuracy)