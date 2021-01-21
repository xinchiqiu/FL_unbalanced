from collections import OrderedDict
from typing import Tuple
import argparse
import time
from tqdm import *
import torch
import torch.nn as nn
import torch.nn.functional as F
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

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train-dataset", type=str, default='datasets/speech_commands/train', help='path of train dataset')
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
parser.add_argument("--max-epochs", type=int, default=20, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of NN')
parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
#parser.add_argument('--mixup', action='store_true', help='use mixup')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
n_mels = 32

# get the data
data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
bg_dataset = BackgroundNoiseDataset(args.background_noise, data_aug_transform)
add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
trainset = SpeechCommandsDataset(args.train_dataset,
                                Compose([LoadAudio(),
                                         data_aug_transform,
                                         add_bg_noise,
                                         train_feature_transform]))
valid_feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
valid_dataset = SpeechCommandsDataset(args.valid_dataset,
                                Compose([LoadAudio(),
                                         FixAudioLength(),
                                         valid_feature_transform]))
weights = trainset.make_weights_for_balanced_classes()
sampler= WeightedRandomSampler(weights, len(weights))
train_dataloader = DataLoader(trainset, batch_size=args.batch_size, sampler=sampler,
                             pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=use_gpu, num_workers=args.dataload_workers_nums)

feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])
dataset_dir = 'datasets/speech_commands/test'
testset = SpeechCommandsDataset(dataset_dir, transform, silence_percentage=0)
testloader = DataLoader(testset, batch_size=args.batch_size, sampler=None,
                            pin_memory=use_gpu, num_workers=args.dataload_workers_nums)


# set up model
model = models.create_model(model_name=args.model, num_classes=len(CLASSES), in_channels=1)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=-1)
criterion = torch.nn.CrossEntropyLoss()

start_epoch = 0
best_accuracy = 0
best_loss = 1e100
global_step = 0


          
def train(epoch):
    global global_step

    phase = 'train'
    model.train()  # Set model to training mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

    
        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)

        # forward/backward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        
        correct += pred.eq(targets.data.view_as(pred)).sum().item()
        total += targets.size(0)


        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    accuracy = correct/total
    epoch_loss = running_loss / it


def valid(epoch):
    global best_accuracy, best_loss, global_step

    phase = 'valid'
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        inputs = Variable(inputs, volatile = True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

         # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })
    return epoch_loss



# testing function

dataset_dir = 'datasets/speech_commands/test'
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
        pbar = tqdm(testloader, unit="audios", unit_scale=testloader.batch_size)
        for batch in pbar:
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']

            n = inputs.size(0)
            #inputs = Variable(inputs, volatile = True)
            #targets = Variable(targets, requires_grad=False)

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)

            # forward

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # statistics
            it += 1
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # pylint: disable=no-member
            #pred = outputs.data.max(1, keepdim=True)[1]
            correct += (predicted == targets).sum().item()
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
    print('test accuracy=', accuracy)

    return accuracy


print("training %s for Google speech commands..." % args.model)
for epoch in range(0, args.max_epochs):
    print('epoch = ', epoch)
    lr_scheduler.step()
    train(epoch)
    epoch_loss = valid(epoch)
    epoch_loss_test, accuracy = test(model,testloader,device)     
    #time_elapsed = time.time() - since
    #time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
    #print("%s, best accuracy: %.02f%%, best loss %f" % (time_str, 100*best_accuracy, best_loss))
print("finished raining")



#epoch_loss, accuracy = test(model,testloader,device)           

