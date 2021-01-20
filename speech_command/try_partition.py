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
#from partition import create_dla_partitions
#from .partition import *
import numpy as np
from typing import List, Optional,cast

np.random.seed(2020)

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = Tuple[XYList, XYList]




def float_to_int(i: float) -> int:
    """Return float as int but raise if decimal is dropped."""
    if not i.is_integer():
        raise Exception("Cast would drop decimals")

    return int(i)


def sort_by_label(x: np.ndarray, y: np.ndarray) -> XY:
    """Sort by label.
    Assuming two labels and four examples the resulting label order
    would be 1,1,2,2
    """
    idx = np.argsort(y, axis=0).reshape((y.shape[0]))
    return (x[idx], y[idx])
def sort_by_label_repeating(x: np.ndarray, y: np.ndarray) -> XY:
    """Sort by label in repeating groups. Assuming two labels and four examples
    the resulting label order would be 1,2,1,2.
    Create sorting index which is applied to by label sorted x, y
    .. code-block:: python
        # given:
        y = [
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9
        ]
        # use:
        idx = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
        ]
        # so that y[idx] becomes:
        y = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        ]
    """
    x, y = sort_by_label(x, y)

    num_example = x.shape[0]
    num_class = np.unique(y).shape[0]
    idx = (
        np.array(range(num_example), np.int64)
        .reshape((num_class, num_example // num_class))
        .transpose()
        .reshape(num_example)
    )

    return (x[idx], y[idx])
def split_at_fraction(x: np.ndarray, y: np.ndarray, fraction: float) -> Tuple[XY, XY]:
    """Split x, y at a certain fraction."""
    splitting_index = float_to_int(x.shape[0] * fraction)
    # Take everything BEFORE splitting_index
    x_0, y_0 = x[:splitting_index], y[:splitting_index]
    # Take everything AFTER splitting_index
    x_1, y_1 = x[splitting_index:], y[splitting_index:]
    return (x_0, y_0), (x_1, y_1)


def shuffle(x: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def partition(x: np.ndarray, y: np.ndarray, num_partitions: int) -> List[XY]:
    """Return x, y as list of partitions."""
    return list(zip(np.split(x, num_partitions), np.split(y, num_partitions)))


def combine_partitions(xy_list_0: XYList, xy_list_1: XYList) -> XYList:
    """Combine two lists of ndarray Tuples into one list."""
    return [
        (np.concatenate([x_0, x_1], axis=0), np.concatenate([y_0, y_1], axis=0))
        for (x_0, y_0), (x_1, y_1) in zip(xy_list_0, xy_list_1)
    ]
def shift(x: np.ndarray, y: np.ndarray) -> XY:
    """Shift x_1, y_1 so that the first half contains only labels 0 to 4 and
    the second half 5 to 9."""
    x, y = sort_by_label(x, y)

    (x_0, y_0), (x_1, y_1) = split_at_fraction(x, y, fraction=0.5)
    (x_0, y_0), (x_1, y_1) = shuffle(x_0, y_0), shuffle(x_1, y_1)
    x, y = np.concatenate([x_0, x_1], axis=0), np.concatenate([y_0, y_1], axis=0)
    return x, y

def create_partitions(
    unpartitioned_dataset: XY,
    iid_fraction: float,
    num_partitions: int,
) -> XYList:
    """Create partitioned version of a training or test set.
    Currently tested and supported are MNIST, FashionMNIST and
    CIFAR-10/100
    """
    x, y = unpartitioned_dataset

    x, y = shuffle(x, y)
    x, y = sort_by_label_repeating(x, y)

    (x_0, y_0), (x_1, y_1) = split_at_fraction(x, y, fraction=iid_fraction)

    # Shift in second split of dataset the classes into two groups
    x_1, y_1 = shift(x_1, y_1)

    xy_0_partitions = partition(x_0, y_0, num_partitions)
    xy_1_partitions = partition(x_1, y_1, num_partitions)

    xy_partitions = combine_partitions(xy_0_partitions, xy_1_partitions)

    # Adjust x and y shape
    return [adjust_xy_shape(xy) for xy in xy_partitions]


def create_partitioned_dataset(
    keras_dataset: Tuple[XY, XY],
    iid_fraction: float,
    num_partitions: int,
) -> Tuple[PartitionedDataset, XY]:
    """Create partitioned version of keras dataset.
    Currently tested and supported are MNIST, FashionMNIST and
    CIFAR-10/100
    """
    xy_train, xy_test = keras_dataset

    xy_train_partitions = create_partitions(
        unpartitioned_dataset=xy_train,
        iid_fraction=iid_fraction,
        num_partitions=num_partitions,
    )

    xy_test_partitions = create_partitions(
        unpartitioned_dataset=xy_test,
        iid_fraction=iid_fraction,
        num_partitions=num_partitions,
    )

    return (xy_train_partitions, xy_test_partitions), adjust_xy_shape(xy_test)
def log_distribution(xy_partitions: XYList) -> None:
    """Print label distribution for list of paritions."""
    distro = [np.unique(y, return_counts=True) for _, y in xy_partitions]
    for d in distro:
        print(d)


def adjust_xy_shape(xy: XY) -> XY:
    """Adjust shape of both x and y."""
    x, y = xy
    if x.ndim == 3:
        x = adjust_x_shape(x)
    if y.ndim == 2:
        y = adjust_y_shape(y)
    return (x, y)
def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, y, z) into (x, y, z, 1)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
    return cast(np.ndarray, nda_adjusted)


def adjust_y_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, 1) into (x)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0]))
    return cast(np.ndarray, nda_adjusted)
def create_dla_partitions(
    dataset: XY,
    dirichlet_dist: np.ndarray = np.empty(0),
    num_partitions: int = 100,
    concentration: float = 0.5,
) -> Tuple[np.ndarray, XYList]:
    """Create ibalanced non-iid partitions using Dirichlet Latent
    Allocation(LDA) without resampling.
    Args:
        dataset (XY): Datasets containing samples X
            and labels Y.
        dirichlet_dist (numpy.ndarray, optional): previously generated distribution to
            be used. This s useful when applying the same distribution for train and
            validation sets.
        num_partitions (int, optional): Number of partitions to be created.
            Defaults to 100.
        concentration (float, optional): Dirichlet Concentration (:math:`\\alpha`)
            parameter.
            An :math:`\\alpha \\to \\Inf` generates uniform distributions over classes.
            An :math:`\\alpha \\to 0.0` generates on class per client. Defaults to 0.5.
    Returns:
        Tuple[numpy.ndarray, XYList]: List of XYList containing partitions
            for each dataset.
    """

    x, y = dataset
    x, y = shuffle(x, y)
    x, y = sort_by_label(x, y)
    x_l: List[np.ndarray] = list(x)

    # Get number of classes and verify if they matching with
    classes, num_samples_per_class = np.unique(y, return_counts=True)
    num_classes: int = classes.size
    remaining_indices = [j for j in range(num_classes)]

    if dirichlet_dist.size != 0:
        dist_num_partitions, dist_num_classes = dirichlet_dist.shape
        if dist_num_classes != num_classes:
            raise ValueError(
                f"""Number of classes in dataset ({num_classes})
              differs from the one in the provided partitions {dist_num_classes}."""
            )
        if dist_num_partitions != num_partitions:
            raise ValueError(
                f"""The value in `num_partitions` ({num_partitions})
                differs from the one from `dirichlet_dist` {dist_num_partitions}."""
            )

    # Assuming balanced distribution
    num_samples = x.shape[0]
    num_samples_per_partition = num_samples // num_partitions
    
    boundaries: List[int] = np.append(
        [0], np.cumsum(num_samples_per_class, dtype=np.int)
    )
    list_samples_per_class: List[List[np.ndarray]] = [
        x_l[boundaries[idx] : boundaries[idx + 1]] # noqa: E203
        for idx in range(num_classes)  # noqa: E203
    ]

    if dirichlet_dist.size == 0:
        dirichlet_dist = np.random.dirichlet(
            alpha=[concentration] * num_classes, size=num_partitions
        )
    original_dirichlet_dist = dirichlet_dist.copy()

    data: List[List[Optional[np.ndarray]]] = [[] for _ in range(num_partitions)]
    target: List[List[Optional[np.ndarray]]] = [[] for _ in range(num_partitions)]

    for partition_id in range(num_partitions):
        for _ in range(num_samples_per_partition):
            sample_class: int = np.where(
                np.random.multinomial(1, dirichlet_dist[partition_id]) == 1
            )[0][0]
            sample: np.ndarray = list_samples_per_class[sample_class].pop()

            data[partition_id].append(sample)
            target[partition_id].append(sample_class)

            # If last sample of the class was drawn,
            # then set pdf to zero for that class.
            num_samples_per_class[sample_class] -= 1
            if num_samples_per_class[sample_class] == 0:
                remaining_indices.remove(np.where(classes == sample_class)[0][0])
                # Be careful to distinguish between original zero-valued
                # classes and classes that are empty
                dirichlet_dist[:, sample_class] = 0.0
                dirichlet_dist[:, remaining_indices] += 1e-5

                sum_rows = np.sum(dirichlet_dist, axis=1)
                dirichlet_dist = dirichlet_dist / (
                    sum_rows[:, np.newaxis] + np.finfo(float).eps
                )

    partitions = [
        (np.concatenate([data[idx]]), np.concatenate([target[idx]])[..., np.newaxis])
        for idx in range(num_partitions)
    ]

    return partitions, original_dirichlet_dist

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


label = np.load("label.npy")
idx = [i for i in range(len(label))]
idx = np.array(idx)
dataset  = (idx,label)

partitions, _ = create_dla_partitions(dataset,np.empty(0),num_partitions=10, concentration = 0.5)
log_distribution(partitions)

class dataset_afterpartition(Dataset):
    def __init__(
        self,
        client_id,
        num_partitions:str,
        partitions,
        trainset,
        ):

        self.client_id = client_id
        self.num_partitions = num_partitions
        self.idx = partitions[int(self.client_id)][0]
        self.label = partitions[int(self.client_id)][1]
        self.trainset = trainset
        self.classes = np.unique(self.label) 

    def __len__(self):
        return len(range(len(self.label)))



    def __getitem__(self,index):
        'Generates one sample of data'

        #if (self.train):
        x = self.trainset[self.idx[index]]['input']
        y = self.label[index]

        return x, y
    """
    def make_weights_for_balanced_classes(self):
       # adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight
    """

trainset_first = dataset_afterpartition(client_id = int(7),num_partitions=10,partitions=partitions,trainset=trainset)

#weights = trainset_first.make_weights_for_balanced_classes()
#sampler= WeightedRandomSampler(weights, len(weights))
trainloader_first = DataLoader(trainset_first, batch_size=args.batch_size, sampler=None,
                             pin_memory=use_gpu, num_workers=args.dataload_workers_nums)



# set up model
model = models.create_model(model_name=args.model, num_classes=len(CLASSES), in_channels=1)
model.to(device)


def get_lr():
    return optimizer.param_groups[0]['lr']

start_epoch = 0
best_accuracy = 0
best_loss = 1e100
global_step = 0


def train(net, trainloader,epoches,device):
    #global global_step

    start_epoch = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=start_epoch-1)
    criterion = torch.nn.CrossEntropyLoss()

    #print(f'Training {epochs} epoch(s) w/ {len(trainloader)} batches each')

    for epoch in range(epoches):
        lr_scheduler.step()
        #print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
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

train(model,trainloader_first,50,device)
print('finish')
