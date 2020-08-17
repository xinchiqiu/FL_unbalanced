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
''' load and partition the data '''

from typing import List, Tuple, cast
import numpy as np
import os
import pickle
import urllib.request
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from dataset import XY


def download(filename: str, path: str) -> None:
    """Download hotkey dataset."""
    urls = {
        "hotkey_test_x.pkl": "https://www.dropbox.com/s/ve0g1m3wtuecb7r/hotkey_test_x.pkl?dl=1",
        "hotkey_test_y.pkl": "https://www.dropbox.com/s/hlihc8qchpo3hhj/hotkey_test_y.pkl?dl=1",
        "hotkey_train_x.pkl": "https://www.dropbox.com/s/05ym4jg8n7oi5qh/hotkey_train_x.pkl?dl=1",
        "hotkey_train_y.pkl": "https://www.dropbox.com/s/k69lhw5j02gsscq/hotkey_train_y.pkl?dl=1",
    }
    url = urls[filename]
    urllib.request.urlretrieve(url, path)
    print("Downloaded ", url)


def hotkey_load(dirname: str = "./data/hotkey/"):
    """Load Hotkey dataset from disk."""
    files = [
        "hotkey_train_x.pkl",
        "hotkey_train_y.pkl",
        "hotkey_test_x.pkl",
        "hotkey_test_y.pkl",
    ]
    paths = []

    for f in files:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        path = os.path.join(dirname, f)
        if not os.path.exists(path):
            download(f, path)
        paths.append(path)

    with open(paths[0], "rb") as input_file:
        x_train = pickle.load(input_file)

    with open(paths[1], "rb") as input_file:
        y_train = pickle.load(input_file)

    with open(paths[2], "rb") as input_file:
        x_test = pickle.load(input_file)

    with open(paths[3], "rb") as input_file:
        y_test = pickle.load(input_file)
    
    xtrain = x_train[0:31000, :, :]
    ytrain = y_train[0:31000]
    xtest = x_test[0:4000, :, :]
    ytest = y_test[0:4000]

    return xtrain,ytrain,xtest,ytest

x_train,y_train,x_test,y_test = hotkey_load() # output is numpy array

class hotkey_dataset(Dataset):
    def __init__(self,train):
        self.train = bool
        self.xtrain = x_train
        self.ytrain = y_train
        self.xtest = x_test
        self.ytest = y_test
        self.trainlist = list(range(31000))
        self.testlist = list(range(4000))
    
    def __len__(self):
        'Denotes the total number of samples'

        if (self.train):
            return len(self.trainlist)
        else:
            return len(self.testlist)

    def __getitem__(self,index):
        'Generates one sample of data'

        if (self.train):
            x = self.xtrain[index]
            y = self.ytrain[index]
        else:
            x = self.xtest[index]
            y = self.ytest[index]

        x = torch.from_numpy(np.expand_dims(x,axis = 0))
        #y = torch.from_numpy(np.expand_dims(y,axis = 0))
        return x, y



