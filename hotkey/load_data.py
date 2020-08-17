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
from torch.utils.data import Dataset, DataLoader
import torch
from dataset import (
    XY,
    PartitionedDataset,
    create_partitioned_dataset,
    log_distribution,
    sort_by_label,
    sort_by_label_repeating,
    shift,
)


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


class hotkey_dataset(torch.utils.data.Dataset):
    def __init__(self,train):
        self.train = bool
        self.trainlist = list(range(31000))
        self.testlist = list(range(4000))
        self.files = [
        "hotkey_train_x.pkl",
        "hotkey_train_y.pkl",
        "hotkey_test_x.pkl",
        "hotkey_test_y.pkl",
        ]
        self.data_dir = "./data/hotkey/"
        self.paths = []
        #self.transform = transform
        for f in self.files:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            path = os.path.join(self.data_dir, f)
            if not os.path.exists(path):
                download(f, path)
            self.paths.append(path)

    def __len__(self):
        'Denotes the total number of samples'

        if (self.train):
            return len(self.trainlist)
        else:
            return len(self.testlist)
    
    def __getitem__(self,index):
        'Generates one sample of data'

        if (self.train):

            # Select sample
            trainID = self.trainlist[index]

            with open(self.paths[0], "rb") as input_file:
                x_train = pickle.load(input_file)[self.trainlist][index]

            with open(self.paths[1], "rb") as input_file:
                y_train = pickle.load(input_file)[self.trainlist][index]

            x = x_train
            y = y_train
        else:
            testID = self.testlist[index]

            # Load data and get label

            with open(paths[2], "rb") as input_file:
                x_test = pickle.load(input_file)[self.testlist][index]
            np.expand_dims(x_test,axis = 0)

            with open(paths[3], "rb") as input_file:
                y_test = pickle.load(input_file)[self.testlist][index]   

            x = x_test
            y = y_test

        x = torch.from_numpy(np.expand_dims(x,axis = 0))
        #y = torch.from_numpy(np.expand_dims(y,axis = 0))
        return x, y


