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
"""Flower client example using PyTorch for Speech Command dataset."""

import argparse
import timeit

import torch
import torchvision

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

#from . import DEFAULT_SERVER_ADDRESS, speech_command, partition
from speech_command import load_trainset, load_testset, train, test, load_model
from partition import create_dla_partitions, log_distribution,dataset_afterpartition

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
DEFAULT_SERVER_ADDRESS = "[::]:8080"
# pylint: enable=no-member


class SpeechCommandClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        cid: str,
        model,
        trainset_after_partition,
        testset,
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainset = trainset_after_partition
        self.testset = testset

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = self.model.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        num_workers = 6

        # Set model parameters
        self.model.set_weights(weights)

        # Train model
        trainloader = torch.utils.data.DataLoader(trainset_after_partition, batch_size=batch_size, sampler=None,
                              pin_memory=use_gpu, num_workers=num_workers)
        speech_command.train(self.model, trainloader, epochs=epochs, device=DEVICE)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = self.model.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        fit_duration = timeit.default_timer() - fit_begin
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=None, pin_memory=use_gpu, num_workers=num_workers)

        loss, accuracy = speech_command.test(self.model, testloader, device=DEVICE)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            num_examples=len(self.testset), loss=float(loss), accuracy=float(accuracy)
        )


def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        help="number of total clients",
    )
    parser.add_argument(
        "--concentration",
        type=float,
        default = 0.5,.
        help="concentration for LDA alpha",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Load model and data
    model = load_model()
    model.to(DEVICE)
    trainset = load_trainset
    testset = load_testset
    partitions, _ = create_dla_partitions(dataset,np.empty(0),args.num_partitions, args.concentration)
    trainset_after_partition = dataset_afterpartition(client_id = args.cid,num_partitions = args.num_partitions,partitions=partitions,trainset=trainset)
    
    # Start client
    client = SpeechCommandClient(args.cid, model, trainset, testset)
    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()