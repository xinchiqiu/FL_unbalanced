import argparse
import timeit
from collections import OrderedDict
import torch
import torchvision
import torchvision.models as models
import flwr as fl
import numpy as np
import cifar
from dataset_partition import dataset_afterpartition, get_partition
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

#from . import DEFAULT_SERVER_ADDRESS
DEFAULT_SERVER_ADDRESS = "[::]:8080"
# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member
#DEVICE = "cpu" 
#torch.set_num_threads(2)

def get_weights(model) -> Weights:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights: Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {k: torch.Tensor(np.atleast_1d(v)) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict, strict=True)


class CifarClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        cid: str,
        #model: cifar.Net,
        #model:torch.nn.Module,
        #trainset: torchvision.datasets,
        #testset: torchvision.datasets,
        xy_train_partitions,
        xy_test_partitions,
        nb_clients: int
    ) -> None:
        super().__init__(cid)
        #self.model = model
        self.model = models.resnet18().to(DEVICE)
        self.xy_train_partitions = xy_train_partitions
        self.xy_test_partitions = xy_test_partitions
        self.nb_clients = nb_clients
    
    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        #weights: Weights = self.model.get_weights()
        weights: Weights = get_weights(self.model)
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)
    
    def fit(self, ins: FitIns) -> FitRes:
        # Set the seed so we are sure to generate the same global batches
        # indices across all clients
        np.random.seed(123)

        print(f"Client {self.cid}: fit")

        weights: Weights = fl.common.parameters_to_weights(ins[0])
        config = ins[1]
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        #self.model.set_weights(weights)
        set_weights(self.model, weights)
        
        # get IID dataset
        trainset = dataset_afterpartition(train=True, client_id = int(self.cid),num_partitions = self.nb_clients,
            xy_train_partitions = self.xy_train_partitions, xy_test_partitions= self.xy_test_partitions)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False 
        )

        cifar.train(self.model, trainloader, epochs=epochs, device=DEVICE)

        # Return the refined weights and the number of examples used for training
        #weights_prime: Weights = self.model.get_weights()
        weights_prime: Weights = get_weights(self.model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(trainset)
        fit_duration = timeit.default_timer() - fit_begin
        return params_prime, num_examples_train, num_examples_train, fit_duration
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Set the set so we are sure to generate the same batches
        # accross all clients.
        np.random.seed(123)

        print(f"Client {self.cid}: evaluate")
        config = ins[1]
        #batch_size = int(config["batch_size"])

        weights = fl.common.parameters_to_weights(ins[0])

        # Use provided weights to update the local model
        #self.model.set_weights(weights)
        set_weights(self.model, weights)

        # get IID test dataset
        testset = dataset_afterpartition(train=False,client_id = int(self.cid),num_partitions = self.nb_clients,
            xy_train_partitions= self.xy_train_partitions, xy_test_partitions = self.xy_test_partitions)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=32, shuffle=False
        )
        loss, accuracy = cifar.test(self.model, testloader, device=DEVICE)

        # Return the number of evaluation examples and the evaluation result (loss)
        return len(testset), float(loss), float(accuracy)
    
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
        "--log_host", type=str, help="Logserver address (no default)",
    )
    parser.add_argument(
        "--nb_clients", type=int, default=10, help="Total number of clients",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Load model and data
    #model = cifar.load_model()
    #model.to(DEVICE)
    #trainset, testset = cifar.load_data()
    (xy_train_partitions, xy_test_partitions), xy_test = get_partition(iid_fraction = 1.0, num_partitions = args.nb_clients)


    # Start client
    #client = CifarClient(args.cid, model, trainset, testset, args.nb_clients)
    client = CifarClient(args.cid, xy_train_partitions, xy_test_partitions, args.nb_clients)
    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()