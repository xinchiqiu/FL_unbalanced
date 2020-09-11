import argparse
import timeit
from collections import OrderedDict
import torch
import torchvision
import torchvision.models as models
import flwr as fl
import numpy as np
import fashionmnist
from dataset_partition import dataset_afterpartition,get_partition
from torch.utils.data import Dataset, DataLoader, TensorDataset
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

#from . import DEFAULT_SERVER_ADDRESS
DEFAULT_SERVER_ADDRESS = "[::]:8000"
# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member
#DEVICE = "cpu" 
#torch.set_num_threads(2)

class FashionmnistClient(fl.client.Client):

    def __init__(
        self,
        cid: str,
        model: fashionmnist.Net,
        #model:torch.nn.Module,
        #trainset: torchvision.datasets,
        #testset: torchvision.datasets,
        xy_train_partitions,
        xy_test_partitions,
        nb_clients: int
    ) -> None:
        super().__init__(cid)
        self.model = model
        #self.model = models.resnet18().to(DEVICE)
        #self.trainset = trainset
        #self.testset = testset
        self.xy_train_partitions = xy_train_partitions
        self.xy_test_partitions = xy_test_partitions
        self.nb_clients = nb_clients

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = self.model.get_weights()
        #weights: Weights = get_weights(self.model)
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
        rnd = int(config["rnd"])
        lr = 0.001

        # Set model parameters
        self.model.set_weights(weights)

        # load IID partitioned dataset        
        trainset = dataset_afterpartition(train=True, client_id = int(self.cid),num_partitions = self.nb_clients,
            xy_train_partitions = self.xy_train_partitions, xy_test_partitions= self.xy_test_partitions)
        
        # Train model
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
        fashionmnist.train(self.model, trainloader, epochs=epochs,lr = lr,  device=DEVICE)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = self.model.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(trainset)
        fit_duration = timeit.default_timer() - fit_begin
        return params_prime, num_examples_train, num_examples_train, fit_duration
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins[0])

        # Use provided weights to update the local model
        self.model.set_weights(weights)
        #set_weights(self.model, weights)

        # get IID test dataset
        testset = dataset_afterpartition(train=False,client_id = int(self.cid),num_partitions = self.nb_clients,
            xy_train_partitions= self.xy_train_partitions, xy_test_partitions = self.xy_test_partitions)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=32, shuffle=False
        )
        loss, accuracy = fashionmnist.test(self.model, testloader, device=DEVICE)

        # Return the number of evaluation examples and the evaluation result (loss)
        return len(testset), float(loss), float(accuracy)


def main() -> None:
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
    parser.add_argument(
        "--iid_fraction", type=float, default=1.0, help="Total number of clients",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Load model and data
    model = fashionmnist.load_model()
    model.to(DEVICE)
    (xy_train_partitions, xy_test_partitions), xy_test = get_partition(iid_fraction = args.iid_fraction, num_partitions = args.nb_clients)
    #trainset, testset = fashionmnist.load_data()

    # Start client
    client = FashionmnistClient(args.cid,model,xy_train_partitions, xy_test_partitions,args.nb_clients)
    fl.client.start_client(args.server_address, client)

if __name__ == "__main__":
    main()
