import argparse
from typing import Callable, Dict, Optional, Tuple
import torch
import torchvision
import torchvision.models as models
import flwr as fl
import fashionmnist

DEFAULT_SERVER_ADDRESS = "[::]:8000"
# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


def main() -> None:
    """Start server and train five rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=2,
        help="Minimum number of clients used for fit/evaluate (default: 2)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--log_host", type=str, help="Logserver address (no default)",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    # Load evaluation data
    _, testset = fashionmnist.load_data()

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.DefaultStrategy(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    fl.server.start_server(
        args.server_address, server, config={"num_rounds": args.rounds},
    )


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "rnd": str(rnd),
        "epochs": str(1),
        "batch_size": str(10),
    }
    return config


def get_eval_fn(
    testset: torchvision.datasets.FashionMNIST,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire FashionMNIST test set for evaluation."""
        model = fashionmnist.load_model()
        model.set_weights(weights)
        model.to(DEVICE)
        model.eval()

        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        return fashionmnist.test(model, testloader, device=DEVICE)

    return evaluate

if __name__ == "__main__":
    main()
