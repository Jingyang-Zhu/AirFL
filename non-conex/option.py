import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type = str,
        default = 'mnist',
        help = 'name of the dataset: mnist'
    )
    parser.add_argument(
        "--dataset_path",
        type = str,
        default = 'data/mnist',
        help = 'dataset  folder'
    )
    parser.add_argument(
        "--input_channels",
        type = int,
        default = 1,
        help = 'input channels'
    )
    parser.add_argument(
        "--output_channels",
        type = int,
        default = 10,
        help = "output channels"
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 50,
        help = 'batch size when trained on devices'
    )
    parser.add_argument(
        "--lr",
        type = float,
        default = 0.05,
        help = 'learning rate of the SGD when trained on devices'

    )
    parser.add_argument(
        "--num_users",
        type = int,
        default = 50,
        help = "number of all available user devices"
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 1,
        help = "random seed (default : 1)"
    )
    parser.add_argument(
        "--iid",
        type = int,
        default = 0,
        help = 'iid'
    )
    parser.add_argument(
        "--gpu",
        type = int,
        default = 0,
        help = 'GPU to be selected,0,1,2 3'
    )
    parser.add_argument(
        "--num_local_update",
        type = int,
        default = 5,
        help = "number of local update"
    )
    parser.add_argument(
        "--num_communication",
        type = int,
        default = 1500,
        help = "number of communication rounds with the cloud server"
    )
    parser.add_argument(
        "--noise_var",
        type = float,
        default = 0,
        help = "the variance of AWGN channel noise"
    )
    parser.add_argument(
        "--P",
        type = float,
        default = 1.0,
        help = "power"
    )
    parser.add_argument(
        "--case",
        type = int,
        default = 0,
        help = "0: gradient; 1: one-step model; \
                2: model differentiation; 3: multi-step model"
    )
    parser.add_argument(
        "--algorithm",
        type = str,
        default = "FedAvg",
        help = "Federated Average"
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args