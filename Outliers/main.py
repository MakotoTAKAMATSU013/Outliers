import argparse
import numpy
import torch
from .utils import pre

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150,
                        help="number of epochs")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = get_dataset(args)



