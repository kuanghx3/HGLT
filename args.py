import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())

from myfunctions import seed_torch

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument('--data_path', type=str,
                    default="./RATE35/",
                    help='The directory containing the parking lot data.')
parser.add_argument('--LOOK_BACK', type=int, default=12,
                    help='Number of time step of the Look Back Mechanism.')
parser.add_argument('--predict_time', type=int, default=12,
                    help='Number of time step of the predict time.')
parser.add_argument('--nodes', type=int, default=35,
                    help='Number of parking lots.')
parser.add_argument('--max_epochs', type=int, default=2000,
                    help='The max training epochs.')
parser.add_argument('--training_rate', type=float, default=0.6,
                    help='The rate of meta training.')
parser.add_argument('--valid_rate', type=float, default=0.2,
                    help='The rate of fine tuning training.')
parser.add_argument('--layer', type=float, default=5,
                    help='layer.')
parser.add_argument('--k', type=float, default=5,
                    help='k.')
parser.add_argument('--MLP_hidden', type=float, default=12,
                    help='MLP_hidden.')




args = parser.parse_args(args=[]) # jupyter
# args = parser.parse_args()      # python

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
seed_torch(2023)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# dev = 'cpu'