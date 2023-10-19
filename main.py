import numpy as np
import pandas as pd
import torch
import os

import ablation_model
import myfunctions as fn
from torch.utils.data import DataLoader
import model
from torch import nn
import scipy.sparse as sp
from args import args, dev

fn.seed_torch(2023)
train_x_tensor, train_y_tensor, test_x_tensor, test_y_tensor = fn.get_data(args, dev)
edge_index1, edge_index2 = fn.get_adj(args, dev)
train_model = model.GAT_LSTM(args, edge_index1, edge_index2).to(dev)
model.training(train_model, train_x_tensor,train_y_tensor,args)





