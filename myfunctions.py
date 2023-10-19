import os
import torch
from torch import nn
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error
import scipy.sparse as sp

def creat_interval_dataset(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - 2 * lookback):
        x.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])

    return np.array(x), np.array(y)


def seed_torch(seed):
    """
    Set all random seed
    Args:
        seed: random seed

    Returns: None

    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def get_data(args, dev):
    train_x_list = []
    test_x_list = []
    train_y_list = []
    test_y_list = []
    # ------------input data---------

    parking_name_list = os.listdir(args.data_path)
    for i, park in enumerate(parking_name_list):
        dataset = pd.read_csv(args.data_path + park)
        rate = dataset['RATE'].values.astype('float32')
        train_size = int(len(rate) * 0.8)
        train_rate = rate[:train_size]
        test_rate = rate[train_size:]
        train_rate = np.array(train_rate.reshape(-1, 1))
        test_rate = np.array(test_rate.reshape(-1, 1))

        # -------------make data------------
        # -------train data
        train_x, train_y = creat_interval_dataset(train_rate, args.LOOK_BACK, args.predict_time)
        test_x, test_y = creat_interval_dataset(test_rate, args.LOOK_BACK, args.predict_time)

        train_x_list.append(train_x)
        train_y_list.append(train_y)
        test_x_list.append(test_x)
        test_y_list.append(test_y)

    test_x_list = np.array(test_x_list)
    test_y_list = np.array(test_y_list)


    train_x_con = np.concatenate(train_x_list, axis=2)
    test_x_con = np.concatenate(test_x_list, axis=2)
    train_y_con = np.concatenate(train_y_list, axis=1)
    test_y_con = np.concatenate(test_y_list, axis=1)

    # -------train data
    train_x_tensor = torch.from_numpy(train_x_con).to(dev)
    train_y_tensor = torch.from_numpy(train_y_con).to(dev)

    # -------test data
    test_x_tensor = torch.from_numpy(test_x_con).to(dev)
    test_y_tensor = torch.from_numpy(test_y_con).to(dev)
    return train_x_tensor, train_y_tensor, test_x_tensor, test_y_tensor

def get_adj(args, dev):
    df = pd.read_csv('similar.csv', index_col=0, header=0)
    adj_dense = np.array(df, dtype=float)
    adj_dense = torch.Tensor(adj_dense).to(dev)
    edge_index1 = adj_dense.to_sparse_coo().to(dev)

    df2 = pd.read_csv('space.csv',index_col=0, header=0)
    adj_dense = np.array(df2, dtype=float)
    adj_dense = torch.Tensor(adj_dense).to(dev)
    edge_index2 = adj_dense.to_sparse_coo().to(dev)

    return edge_index1, edge_index2

def norm(adj):
    adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    degree = np.diag(np.power(degree, -0.5))
    return degree.dot(adj).dot(degree)




def get_metrics(test_pre, test_real):

    MAPE = mean_absolute_percentage_error(test_real, test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(test_real, test_pre)
    RAE = np.sum(abs(test_pre - test_real)) / np.sum(abs(np.mean(test_real) - test_real))

    print('MAPE: {}'.format(MAPE))
    print('MAE:{}'.format(MAE))
    print('MSE:{}'.format(MSE))
    print('RMSE:{}'.format(RMSE))
    print('R2:{}'.format(R2))
    print(('RAE:{}'.format(RAE)))

    output_list = [MSE, RMSE, MAPE, RAE, MAE, R2]
    return output_list