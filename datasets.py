import os
from numpy import random
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
import scipy.io as scio
from sklearn.preprocessing import StandardScaler, Normalizer




def spambase_normalization(random_seed):
    path = './data/spambase_normalization.csv'
    df = pd.read_csv(path, header=None)
    map_class = {1: -1, 0: 1}
    df[57] = df[57].map(map_class)
    df.fillna(0)
    dataset = df.values
    data = dataset[:, :-1]
    label = dataset[:, -1].astype(int)
    data, label = shuffle(data, label, random_state=random_seed)
    data = torch.Tensor(data)
    label = torch.Tensor(label)
    data_list = []
    label_list = []

    for i in range(10):
        data_list.append(data[int(4200 * (0.1 * (i))):int(4200 * (0.1 * (i + 1))), :int(57 * (0.1 * (i + 1)))])
        label_list.append(label[int(4200 * (0.1 * (i))):int(4200 * (0.1 * (i + 1)))])

    return data_list, label_list



def musk_normalization(random_seed):
    mat = scio.loadmat(os.path.join('data', 'musk.mat'))
    data = mat['X']
    label = mat['y'].reshape(1, 3062)[0].astype(int)
    label[label == 1] = int(-1)
    label[label == 0] = int(1)
    scaler = Normalizer().fit(data)
    data_norm = scaler.transform(data)
    data_norm, label = shuffle(data_norm, label, random_state=random_seed)
    data_norm = torch.Tensor(data_norm)
    label = torch.Tensor(label)
    data_list = []
    label_list = []
    for i in range(10):

        data_list.append(data_norm[int(3060 * (0.1 * i)):int(3060 * (0.1 * (i + 1))), :int(166 * (0.1 * (i + 1)))])
        label_list.append(label[int(3060 * (0.1 * i)):int(3060 * (0.1 * (i + 1)))])


    return data_list, label_list


def internetads_normalization(random_seed):
    path = './data/ad.data'
    df = pd.read_csv(path, header = None, low_memory=False)
    map_class = {'ad.':-1, 'nonad.':1}
    df[1558] = df[1558].map(map_class)
    df.replace('\s+','',regex=True,inplace=True)
    dataset = df.replace('?', 0).astype(float)

    dataset = dataset.values
    data = dataset[:, :-1]
    label = dataset[:, -1].astype(int)
    data, label = shuffle(data, label, random_state=random_seed)
    scaler = Normalizer().fit(data)
    data = scaler.transform(data)
    data = torch.Tensor(data)
    label = torch.Tensor(label)
    data_list = []
    label_list = []
    for i in range(10):
        data_list.append(data[int(1960 * (0.1 * (i))):int(1960 * (0.1 * (i + 1))):, :int(1558 * (0.1 * (i + 1)))])
        label_list.append(label[int(1960 * (0.1 * (i))):int(1960 * (0.1 * (i + 1)))])

    return data_list, label_list


def nslkdd_normalization(random_seed):
    path = './data/nslkdd_normalization.csv'
    df = pd.read_csv(path, header=None)
    map_class = {1: 1, 0: -1}
    df[123] = df[123].map(map_class)
    dataset = df.values
    data = dataset[:, :-2]
    label = dataset[:, -1].astype(int)
    scaler = Normalizer().fit(data)
    data_norm = scaler.transform(data)
    data_norm, label = shuffle(data_norm, label, random_state=random_seed)
    data_norm = torch.Tensor(data_norm)
    label = torch.Tensor(label)
    data_list = []
    label_list = []
    for i in range(10):
        data_list.append(data_norm[int(14000*(0.1*i)):int(14000*(0.1*(i+1))), :int(122*(0.1*(i+1)))])
        label_list.append(label[int(14000*(0.1*i)):int(14000*(0.1*(i+1)))])
    return data_list, label_list

