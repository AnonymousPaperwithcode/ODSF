import numpy as np
import torch
from matplotlib import pyplot as plt, pylab
from matplotlib.pyplot import xticks
from scipy.interpolate import make_interp_spline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from scipy import signal

def accuracy_calc(pred, label):
    return torch.sum(pred == label) / label.shape[0]

def mae_calc(pred, label):
    return torch.sum( torch.abs(pred - label) ) / pred.shape[0]

def rmse_calc(pred, label):
    return torch.sqrt( torch.sum( (pred - label) ** 2 ) / pred.shape[0] )

def mask_list(data, ratio):
    data_idx = torch.arange(0, data.size(0))
    train_mask = torch.zeros(data.size(0), dtype=torch.bool)
    train_idx = data_idx[:int(data.size(0) * ratio)]
    train_mask[train_idx] = True
    return train_mask


def F_score(y_true, y_pred):


    p = precision_score(y_true, y_pred, average='binary')
    r = recall_score(y_true, y_pred, average='binary')
    f1score = f1_score(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    auc_curve = roc_auc_score(y_true, y_pred)
    return p, r, f1score, accuracy, auc_curve

def accuracy_curve(label, pred, path, method):

    precision = []
    recall = []
    f1 = []
    for i in range(len(label)):
        precision.append(precision_score(label[:i], pred[:i]))
        recall.append(recall_score(label[:i], pred[:i]))
        f1.append(f1_score(label[:i], pred[:i]))
    print(precision[-1])
    print(recall[-1])
    print(f1[-1])
    np.save('./results/' + path + method + 'plot_f1.npy', f1, )

    np.save('./results/' + path + method + 'plot_pre.npy', precision, )

    np.save('./results/' + path + method + 'plot_recall.npy', recall, )



def meanandstd(list):
    mean = []
    std = []
    list = np.array(list)
    for i in range(len(list[0])):
        mean.append(round(np.mean(list[:,i]),3))
        std.append(round(np.std(list[:,i]),3))
    return np.array(mean), np.array(std)








