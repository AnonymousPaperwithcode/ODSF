import matplotlib.pyplot as plt
import torch
from pyod.utils import evaluate_print
from torch_geometric.data import Data
from datasets import *
from graph_generation import graph_generation, graph_reconstruction, create_label
from hypersphere import HyperSphere_Construction
from model import GraphModel, HypersphereModel
from model_initial import model_initial
from metric import F_score, mask_list, meanandstd, accuracy_curve
from sklearn.manifold import TSNE
import time





def test(w_pred, R, C, data, label, path, tnse =0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, feature_num = data.size()
    W = w_pred[:feature_num]

    rec_samples = []
    sum = 0
    for element in data:
        if len(np.flatnonzero(element)) != 0:
            for i, index in enumerate(np.flatnonzero(element)):
                sum += element[index].numpy() * W[index].cpu().detach().numpy()
        else:
            sum = element[0].numpy() * W[0].cpu().detach().numpy()
        rec_samples.append(sum)
        sum = 0

    rec_samples = np.array(rec_samples)

    C = np.mean(rec_samples[:int(0.2*(len(label)))][label[:int(0.2*(len(label)))] == 0], axis=0)
    prediction = []
    scores = []

    for i, x in enumerate(rec_samples):
        x = torch.Tensor(x).to(device)
        dist = torch.sum((x - C) ** 2)
        scores.append(dist.cpu().detach().numpy())
    treshold = []
    for i in range(int(0.1*len(label))):

        if label[i] == 0:
            treshold.append(scores[i])
    copy = treshold.copy()

    treshold = np.mean(copy)


    for i in range(len(label)):
        if scores[i] >= treshold:
            prediction.append(1)
        else:
            prediction.append(0)
    print("\nOn Test Data:")


    p, r, f1score,accuracy, auc = F_score(label.detach().numpy().tolist(), prediction)

    print("Precision = {:.2f}".format(p))
    print("Recall = {:.2f}".format(r))
    print("F1-Score = {:.2f}".format(f1score))


    return f1score, p, r, prediction

def train(data_list, label_list,model_layers, training_epoch, reduced_size, label_rate, path):
    print('Model Initialing')
    mask_data = mask_list(data_list[0], label_rate)
    _, feature_num = data_list[0].size()
    graph_data = graph_generation(data_list[0], reduced_size, label_list[0])
    model_center = GraphModel(model_layers, reduced_size)
    center = model_initial(graph_data, feature_num, model_center, mask_data, data_list[0], label_list[0])
    print('Model Initialed')
    print('--------------------------------------------')

    f1_list= []
    pre_list = []
    recall_list = []
    groundtruth = []
    prediction = []
    Starter = HyperSphere_Construction(mask_data, center, reduced_size=reduced_size,layer_num=model_layers, Epoch=training_epoch)

    for times, data in enumerate(data_list):

        if times <= 8:
            _, feature_num_1 = data_list[times].size()
            _, feature_num_2 = data_list[times + 1].size()
            print('The {}-th Trapezoidal Data Stream Training'.format(times + 1))
            old_embedding_vectors = torch.load('./parameter/w_old')
            graph_data_1 = graph_reconstruction(old_embedding_vectors, data, label_list[times], 0, reduced_size)
            label_test = label_list[times]
            Starter.First_Construction(data, graph_data_1, label_list[times], feature_num_1)
            #---------------------------------------------------------------------------------------------------

            feature_gap = feature_num_2 - feature_num_1
            old_embedding_vectors = torch.load('./parameter/w_old')
            graph_data_2 = graph_reconstruction(old_embedding_vectors, data_list[times + 1], label_list[times+1], feature_gap, reduced_size)
            Starter.Second_Construction(data, graph_data_2, label_list[times+1], feature_num_2)

            label_test[label_test == 1] = 0
            label_test[label_test == -1] = 1
            print(Starter.center)
            f1score, p, r, pred = test(Starter.graph_data_1.x, 0.2, Starter.center, data,  label_test,path = path, tnse=times)
            prediction+=pred

            groundtruth+=label_test.tolist()

            f1_list.append(f1score)
            pre_list.append(p)
            recall_list.append(r)

        else:
            _, feature_num_1 = data_list[times].size()
            print('The {}-th Trapezoidal Data Stream Training'.format(times + 1))
            old_embedding_vectors = torch.load('./parameter/w_old')
            graph_data_1 = graph_reconstruction(old_embedding_vectors, data, label_list[times], 0, reduced_size)
            label_test = label_list[times]
            Starter.First_Construction(data, graph_data_1, label_list[times], feature_num_1)

            label_test[label_test == 1] = 0
            label_test[label_test == -1] = 1

            f1score, p, r, pred = test(Starter.graph_data_1.x, 0.1, Starter.center, data, label_test, path=path,
                       tnse=times)
            prediction += pred
            groundtruth += label_test.tolist()
            f1_list.append(f1score)
            pre_list.append(p)
            recall_list.append(r)


    return f1_list,pre_list,recall_list, groundtruth, prediction

if __name__ == '__main__':

    model_layers = 8
    training_epoch = 5
    reduced_size = 5
    label_rate = 0.2

    random_seed =1001

    start = time.time()
    f1_list = []
    precision = []
    recall = []

    for path in ['musk']:
        for i in range(1):
            if path == 'internetads':
                data_list, label_list = internetads_normalization(random_seed)
            elif path == 'spambase':
                data_list, label_list = spambase_normalization(random_seed)
            elif path == 'nslkdd':
                data_list, label_list = nslkdd_normalization(random_seed)
            elif path == 'musk':
                data_list, label_list = musk_normalization(random_seed)
            elif path == 'reuteren':
                data_list, label_list = reuteren(random_seed)


            f1, p, r, groundtruth, prediction = train(data_list, label_list, model_layers, training_epoch, reduced_size, label_rate, path = path)
            f1_list.append(f1)
            precision.append(p)
            recall.append(r)

        accuracy_curve(groundtruth, prediction, path, 'ours')
        mean_f1, std_f1 = meanandstd(f1_list)
        mean_pre, std_pre = meanandstd(precision)
        mean_recall, std_recall = meanandstd(recall)
        #average_mean = np.mean(mean_f1)
        print('The mean of F1-Score is {} and the standard deviation of F1-Score is {}'.format(mean_f1, std_f1))
        print('The mean of Precision is {} and the standard deviation of Precision is {}'.format(mean_pre, std_pre))
        print('The mean of Recall is {} and the standard deviation of Recall is {}'.format(mean_recall, std_recall))






