from torch_geometric.data import Data
from graph_generation import create_data
from model import GraphModel, HypersphereModel
import numpy as np
import torch
from torch import nn
from metric import F_score
from torch.nn.functional import normalize
import torch.nn.parameter as Parameter

class HyperSphere_Construction:
    def __init__(self,  mask, center, reduced_size = 5, layer_num =3, Epoch =50):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = mask
        self.reduced_size = reduced_size
        self.layer_num = layer_num
        self.center = torch.Tensor(center).to(self.device)
        self.Hypersphere = HypersphereModel(self.center).to(self.device)
        self.model = GraphModel(self.layer_num, self.reduced_size).to(self.device)
        self.path_state_dict = "./parameter/model_state_dict.pkl"
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.Epoch = Epoch




    def First_Construction(self,data, graph_data_1, label_1, feature_num_1):
        self.data = data
        self.graph_data_1 = graph_data_1.to(self.device)
        self.label_1 = label_1.to(self.device)
        self.feature_num_1 = feature_num_1
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001,weight_decay=0.01)

        self.masklabel = self.label_1.clone().detach()
        self.masklabel[~self.mask] = 0
        rec = self.detectloss(self.data, self.graph_data_1.x[:self.feature_num_1])
        for _ in range(self.Epoch):

            graph_data_x, outputs = self.model(self.graph_data_1.x, self.graph_data_1.edge_index, rec)
            optimizer.zero_grad()
            rec = self.detectloss(self.data, graph_data_x[:self.feature_num_1])
            loss1 = self.SmoothL1Loss(rec, graph_data_x[self.feature_num_1:])
            loss_det = self.Hypersphere(rec,self.masklabel)
            loss_pre = self.BCELoss(outputs[self.mask], self.graph_data_1.y[self.mask])
            loss = torch.mean(loss_det) + loss_pre + loss1
            loss.backward(retain_graph=True)
            optimizer.step()

            self.center = torch.mean(rec[self.masklabel == 1], dim=0)
        net_state_dict = self.model.state_dict()
        torch.save(net_state_dict, self.path_state_dict)
        torch.save(graph_data_x[:self.feature_num_1], './parameter/w_old')

    def Second_Construction(self, data, graph_data_2, label_2,feature_num_2):
        self.data = data
        self.graph_data_2 = graph_data_2.to(self.device)
        self.label_2 = label_2.to(self.device)
        self.feature_num_2 = feature_num_2
        self.feature_increase = self.feature_num_2 - self.feature_num_1

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.masklabel = self.label_2.clone().detach()
        self.masklabel[~self.mask] = 0
        self.w_old = torch.load('./parameter/w_old').to(self.device)
        rec = self.detectloss(self.data, self.graph_data_2.x[:self.feature_num_2])
        for _ in range(self.Epoch):
            optimizer.zero_grad()
            graph_data_x, outputs = self.model(self.graph_data_2.x, self.graph_data_2.edge_index, rec)
            rec = self.detectloss(self.data, graph_data_x[:self.feature_num_2])
            loss1 = self.SmoothL1Loss(rec, graph_data_x[self.feature_num_2:])

            loss_det = self.Hypersphere(rec, self.masklabel)
            loss_rec = self.reconstructionloss(self.data, self.w_old, graph_data_x[:self.feature_num_1])
            loss = torch.mean(loss_det) + loss_rec + loss1
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            optimizer.step()

        torch.save(graph_data_x[:self.feature_num_2], './parameter/w_old')


    def load_model(self, model):
        state_dict_load = torch.load(self.path_state_dict)
        model.load_state_dict(state_dict_load)
        return model

    def get_center(self, graph_data_x, data):
        W = graph_data_x[:self.feature_num]
        rec_samples = []
        sum = 0
        for element in data:
            for i, feature in enumerate(element):
                sum += feature * W[i]
            rec_samples.append(sum)
            sum = 0
        for i, x in enumerate(rec_samples):

            dist = self.SmoothL1Loss(x, self.Hypersphere.center)
            if self.label[i] == 1:
                if self.Hypersphere.radius <= dist:
                    self.Hypersphere.radius = dist

    def reconstructionloss(self, data, EM1, EM2):
        rec1 = []
        rec2 = []
        sum1 = 0
        sum2 = 0
        for element in data:
            if len(np.flatnonzero(element)) != 0:
                for i, index in enumerate(np.flatnonzero(element)):
                    sum1 += element[index].numpy() * EM1[index].cpu().detach().numpy()
                    sum2 += element[index].numpy() * EM2[index].cpu().detach().numpy()
            else:
                sum1 += element[0].numpy() * EM1[0].cpu().detach().numpy()
                sum2 += element[0].numpy() * EM2[0].cpu().detach().numpy()
            rec1.append(sum1)
            rec2.append(sum2)
            sum1 = 0
            sum2 = 0
        rec_1 = torch.Tensor(np.array(rec1)).to(self.device)
        rec_2 = torch.Tensor(np.array(rec2)).to(self.device)
        loss = self.SmoothL1Loss(rec_1, rec_2)
        return loss

    def detectloss(self, data, EM1):
        rec1 = []
        sum1 = 0
        for element in data:
            if len(np.flatnonzero(element)) != 0:
                for i, index in enumerate(np.flatnonzero(element)):
                    sum1 += element[index].numpy() * EM1[index].cpu().detach().numpy()
            else:
                sum1 = element[0].numpy() * EM1[0].cpu().detach().numpy()
            rec1.append(sum1)
            sum1 = 0
        rec_1 = torch.Tensor(np.array(rec1)).to(self.device)
        return rec_1





