from torch_geometric.nn import GATConv
from graph_generation import *
import torch.nn
import torch.nn.functional as F
class GraphModel(nn.Module):
    def __init__(self,layer_num, in_features, graphconv = 'SAGE'):
        super(GraphModel, self).__init__()

        self.convs = nn.ModuleList()
        for _ in range(layer_num):

            if graphconv == 'GCN':
                self.convs.append(GCNConv(in_features, in_features))
            elif graphconv == 'SAGE':
                self.convs.append(SAGEConv(in_features, in_features, normalize=True))
            elif graphconv == 'SGC':
                self.convs.append(SGConv(in_features, in_features))
            elif graphconv == 'GAT':
                self.convs.append(GATConv(in_features, in_features),)
            else:
                raise NotImplementedError

        self.Linear1 = nn.Linear(in_features,32)
        self.Linear2 = nn.Linear(32,2)

    def forward(self, x, edge_index, data):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
        w_pred = h
        output = torch.sigmoid(self.Linear2(F.relu(self.Linear1(data))))
        return w_pred, output

class HypersphereModel(nn.Module):
    def __init__(self, center, radius = 0.5, nu: float = 0.05, ):
        super(HypersphereModel, self).__init__()
        self.radius = radius  # hypersphere radius R
        self.center = center  # hypersphere center c
        self.nu = nu
        self.distance = nn.SmoothL1Loss()


    def forward(self, x, y):
        dist = torch.sum((x - self.center) ** 2, dim=1)

        loss = torch.where(y == -1, 100 * ((dist) ** y.float()),  dist)
        return loss
    def get_radius(self, dist: torch.Tensor, nu = 0.05):
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


