import torch.utils.data as Data
from torch_geometric.data import Data
from datasets import *
from torch_geometric.nn import GCNConv, SAGEConv, SGConv

def create_data(data, embedding_matrix):
    W = embedding_matrix

    rec_samples = []
    sum = 0
    for element in data:
        for i, feature in enumerate(element):
            sum += feature * W[i]
        rec_samples.append(sum)
        sum = 0

    return rec_samples

def create_node(data, feature_sum, reduced_size):
    W = np.random.rand(feature_sum, reduced_size)

    rec_samples = []
    sum = 0
    for element in data:
        for i, feature in enumerate(element):
            sum += feature.numpy() * W[i]
        rec_samples.append(sum)
        sum = 0


    W = torch.Tensor(W)

    rec_samples = torch.Tensor(np.array(rec_samples))
    return W, rec_samples

def create_edge(x, feature_num):
    init_source, init_end = torch.nonzero(x).transpose(0, 1)

    edge_source = init_source + feature_num
    edge_end = init_end
    edge_source_ = torch.cat([edge_source, edge_end], dim=0)

    edge_end_ = torch.cat([edge_end, edge_source], dim=0)
    edge_index = torch.stack([edge_source_, edge_end_], dim=0) # [2, edge_num]
    return edge_index

def create_label(label):
    y = []
    for i in label:
        if i == 1:
            y.append([1,0])
        else:
            y.append([0,1])
    y = torch.Tensor(y)
    return y

def graph_generation(x, reduced_size, y):
    sample_num, feature_num = x.size()
    W, samples = create_node(x, feature_num, reduced_size)

    node_feature = torch.cat((W, samples), dim = 0)
    edge_index = create_edge(x, feature_num)
    y = create_label(y)
    data = Data(x=node_feature, edge_index=edge_index, y = y)
    return data

def graph_reconstruction(old_embedding_vectors, new_samples, new_label, new_feature_num, reduced_size):


    new_graph_edge = create_edge(new_samples, new_feature_num)

    if new_feature_num != 0:
        new_embedding_vectors = torch.rand(new_feature_num, reduced_size)
        embedding_matrix = torch.cat((old_embedding_vectors.cpu(), new_embedding_vectors), dim = 0)
    else:
        embedding_matrix = old_embedding_vectors.cpu()

    new_rec_samples = []
    sum = 0
    for element in new_samples:
        for i, feature in enumerate(element):
            sum += feature.numpy() * embedding_matrix[i].detach().numpy()
        new_rec_samples.append(sum)
        sum = 0
    new_rec_samples = torch.Tensor(np.array(new_rec_samples))


    node_feature = torch.cat((embedding_matrix, new_rec_samples), dim = 0)
    new_label = create_label(new_label)
    data = Data(x = node_feature, edge_index = new_graph_edge, y = new_label)
    return data

