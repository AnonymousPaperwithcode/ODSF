from graph_generation import *
from model import GraphModel


def detectloss( data, EM1):
    rec1 = []
    sum1 = 0
    for element in data:
        for i, feature in enumerate(element):
            sum1 += feature.numpy() * EM1[i].cpu().detach().numpy()
        rec1.append(sum1)
        sum1 = 0
    rec_1 = torch.Tensor(np.array(rec1))
    return rec_1



def model_initial(graph_data, feature_num, model, mask, data, label, epoch = 10):
    optim = torch.optim.Adam(model.parameters(), lr = 0.001)
    lossfunction = nn.BCELoss()
    lossfunction2 = nn.SmoothL1Loss()

    rec = detectloss(data, graph_data.x[:feature_num])
    for i in range(epoch):
        loss1 = lossfunction2(rec, graph_data.x[feature_num:])
        w_pred, outputs = model(graph_data.x, graph_data.edge_index, rec)
        rec = detectloss(data, w_pred[:feature_num])
        optim.zero_grad()
        loss = lossfunction(outputs[mask], graph_data.y[mask])
        loss = loss1 + loss
        loss.backward(retain_graph=True)
        optim.step()


    W =  w_pred[:feature_num]
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


    rec_samples = torch.Tensor(np.array(rec_samples))[:200][label[:200] == 1]
    center = torch.mean(rec_samples, axis = 0)
    torch.save(w_pred[:feature_num], './parameter/w_old' )
    path_state_dict = "./parameter/model_state_dict.pkl"
    net_state_dict = model.state_dict()
    torch.save(net_state_dict, path_state_dict)
    return center




