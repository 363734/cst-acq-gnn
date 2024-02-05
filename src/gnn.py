import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names): #TODO how to deal with not the same number of input features from each type of node
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        # print(inputs)
        h = self.conv1(graph, inputs)
        # print("A")
        # print(h)
        # h = {k: F.relu(v) for k, v in h.items()} #TODO cannot use more than one layer due to directed graph (bidirectionnal ?)
        # print("B")
        # print(h)
        # h = self.conv2(graph, h)
        # print("C")
        # print(h)
        return h
