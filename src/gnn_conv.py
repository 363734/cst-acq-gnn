import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


def gnn_conv(nb_layer, in_feats, hid_feats, out_feats, rel_names):
    if nb_layer == 1:
        return GNNconv_1layer(in_feats, out_feats, rel_names)
    elif nb_layer == 2:
        return GNNconv_2layers(in_feats, hid_feats, out_feats, rel_names)
    elif nb_layer == 3:
        return GNNconv_3layers(in_feats, hid_feats, out_feats, rel_names)
    elif nb_layer == 4:
        return GNNconv_4layers(in_feats, hid_feats, out_feats, rel_names)


class GNNconv_4layers(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        print("-:: Creation of GNN with:")
        print("    -: nb layers = {}".format(4))
        print("    -: size of input feature vect = {}".format(in_feats))
        print("    -: size of hidden feature vect = {}".format(hid_feats))
        print("    -: size of output feature vect (nb class) = {}".format(out_feats))

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, hid_feats, "mean")
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, hid_feats, "mean")
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, hid_feats, "mean")
            for rel in rel_names}, aggregate='sum')
        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, out_feats, "mean")
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        # print(inputs)
        h = self.conv1(graph, inputs)
        # print("A")
        # print(h)
        h = {k: F.relu(v) for k, v in h.items()}
        # print("B")
        # print(h)
        h = self.conv2(graph, h)
        # print("C")
        # print(h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv4(graph, h)
        return h


class GNNconv_3layers(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        print("-:: Creation of GNN with:")
        print("    -: nb layers = {}".format(3))
        print("    -: size of input feature vect = {}".format(in_feats))
        print("    -: size of hidden feature vect = {}".format(hid_feats))
        print("    -: size of output feature vect (nb class) = {}".format(out_feats))


        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, hid_feats, "mean")
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, hid_feats, "mean")
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, out_feats, "mean")
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        # print(inputs)
        h = self.conv1(graph, inputs)
        # print("A")
        # print(h)
        h = {k: F.relu(v) for k, v in h.items()}
        # print("B")
        # print(h)
        h = self.conv2(graph, h)
        # print("C")
        # print(h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h


class GNNconv_2layers(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        print("-:: Creation of GNN with:")
        print("    -: nb layers = {}".format(2))
        print("    -: size of input feature vect = {}".format(in_feats))
        print("    -: size of hidden feature vect = {}".format(hid_feats))
        print("    -: size of output feature vect (nb class) = {}".format(out_feats))


        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, hid_feats, "mean")
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, out_feats, "mean")
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        # print(inputs)
        h = self.conv1(graph, inputs)
        # print("A")
        # print(h)
        h = {k: F.relu(v) for k, v in h.items()}
        # print("B")
        # print(h)
        h = self.conv2(graph, h)
        # print("C")
        # print(h)
        return h


class GNNconv_1layer(nn.Module):
    def __init__(self, in_feats, out_feats, rel_names):
        super().__init__()

        print("-:: Creation of GNN with:")
        print("    -: nb layers = {}".format(1))
        print("    -: size of input feature vect = {}".format(in_feats))
        print("    -: size of output feature vect (nb class) = {}".format(out_feats))


        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, out_feats, "mean")
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        # print(inputs)
        h = self.conv1(graph, inputs)
        # print("A")
        # print(h)
        return h
