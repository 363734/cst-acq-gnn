import torch.nn as nn
import torch.nn.functional as F


# Simple 2-layer MLP, maps feature vector of size in_feats to vector of size out_feats. Internal layer of size h_feats.
# Apply to all nodes of type node_cat
class MLPPredictor(nn.Module):
    def __init__(self, node_cat, in_feats, h_feats, out_feats):
        super().__init__()
        # store application type
        self.node_cat = node_cat
        # create layers
        self.W1 = nn.Linear(in_feats, h_feats)
        self.W2 = nn.Linear(h_feats, out_feats)

    def apply_nodes(self, node):
        return {'feats': self.W2(F.relu(self.W1(node.data['feats_raw'])))}

    def forward(self, g):
        with g.local_scope():
            g.apply_nodes(self.apply_nodes, ntype=self.node_cat)
            return g.nodes[self.node_cat].data['feats']
