import dgl
import torch

from src.graph import get_mask

# batch a list of dglgraph together
def batch_graphs(lst_graphs):
    return dgl.batch(lst_graphs)

# given a list of tuples (dglgraph, groundtruth), return one batch graph and one aggregated groundtruth
def batch_graphs_gts(lst):
    graphs = [a[0] for a in lst]
    batched = batch_graphs(graphs)
    gts = torch.tensor([e for a in lst for e in a[1]])
    return batched, gts

# take a list of tuple (graph,gt) where graph is a dglgraph and gt is a list of ground truth values for the cst nodes
def get_graph_batch_gts_mask(lst):
    graph, gts = batch_graphs_gts(lst)
    mask = get_mask(graph)
    print(mask)
    return graph, gts, mask