import itertools

import torch
import torch.nn.functional as F
import sklearn
import numpy

from src.batch import get_graph_batch_gts_mask
from src.gnn_conv import gnn_conv
from src.graph import nb_feat_cst, nb_feat_var
from src.metrics import compute_metrics, print_metrics_all
from src.mlp import MLPPredictor


# lst_input = list of tuples (dgl graph,ground_truth)
def training(opts, lst_train, lst_test):
    print('step 1')
    # step 1: create batch and ground truth + masks
    train_graph, train_gts, train_mask = get_graph_batch_gts_mask(lst_train)
    print(train_graph)
    print(len(train_gts))
    print(len(train_mask))
    print(train_mask)
    test_graph, test_gts, test_mask = get_graph_batch_gts_mask(lst_test)
    print(test_graph)
    print(len(test_gts))
    print(test_gts)
    print(len(test_mask))
    print(test_mask)
    masked_gts = {'train': train_gts[train_mask], 'test': test_gts[test_mask]}

    print('step 2')
    # step 2: create model
    nb_class = 2  # yes or no
    assert(nb_feat_var(train_graph) < opts.gnn_infeat, "opts.gnn_infeat is not big enough for var (create reduction of feature size)")
    assert(nb_feat_cst(train_graph) < opts.gnn_infeat, "opts.gnn_infeat is not big enough for cst (create reduction of feature size)")
    mlp_var = MLPPredictor("var", nb_feat_var(train_graph), opts.gnn_infeat, opts.gnn_infeat)
    mlp_cst = MLPPredictor("cst", nb_feat_cst(train_graph), opts.gnn_infeat, opts.gnn_infeat)
    model = gnn_conv(opts.gnn_nblayers, opts.gnn_infeat, opts.gnn_hidfeat, nb_class, train_graph.etypes)
    opt = torch.optim.Adam(itertools.chain(model.parameters(), mlp_cst.parameters(), mlp_var.parameters()),
                           lr=opts.gnn_learning_rate)

    print('step 3')
    # step 3: compute weights to deal with unbalanced dataset
    weights = sklearn.utils.class_weight.compute_class_weight('balanced', classes=numpy.unique(masked_gts['train']),
                                                              y=masked_gts['train'].numpy())
    weights = torch.tensor(weights, dtype=torch.float)

    stats = {}

    def prediction_no_grad(graph):
        model.eval()
        mlp_cst.eval()
        mlp_var.eval()
        with torch.no_grad():
            # do the prediction
            features = {'var': mlp_var(graph), 'cst': mlp_cst(graph)}
            logits = model(graph, features)
            logits = logits['cst']
            return logits

    print('step 4')
    stats["nb-epoch"] = opts.gnn_nbepochs
    # step 4: train model
    for epoch in range(stats["nb-epoch"]):
        stats[epoch] = {}
        print("epoch {}".format(epoch))
        model.train()
        mlp_cst.train()
        mlp_var.train()
        node_features = {'var': mlp_var(train_graph), 'cst': mlp_cst(train_graph)}
        # forward propagation by using all nodes and extracting the user embeddings
        logits = model(train_graph, node_features)
        # print(logits)
        logits = logits['cst']
        # compute loss
        loss = F.cross_entropy(logits[train_mask], train_gts[train_mask], weights)

        # metrics
        print("loss : {}".format(loss.item()))

        train_logits = prediction_no_grad(train_graph)
        print(train_logits)
        train_metrics = compute_metrics(train_logits[train_mask], masked_gts['train'])
        print("Train-set metrics:")
        print_metrics_all(train_metrics)

        test_logits = prediction_no_grad(test_graph)
        test_metrics = compute_metrics(test_logits[test_mask], masked_gts['test'])
        print("Test-set metrics:")
        print_metrics_all(test_metrics)
        stats[epoch] = {'train-loss': loss.item(), 'train': train_metrics, 'test': test_metrics}

        if epoch < stats["nb-epoch"] - 1:
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()

    return stats


if __name__ == "__main__":
    print('test')
