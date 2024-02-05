import dgl
import torch
import torch.nn.functional as F

from src.gnn import RGCN


# lst_input = list of tuples (dgl graph,ground_truth)
def training(lst_input):
    print('t')

    print('step 1')
    # step 1: create batch and ground truth
    graphs = [a[0] for a in lst_input]
    print(graphs)
    train_graph = dgl.batch(graphs)
    print(train_graph)
    gts = torch.tensor([e for a in lst_input for e in a[1]])
    print(train_graph.etypes)

    print('step 2')
    # step 2: create model
    nb_class = 2
    #model = RGCN(4, 20, nb_class, train_graph.etypes)
    model = RGCN(4, nb_class, nb_class, train_graph.etypes)
    opt = torch.optim.Adam(model.parameters())

    var_feats = train_graph.nodes['var'].data['feats']
    cst_feats = train_graph.nodes['cst'].data['feats']
    node_features = {'var': var_feats, 'cst': cst_feats}

    def evaluate(model, graph, features, labels):
        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            logits = logits['cst']
            labels = labels
            _, indices = torch.max(logits, dim=1)
            print(indices)
            correct = (indices == labels)
            correct_TrueP = torch.sum(correct * labels)
            print("true positives : {}/{}".format(correct_TrueP, sum(labels)))
            correct_TrueN = torch.sum(correct * (1-labels))
            print("true negative : {}/{}".format(correct_TrueN, len(labels)-sum(labels)))
            return (torch.sum(correct) * 1.0/len(labels),correct_TrueP.item() * 1.0 / sum(labels), correct_TrueN.item() * 1.0 / (len(labels)-sum(labels)))

    print('step 3')
    # step 3: train model
    for epoch in range(200):
        print("epoch {}".format(epoch))
        model.train()
        # forward propagation by using all nodes and extracting the user embeddings
        logits = model(train_graph, node_features)
        # print(logits)
        logits = logits['cst']
        # compute loss
        loss = F.cross_entropy(logits, gts)
        # Compute validation accuracy.  Omitted in this example.
        # backward propagation
        acc = evaluate(model, train_graph, node_features, gts)
        print("accuracy : {}".format(acc))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("loss : {}".format(loss.item()))





if __name__ == "__main__":

    print('test')