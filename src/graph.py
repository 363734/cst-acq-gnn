import copy

import dgl
import torch


def create_graph_unknown(bias):
    nb_vars = 0
    map_vars = {}
    edges = []
    b_arity = []
    x_lb = []
    x_ub = []
    x_int = []
    x_bool = []
    for i in range(len(bias)):
        b = bias[i]
        xs = b.args
        for x in xs:
            if x not in map_vars:
                map_vars[x] = nb_vars
                nb_vars += 1
                x_lb.append(x.ub)
                x_ub.append(x.ub)
                if x.is_bool():
                    x_bool.append(1)
                    x_int.append(0)
                else:
                    x_bool.append(0)
                    x_int.append(1)
            edges.append((map_vars[x], i))
        b_arity.append(len(xs))

    graph_data = {  # TODO test bidirectionnal ?
        ('var', 'in scope of', 'cst'): ([e[0] for e in edges], [e[1] for e in edges])
    }
    graph = dgl.heterograph(graph_data)
    # TODO find the right features
    graph.nodes['cst'].data['arity'] = torch.tensor(b_arity)
    graph.nodes['cst'].data['unknown'] = torch.tensor([1] * len(bias))
    graph.nodes['cst'].data['yes'] = torch.tensor([0] * len(bias))
    graph.nodes['cst'].data['no'] = torch.tensor([0] * len(bias))

    graph.nodes['cst'].data['feats'] = torch.tensor([[b_arity[i], 1, 0, 0] for i in range(len(bias))])

    graph.nodes['var'].data['isint'] = torch.tensor(x_int)
    graph.nodes['var'].data['isbool'] = torch.tensor(x_bool)
    graph.nodes['var'].data['lb'] = torch.tensor(x_lb)
    graph.nodes['var'].data['ub'] = torch.tensor(x_ub)

    graph.nodes['var'].data['feats'] = torch.tensor([[x_int[i],x_bool[i],x_lb[i],x_ub[i]] for i in range(nb_vars)])

    return graph


def switch_senario(graph, indicators):
    unknown, yes, no = indicators
    uptaded_g = copy.deepcopy(graph)  # TODO: for training, maybe pass by pickle to load quickly new copy
    uptaded_g.nodes['cst'].data['unknown'] = torch.tensor(unknown)
    uptaded_g.nodes['cst'].data['yes'] = torch.tensor(yes)
    uptaded_g.nodes['cst'].data['no'] = torch.tensor(no)
    for i in range(len(uptaded_g.nodes['cst'].data['feats'])):
        uptaded_g.nodes['cst'].data['feats'][i][1] = unknown[i]
        uptaded_g.nodes['cst'].data['feats'][i][2] = yes[i]
        uptaded_g.nodes['cst'].data['feats'][i][3] = no[i]
    return uptaded_g


if __name__ == "__main__":
    # TODO test
    # - node feature vs removing node that are not unknown or yes

    graph_data = {
        ('var', 'in scope of', 'cst'): ([0, 0, 1], [0, 1, 1])
    }
    graph = dgl.heterograph(graph_data)
    graph.nodes['cst'].data['arity'] = torch.tensor([1, 2])  # put tensor
    # features for the current state of the bias
    graph.nodes['cst'].data['unknown'] = torch.tensor([0, 1])
    graph.nodes['cst'].data['yes'] = torch.tensor([0, 0])
    graph.nodes['cst'].data['no'] = torch.tensor([1, 0])
    graph.nodes['cst'].data['sum'] = torch.tensor([1, 0])
    graph.nodes['cst'].data['logical equality'] = torch.tensor([1, 0])

    graph.edata["lhs"] = torch.tensor([0, 1, 0])
    graph.edata["rhs"] = torch.tensor([0, 0, 1])

    graph.nodes['var'].data['integer'] = torch.tensor([1, 1])
    graph.nodes['var'].data['boolean'] = torch.tensor([0, 0])
    graph.nodes['var'].data['lb'] = torch.tensor([0, 0])
    graph.nodes['var'].data['ub'] = torch.tensor([4, 5])

    print(graph)
    print(graph.ntypes)
    print(graph.etypes)
    print(graph.canonical_etypes)
    print(graph.num_nodes())
    print(graph.num_nodes('var'))
    print(graph.nodes('var'))
    print(graph.nodes('cst'))
