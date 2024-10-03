import copy

import dgl
import torch

from src.utils import get_var_dims, get_var_ndims, get_var_name, get_relation
from cpmpy.transformations.get_variables import get_variables


def get_var_id(var_map, var):
    if var not in var_map:
        new_id = len(var_map)
        name = get_var_name(var)
        var_map[var] = {'var': var, 'id': new_id, 'name': name, 'name_hash': abs(hash(name)) % (10 ** 4),
                        'nb_dims': get_var_ndims(var),
                        'dims': get_var_dims(var), 'lb': var.lb, 'ub': var.ub, 'is_bool': int(var.is_bool()),
                        'is_int': int(not (var.is_bool()))}
        print("new var added: {}".format(var))
    return var_map[var]['id']


def add_var_node_feature(graph, var_map):
    var_list = list(var_map)
    var_list = sorted(var_list, key=lambda var: var_map[var]['id'])
    lexico_list = sorted(var_list, key=lambda var: (var_map[var]['name'], var_map[var]['dims']))
    for i in range(len(lexico_list)):
        var_map[lexico_list[i]]['ordering'] = i
    graph.nodes['var'].data['name_hash'] = torch.tensor([var_map[var]['name_hash'] for var in var_list])
    graph.nodes['var'].data['is_int'] = torch.tensor([var_map[var]['is_int'] for var in var_list])
    graph.nodes['var'].data['is_bool'] = torch.tensor([var_map[var]['is_bool'] for var in var_list])
    graph.nodes['var'].data['lb'] = torch.tensor([var_map[var]['lb'] for var in var_list])
    graph.nodes['var'].data['ub'] = torch.tensor([var_map[var]['ub'] for var in var_list])
    graph.nodes['var'].data['nb_dims'] = torch.tensor([var_map[var]['nb_dims'] for var in var_list])
    graph.nodes['var'].data['dims'] = torch.tensor([var_map[var]['dims'] for var in var_list])
    #TODO set dims here with same size as tensor do not accept sublist of different size...
    graph.nodes['var'].data['ordering'] = torch.tensor([var_map[var]['ordering'] for var in var_list])


def create_graph_unknown(gamma, bias):
    map_vars = {}  # gather var info when new var is selected
    edges = []
    b_arity = []
    b_rel_id = []
    b_share_name = []
    b_dim_0_same = []
    b_dim_0_min = []
    b_dim_0_max = []
    b_dim_1_same = []
    b_dim_1_min = []
    b_dim_1_max = []
    for i in range(len(bias)):
        b = bias[i]
        xs = get_variables(b)
        for x in xs:
            var_id = get_var_id(map_vars, x)  # add if var has not been seen before
            edges.append((var_id, i))  # new edge
        # compute arity of bias
        b_arity.append(len(xs))
        rel_id = get_relation(b, gamma)
        b_rel_id.append([1 if i == rel_id else 0 for i in range(len(gamma) + 1)])
        b_share_name.append(int(all([map_vars[x]["name"] == map_vars[xs[0]]["name"] for x in xs])))
        b_dim_0_same.append(int(all([map_vars[x]["dims"][0] == map_vars[xs[0]]["dims"][0] for x in xs])))
        b_dim_0_min.append(min([map_vars[x]["dims"][0] for x in xs]))
        b_dim_0_max.append(max([map_vars[x]["dims"][0] for x in xs]))
        b_dim_1_same.append(int(all([map_vars[x]["dims"][1] == map_vars[xs[0]]["dims"][1] for x in xs])))
        b_dim_1_min.append(min([map_vars[x]["dims"][1] for x in xs]))
        b_dim_1_max.append(max([map_vars[x]["dims"][1] for x in xs]))
    graph_data = {
        ('var', 'in scope of', 'cst'): ([e[0] for e in edges], [e[1] for e in edges]),
        ('cst', 'use', 'var'): ([e[1] for e in edges], [e[0] for e in edges])
    }
    graph = dgl.heterograph(graph_data)

    graph.nodes['cst'].data['unknown'] = torch.tensor([1] * len(bias))
    graph.nodes['cst'].data['yes'] = torch.tensor([0] * len(bias))
    graph.nodes['cst'].data['no'] = torch.tensor([0] * len(bias))
    graph.nodes['cst'].data['arity'] = torch.tensor(b_arity)
    graph.nodes['cst'].data['shared_name'] = torch.tensor(b_share_name)
    graph.nodes['cst'].data['dim_0_same'] = torch.tensor(b_dim_0_same)
    graph.nodes['cst'].data['dim_0_min'] = torch.tensor(b_dim_0_min)
    graph.nodes['cst'].data['dim_0_max'] = torch.tensor(b_dim_0_max)
    graph.nodes['cst'].data['dim_1_same'] = torch.tensor(b_dim_1_same)
    graph.nodes['cst'].data['dim_1_min'] = torch.tensor(b_dim_1_min)
    graph.nodes['cst'].data['dim_1_max'] = torch.tensor(b_dim_1_max)
    graph.nodes['cst'].data['gamma'] = torch.tensor(b_rel_id)

    add_var_node_feature(graph, map_vars)

    return graph


def choose_feats(graph, opts):
    # TODO automatic selection of feature trought options (to allow loading of graph where everything is precomputed
    # TODO find the right features
    # TODO add automatic max dim
    # TODO add latent dim
    to_stack = []
    if opts.feat_cst_indicators:
        to_stack.append(graph.nodes['cst'].data['unknown'])
        to_stack.append(graph.nodes['cst'].data['yes'])
        to_stack.append(graph.nodes['cst'].data['no'])
    if opts.feat_cst_arity:
        to_stack.append(graph.nodes['cst'].data['arity'])
    if opts.feat_cst_shared_name:
        to_stack.append(graph.nodes['cst'].data['shared_name'])
    if opts.feat_nbdim > 0:
        to_stack.append(graph.nodes['cst'].data['dim_0_same'])
        to_stack.append(graph.nodes['cst'].data['dim_0_min'])
        to_stack.append(graph.nodes['cst'].data['dim_0_max'])
    if opts.feat_nbdim > 1:
        to_stack.append(graph.nodes['cst'].data['dim_1_same'])
        to_stack.append(graph.nodes['cst'].data['dim_1_min'])
        to_stack.append(graph.nodes['cst'].data['dim_1_max'])
    to_cat = graph.nodes['cst'].data['gamma']

    graph.nodes['cst'].data['feats_raw'] = torch.cat((torch.stack(to_stack, dim=1), to_cat), dim=1).type(
        torch.FloatTensor)

    to_stack = []
    to_stack.append(graph.nodes['var'].data['ordering'])
    to_stack.append(graph.nodes['var'].data['nb_dims'])
    if opts.feat_var_name:
        to_stack.append(graph.nodes['var'].data['name_hash'])
    if opts.feat_var_type:
        to_stack.append(graph.nodes['var'].data['is_int'])
        to_stack.append(graph.nodes['var'].data['is_bool'])
    if opts.feat_var_bounds:
        to_stack.append(graph.nodes['var'].data['lb'])
        to_stack.append(graph.nodes['var'].data['ub'])
    graph.nodes['var'].data['feats_raw'] = torch.stack(to_stack, dim=1)
    if opts.feat_nbdim > 0:
        to_cat = [d[:opts.feat_nbdim].tolist() if len(d) >= opts.feat_nbdim else d.tolist() + [0] * (opts.feat_nbdim - len(d)) for d in
                  graph.nodes['var'].data['dims']]
        to_cat = torch.tensor(to_cat)
        graph.nodes['var'].data['feats_raw'] = torch.cat((graph.nodes['var'].data['feats_raw'], to_cat), dim=1)
    graph.nodes['var'].data['feats_raw'] = graph.nodes['var'].data['feats_raw'].type(torch.FloatTensor)


# TODO add dim max checked
# if map_vars[x]['ndims'] >= nb_dim_selected:
#     x_dim.append(map_vars[x]['dims'][:nb_dim_selected])
# else:
#     x_dim.append(map_vars[x]['dims'] + [-1] * (nb_dim_selected - map_vars[x]['ndims']))

# get the mask of the graph, correspond to unknown cst
def get_mask(graph):
    return graph.nodes['cst'].data['unknown'].nonzero(as_tuple=True)


# return the size of the feature of cst nodes
def nb_feat_cst(graph):
    return graph.nodes['cst'].data['feats_raw'].size(dim=1)


# return the size of the feature of var nodes
def nb_feat_var(graph):
    return graph.nodes['var'].data['feats_raw'].size(dim=1)


# apply new scenario to graph, result in a new graph
def switch_senario(graph, indicators):
    unknown, yes, no = indicators
    uptaded_g = copy.deepcopy(graph)  # TODO: for training, maybe pass by pickle to load quickly new copy
    uptaded_g.nodes['cst'].data['unknown'] = torch.tensor(unknown)
    uptaded_g.nodes['cst'].data['yes'] = torch.tensor(yes)
    uptaded_g.nodes['cst'].data['no'] = torch.tensor(no)
    return uptaded_g


if __name__ == "__main__":
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
