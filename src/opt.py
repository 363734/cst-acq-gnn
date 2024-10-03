from optparse import OptionParser

parser = OptionParser()
# Options for the GNN
parser.add_option("--gnn:infeat", default=64, type=int, action='store',
                  dest='gnn_infeat', help="size of the feature embedding (input feature of GNN)")
parser.add_option("--gnn:hidfeat", default=64, type=int, action='store',
                  dest='gnn_hidfeat', help="size of the hidden embedding (hidden layer of GNN)")
parser.add_option("--gnn:nblayers", default=2, type=int, action='store',
                  dest='gnn_nblayers', help="nb of layer of the model")
parser.add_option("--gnn:nbepochs", default=200, type=int, action='store',
                  dest='gnn_nbepochs', help="nb of epoch for training")
parser.add_option("--gnn:learning_rate", default=0.01, type=float, action='store',
                  dest='gnn_learning_rate', help="learning rate of the optimizer")

# Feature options
parser.add_option("--feat:dim", action="store", type=int, dest="feat_nbdim", help="Add nb dim info as graph features (cst node)")

parser.set_defaults(feat_nbdim=2)

# Cst specific feature options
parser.add_option("--feat:cst:arity", action="store_true", dest="feat_cst_arity", help="Add arity as graph features (cst node)")
parser.add_option("--feat:cst:noarity", action="store_false", dest="feat_cst_arity", help="No arity as graph features (cst node)")

parser.add_option("--feat:cst:indicators", action="store_true", dest="feat_cst_indicators", help="Add indicators (unknown, yes, no) as graph features (cst node)")
parser.add_option("--feat:cst:noindicators", action="store_false", dest="feat_cst_indicators", help="No indicators (unknown, yes, no) as graph features (cst node)")

parser.add_option("--feat:cst:sharedname", action="store_true", dest="feat_cst_shared_name", help="Add shared name as graph features (cst node)")
parser.add_option("--feat:cst:nosharedname", action="store_false", dest="feat_cst_shared_name", help="No shared name (unknown, yes, no) as graph features (cst node)")

parser.set_defaults(feat_cst_indicators=False,
                    feat_cst_arity=True,
                    feat_cst_shared_name=True)

# Var specific feature options
parser.add_option("--feat:var:type", action="store_true", dest="feat_var_type", help="Add type (int/bool) as graph features (var node)")
parser.add_option("--feat:var:notype", action="store_false", dest="feat_var_type", help="No type as graph features (var node)")

parser.add_option("--feat:var:bound", action="store_true", dest="feat_var_bounds", help="Add the bounds (lb/ub) as graph features (var node)")
parser.add_option("--feat:var:nobound", action="store_false", dest="feat_var_bounds", help="No bounds as graph features (var node)")

parser.add_option("--feat:var:name", action="store_true", dest="feat_var_name", help="Add the name (numerical hash) as graph features (var node)")
parser.add_option("--feat:var:noname", action="store_false", dest="feat_var_name", help="No name as graph features (var node)")

parser.set_defaults(feat_var_type=True,
                    feat_var_bounds=False,
                    feat_var_name=False)


if __name__ == "__main__":
    (options, args) = parser.parse_args()
    print('test')
    print(args)
    print(options)
    print(options.gnn_hidfeat)
