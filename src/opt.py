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

parser.add_option("--model:nb_dims_max", default=-1, type=int, action='store',
                  dest='model_nb_dims_max', help="nb of dimensions to add to model (-1 if max)")

# - upper bound on nb dimension ?

# parser.add_option("--grp:filterbias", default=False,
#                   action="store_true", dest="grp_filterbiais",
#                   help="do not include 'no' feature in cst nodes")

if __name__ == "__main__":
    (options, args) = parser.parse_args()
    print('test')
    print(args)
    print(options)
