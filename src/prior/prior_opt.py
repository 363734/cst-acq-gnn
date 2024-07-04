import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-nn", action="store_true", dest="architecture_nn", help="")
    parser.add_argument("-gnn", action="store_true", dest="architecture_gnn", help="")
    parser.set_defaults(architecture_nn=False,
                        architecture_gnn=False)

    parser.add_argument("-bm", "--benchmark", type=str, required=True,
                        help="storage directory for the training set instances")

    parser.add_argument("-dd", "--data_directory", type=str, required=True,
                        help="storage directory for the training set instances")

    parser.add_argument("-mf", "--model_file", type=str, required=True,
                        help="name of the file to store the model")

    parser.add_argument("--learning_rate", default=0.01, type=float, action='store',
                      dest='learning_rate', help="learning rate of the optimizer")
    parser.add_argument("--max_dimensions", default=3, type=float, action='store',
                      dest='max_dimensions', help="")
    parser.add_argument("--max_blocks", default=3, type=float, action='store',
                      dest='max_blocks', help="")

    # NN parameters
    parser.add_argument("--nn:hidden_layer_sizes", default=64, type=int, action='store',
                        dest='nn_hidden_layer_sizes', help="size of the hidden layer (input feature of GNN)")

    args = parser.parse_args()

    assert args.architecture_nn ^ args.architecture_gnn, "architecture neural net XOR graph neural net"

    return args
