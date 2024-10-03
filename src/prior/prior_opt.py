import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    group_architecture = parser.add_mutually_exclusive_group(required=True)
    group_architecture.add_argument("-nn", action="store_true", dest="architecture_nn", help="")
    group_architecture.add_argument("-rf", action="store_true", dest="architecture_rf", help="")
    group_architecture.add_argument("-cnb", action="store_true", dest="architecture_cnb", help="")
    group_architecture.add_argument("-gnb", action="store_true", dest="architecture_gnb", help="")
    group_architecture.add_argument("-svm", action="store_true", dest="architecture_svm", help="")
    group_architecture.add_argument("-gnn", action="store_true", dest="architecture_gnn", help="")
    parser.set_defaults(architecture_nn=False,
                        architecture_rf=False,
                        architecture_cnb=False,
                        architecture_gnb=False,
                        architecture_svm=False,
                        architecture_gnn=False)

    parser.add_argument("-fs", "--features_set", type=str, required=False, default="aaai24",
                        help="which set of features we allow")

    parser.add_argument("-bm", "--benchmark", type=str, required=True,
                        help="storage directory for the training set instances")

    parser.add_argument("-dd", "--data_directory", type=str, required=True,
                        help="storage directory for the training set instances")

    parser.add_argument("-mf", "--model_file", type=str, required=True,
                        help="name of the file to store the model")

    parser.add_argument("--learning_rate", default=0.01, type=float, action='store',
                        dest='learning_rate', help="learning rate of the optimizer (input feature of NN and GNN)")
    parser.add_argument("--max_dimensions", default=3, type=float, action='store',
                        dest='max_dimensions', help="")
    parser.add_argument("--max_blocks", default=3, type=float, action='store',
                        dest='max_blocks', help="")

    # NN parameters
    parser.add_argument("--nn:hidden_layer_sizes", default=64, type=int, action='store',
                        dest='nn_hidden_layer_sizes', help="size of the hidden layer (input feature of NN)")
    # Categorican Naive Bayes (CNB)
    parser.add_argument("--cnb:min_categories", default=5, type=int, action='store',
                        dest='cnb_min_categories', help="size of the hidden layer (input feature of NN)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
