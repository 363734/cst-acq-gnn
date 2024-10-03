import argparse

from src.benchmark.load import load_benchmark
from src.log_analysis.experiment_runningCA import plot_table_results
from src.log_analysis.load_evaluation_log import load_evaluation_log
from src.log_analysis.load_training_log import load_training_log
from src.log_analysis.experiment_allbutone_training import plot_loss, plot_train, plot_eval, plot_allbut


def parse_args():
    parser = argparse.ArgumentParser()

    group_architecture = parser.add_mutually_exclusive_group(required=True)
    group_architecture.add_argument("-loss", action="store_true", dest="graph_loss", help="")
    group_architecture.add_argument("-train_met", action="store_true", dest="graph_train_met", help="")
    group_architecture.add_argument("-eval_met", action="store_true", dest="graph_eval_met", help="")
    group_architecture.add_argument("-all_but", action="store_true", dest="graph_all_but", help="")
    group_architecture.add_argument("-results_ca", action="store_true", dest="results_ca", help="")
    parser.set_defaults(graph_loss=False,
                        graph_train_met=False,
                        graph_test_met=False,
                        graph_all_but=False,
                        results_ca=False)

    parser.add_argument("-lf", "--log_file", type=str, required=True,
                        help="")
    parser.add_argument("-of", "--output_file", type=str, required=True,
                        help="")
    parser.add_argument("-pn", "--prior_name", type=str, required=False,
                        help="")

    parser.add_argument("-bm", "--benchmark", type=str, required=False,
                        help="storage directory for the training set instances")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.graph_loss:
        train_stat = load_training_log(args.log_file)
        if "loss" in train_stat:
            plot_loss(train_stat, args.output_file)
        else:
            print("no graph produced, this method do not have loss")

    if args.graph_train_met:
        train_stat = load_training_log(args.log_file)
        plot_train(train_stat, args.output_file)

    if args.graph_eval_met:
        test_stat = load_evaluation_log(args.log_file)
        plot_eval(test_stat, args.output_file)

    if args.graph_all_but:
        bench = load_benchmark(args.benchmark)
        logpatterns = args.log_file.split("---")
        train_pattern = logpatterns[0]
        test_pattern = logpatterns[1]
        train_stat = [load_training_log(train_pattern.format(b[1])) for b in bench]
        test_stat = [load_evaluation_log(test_pattern.format(b[1])) for b in bench]
        plot_allbut([b[1] for b in bench], train_stat, test_stat, args.output_file)

    if args.results_ca:
        assert "prior_name" in args, "should have prior"
        bench = load_benchmark(args.benchmark)
        plot_table_results(bench, args.log_file, args.prior_name, args.output_file)

