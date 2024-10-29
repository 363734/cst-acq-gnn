import time

from sklearn.neural_network import MLPClassifier

from src.prior.sk.sk_classifier import get_classifier
from src.utils.metrics import print_metrics_all, compute_metrics
from src.models.gamma import Gamma, COMPARE_GAMMA
from src.utils.memoization.memoization_nn import save_sk_classifier, load_sk_classifier

from src.benchmark.load import load_benchmark

from src.utils.feature import Feature

from src.models.model import Model, Variables
from src.prior.prior_opt import parse_args


def get_models(benchfile: str, directory: str):
    return [Model(name, family, directory) for family, name in load_benchmark(benchfile)]


def get_features(model: Model, gamma: Gamma, max_dimensions: int, max_blocks: int, features_set: str):
    return [get_features_candidate(c, model.variables, gamma, max_dimensions, max_blocks, features_set) for c in
            model.bias]


def get_features_candidate(candidate, variables: Variables, gamma, max_dimensions: int, max_blocks: int,
                           features_set: str):
    if features_set == "aaai24":
        feat = Feature(candidate)
        feat_vect = [feat.vars_name_is_shared]
        for i in range(max_dimensions):
            feat_vect.append(feat.dims_is_shared(i))
            feat_vect.append(feat.dims_max(i))
            feat_vect.append(feat.dims_min(i))
            feat_vect.append(feat.dims_mean(i))
            feat_vect.append(feat.dims_mean_diff(i))
            # TODO not in AAAI paper
            # for j in range(max_blocks):
            #     feat_vect.append(feat.dims_block_is_shared(i, variables.var_dims_divisors, j))
            #     feat_vect.append(feat.dims_block_max(i, variables.var_dims_divisors, j))
            #     feat_vect.append(feat.dims_block_min(i, variables.var_dims_divisors, j))
            #     feat_vect.append(feat.dims_block_mean(i, variables.var_dims_divisors, j))
            #     feat_vect.append(feat.dims_block_mean_diff(i, variables.var_dims_divisors, j))
        feat_vect.append(feat.get_gamma_id(gamma))  # TODO test with one-hot? for NN ?
        feat_vect.append(feat.arity)
        feat_vect.append(feat.has_constant)
        feat_vect.append(feat.constants(0))
        return feat_vect
    else:
        Exception(f"Feature set {features_set} is not recognized")


def train_prior_sk(args):
    print(f":- Training SK classifier -------------")
    print(f"\twith parameters {args}")
    models = get_models(args.benchmark, args.data_directory)
    classifier, parameters = get_classifier(args)
    gamma = COMPARE_GAMMA
    feats = []
    gt = []
    print(f"::- load training models")
    for m in models:
        feats.extend(get_features(m, gamma, args.max_dimensions, args.max_blocks, args.features_set))
        gt.extend(m.ground_truth)

    print(f"::- start of training")
    start_t = time.time_ns()
    classifier.fit(feats, gt)
    end_t = time.time_ns()
    print(f"::- end of training")
    print(f"::- training time [{(end_t - start_t) / 1000000000.0}] seconds")
    parameters["args_at_training"] = args
    parameters["max_dimensions"] = args.max_dimensions
    parameters["max_blocks"] = args.max_blocks
    parameters["features_set"] = args.features_set
    parameters["gamma"] = gamma
    save_sk_classifier(args.model_file, classifier, parameters)
    print(f":- Training set metrics")
    if args.architecture_nn:
        print(f"\tLoss: {classifier.loss_curve_}")

    y = classifier.predict(feats)
    metrics = compute_metrics(y, gt)
    print_metrics_all(metrics)

    return None


def eval_prior_sk(args):
    print(f":- Evaluate SK classifier -------------")
    print(f"\twith parameters {args}")
    models = get_models(args.benchmark, args.data_directory)
    print(f"::- load prior")
    classifier, parameters = load_sk_classifier(args.model_file)
    feat = []
    gt_all = []
    print(f"::- load testing models")
    start_t = time.time_ns()
    for m in models:
        feat.append(get_features(m, parameters["gamma"], parameters["max_dimensions"], parameters["max_blocks"],
                                 parameters["features_set"]))
        gt_all.extend(m.ground_truth)
    end_t = time.time_ns()
    print(f"::- Total feature computation time [{(end_t - start_t) / 1000000000}] seconds")
    print(f"::- Mean feature computation time [{(end_t - start_t) / 1000000000 / len(gt_all)}] seconds")
    start_t = time.time_ns()

    y = [classifier.predict(feats) for feats in feat]
    end_t = time.time_ns()
    print(f":- Evaluation metrics")
    print(f"::- Total evaluation time [{(end_t - start_t) / 1000000000}] seconds")
    print(f"::- Mean evaluation time [{(end_t - start_t) / 1000000000 / len(gt_all)}] seconds")
    print(f"::- Evaluation All Benchmarks")
    y_all = []
    for y_one in y:
        y_all.extend(y_one)
    metrics = compute_metrics(y_all, gt_all)
    print_metrics_all(metrics)
    for i in range(len(models)):
        m = models[i]
        print(f"::- Evaluation benchmark {m.name}")
        metrics = compute_metrics(y[i], m.ground_truth)
        print_metrics_all(metrics)

    return None


if __name__ == "__main__":
    args = parse_args()

    dataset = [("sudoku", "sudoku4"), ("sudoku", "sudoku9")]

    models = [Model(name, family, args.directory) for family, name in dataset]

    for m in models:
        print(m.bias)
        print(get_features(m, 4, 4))
