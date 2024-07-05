import time

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from src.metrics import compute_metrics, print_metrics_all, compute_metrics
from src.utils.memoization.memoization_nn import save_nn, load_nn

from src.benchmark_classical_ca.load import load_benchmark

from src.utils.feature import Feature

from src.models.model import Model
from src.prior.prior_opt import parse_args


def get_models(benchfile: str, directory: str):
    return [Model(name, family, directory) for family, name in load_benchmark(benchfile)]


def get_features(model: Model, max_dimensions: int, max_blocks: int):
    return [get_features_candidate(model, c, max_dimensions, max_blocks) for c in model.bias]


def get_features_candidate(model: Model, candidate, max_dimensions: int, max_blocks: int):
    feat = Feature(candidate)
    feat_vect = [feat.vars_name_is_shared]
    for i in range(max_dimensions):
        feat_vect.append(feat.dims_is_shared(i))
        feat_vect.append(feat.dims_max(i))
        feat_vect.append(feat.dims_min(i))
        feat_vect.append(feat.dims_mean(i))
        feat_vect.append(feat.dims_mean_diff(i))
        for j in range(max_blocks):
            feat_vect.append(feat.dims_block_is_shared(i, model.variables.var_dims_divisors, j))
            feat_vect.append(feat.dims_block_max(i, model.variables.var_dims_divisors, j))
            feat_vect.append(feat.dims_block_min(i, model.variables.var_dims_divisors, j))
            feat_vect.append(feat.dims_block_mean(i, model.variables.var_dims_divisors, j))
            feat_vect.append(feat.dims_block_mean_diff(i, model.variables.var_dims_divisors, j))
    feat_vect.append(feat.get_gamma_id(model.gamma))
    feat_vect.append(feat.arity)
    feat_vect.append(feat.has_constant)
    feat_vect.append(feat.constants(0))
    return feat_vect


def train_prior_nn(args):
    print(f":- Training NN -------------")
    print(f"\twith parameters {args}")
    models = get_models(args.benchmark, args.data_directory)
    classifier = MLPClassifier(hidden_layer_sizes=(args.nn_hidden_layer_sizes,), activation='relu', solver='adam',
                               random_state=1, learning_rate_init=args.learning_rate)
    feats = []
    gt = []
    print(f"::- load training models")
    for m in models:
        feats.extend(get_features(m, args.max_dimensions, args.max_blocks))
        gt.extend(m.ground_truth)

    print(f"::- start of training")
    start_t = time.time_ns()
    classifier.fit(feats, gt)
    end_t = time.time_ns()
    print(f"::- end of training")
    print(f"::- training time [{(end_t - start_t)/1000}] seconds")

    save_nn(args.model_file, classifier)
    print(f":- Training set metrics")
    print(f"\tLoss: {classifier.loss_curve_}")

    y = classifier.predict(feats)
    metrics = compute_metrics(y, gt)
    print_metrics_all(metrics)

    return None


def eval_prior_nn(args):
    print(f":- Evaluate NN -------------")
    print(f"\twith parameters {args}")
    models = get_models(args.benchmark, args.data_directory)
    classifier = load_nn(args.model_file)
    feats = []
    gt = []
    print(f"::- load training models")
    for m in models:
        feats.extend(get_features(m, args.max_dimensions, args.max_blocks))
        gt.extend(m.ground_truth)

    print(f":- Evaluation metrics")
    start_t = time.time_ns()
    y = classifier.predict(feats)
    end_t = time.time_ns()
    print(f"::- training time [{(end_t - start_t) / 1000}] seconds")
    print(y)
    metrics = compute_metrics(y, gt)
    print_metrics_all(metrics)

    return None


if __name__ == "__main__":
    args = parse_args()

    dataset = [("sudoku", "sudoku4"), ("sudoku", "sudoku9")]

    models = [Model(name, family, args.directory) for family, name in dataset]

    for m in models:
        print(m.bias)
        print(get_features(m, 4, 4))
