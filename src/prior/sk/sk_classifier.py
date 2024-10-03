from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_classifier(args):
    parameters = {}
    if args.architecture_nn:
        parameters["architecture"] = "nn"
        parameters["nn_hidden_layer_sizes"] = args.nn_hidden_layer_sizes
        parameters["learning_rate"] = args.learning_rate
        return MLPClassifier(hidden_layer_sizes=(args.nn_hidden_layer_sizes,), activation='relu', solver='adam',
                             random_state=1, learning_rate_init=args.learning_rate), parameters
    if args.architecture_rf:
        parameters["architecture"] = "rf"
        return RandomForestClassifier(), parameters
    if args.architecture_cnb:
        parameters["architecture"] = "cnb"
        parameters["cnb_min_categories"] = args.cnb_min_categories
        return CategoricalNB(min_categories=args.cnb_min_categories), parameters
    if args.architecture_gnb:
        parameters["architecture"] = "gnb"
        return GaussianNB(), parameters
    if args.architecture_svm:
        parameters["architecture"] = "svm"
        return SVC(kernel='rbf', C=100, gamma='scale', probability=True), parameters

    Exception("No supported architecture has been selected")
