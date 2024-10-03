from src.utils.memoization.memoization import p_save, p_load

_CLASSIFIER_TOKEN = "classifier"
_PARAMETERS_TOKEN = "parameters"


def save_nn(filename, classifier, parameters=None):
    if parameters is None:
        parameters = {}
    p_save(filename, {_CLASSIFIER_TOKEN: classifier, _PARAMETERS_TOKEN: parameters})
    print('\tSave NN classifier ({})'.format(filename))


def load_nn(filename):
    d = p_load(filename)
    classifier = d[_CLASSIFIER_TOKEN]
    parameters = d[_PARAMETERS_TOKEN]
    print('\tLoad NN classifier ({})'.format(filename))
    return classifier, parameters
