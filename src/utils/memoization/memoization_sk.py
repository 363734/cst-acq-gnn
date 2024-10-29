from src.utils.memoization.memoization import p_save, p_load

_CLASSIFIER_TOKEN = "classifier"
_PARAMETERS_TOKEN = "parameters"


def save_sk_classifier(filename, classifier, parameters=None):
    if parameters is None:
        parameters = {}
    p_save(filename, {_CLASSIFIER_TOKEN: classifier, _PARAMETERS_TOKEN: parameters})
    print('\tSave SK classifier ({})'.format(filename))


def load_sk_classifier(filename):
    d = p_load(filename)
    classifier = d[_CLASSIFIER_TOKEN]
    parameters = d[_PARAMETERS_TOKEN]
    print('\tLoad SK classifier ({})'.format(filename))
    return classifier, parameters
