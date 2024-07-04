from src.utils.memoization.memoization import p_save, p_load


def save_nn(filename, classifier):
    p_save(filename, classifier)
    print('\tSave NN classifier ({})'.format(filename))


def load_nn(filename):
    classifier = p_load(filename)
    print('\tLoad NN classifier ({})'.format(filename))
    return classifier