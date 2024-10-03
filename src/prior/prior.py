from src.models.model import Variables
from src.prior.sk.prior_sk import get_features_candidate

from src.utils.memoization.memoization_nn import load_nn

class Prior:
    def __init__(self, file):
        self.vars = None
        self.file = file
        self.classifier, self.parameter = load_nn(self.file)
        self.architecture = self.parameter['architecture']
        if self.architecture == 'gnn':
            print('need to add support for gnn')
        self.gamma = self.parameter['gamma']
        self.max_dimension = self.parameter['max_dimensions']
        self.max_blocks = self.parameter['max_blocks']
        self.features_set = self.parameter['features_set']

    def variables(self, var):
        self.vars = Variables(var.flatten())

    def predict(self, c):
        feat = get_features_candidate(c, self.vars, self.gamma, self.max_dimension, self.max_blocks,
                                      self.features_set)
        return self.classifier.predict([feat])[0]

    def __str__(self):
        s = f"{self.file}\n{self.architecture}\n{self.gamma}\n{self.max_dimension}\n{self.max_blocks}\n{self.features_set}"
        return s


if __name__ == "__main__":
    classifier, parameter = load_nn("../../target/models/nn/nn_classical_ca.pickle")
    print(parameter)
    p = Prior("../../target/models/nn/nn_classical_ca.pickle")
    print(p)
