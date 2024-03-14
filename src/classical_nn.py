import time
from statistics import mean, stdev

from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_validate

from utils import *
from cpmpy import *
from src.memoization import p_load

from sklearn.neural_network import MLPClassifier


class ProblemInstance:

    def __init__(self, labels=None, B=list(), X=set()):

        self.debug_mode = False

        self.B = B
        self.gamma = list(set([c.get_relation() for c in B]))  # Language: relations derived from input model
        self.X = list(X)
        self.labels = labels

        # Hash the variables
        self.hashX = [hash(x) for x in self.X]

        self.datasetX = []

        # To be used in the constraint features
        # -------------------------------------

        # Length of dimensions per variable name
        self.var_names = list(set([get_var_name(x) for x in self.X]))
        var_dims = [[get_var_dims(x) for x in self.X if get_var_name(x) == self.var_names[i]] for i in
                    range(len(self.var_names))]
        self.dim_lengths = [
            [np.max([var_dims[i][k][j] for k in range(len(var_dims[i]))]) + 1 for j in range(len(var_dims[i][0]))] for i
            in range(len(var_dims))]

        self.dim_divisors = []

        for i in range(len(self.dim_lengths)):
            dim_divisors = []
            for j in range(len(self.dim_lengths[i])):
                divisors = get_divisors(self.dim_lengths[i][j])
                dim_divisors.append(divisors)

            self.dim_divisors.append(dim_divisors)

    def get_dataset(self):
        return self. datasetX

    def generate_datasetX(self):

        self.datasetX = []
        for c in self.B:
            self.datasetX.append(self.get_con_features(c))

    def get_con_features(self, c):
        # Returns the features associated with constraint c
        features = []
        scope = get_scope(c)

        var_name = get_var_name(scope[0])
        var_name_same = all([(get_var_name(var) == var_name) for var in scope])
        features.append(var_name_same)

        # Global var dimension properties
        vars_ndims = [get_var_ndims(var) for var in scope]
        ndims_max = max(vars_ndims)

        vars_dims = [get_var_dims(var) for var in scope]
        dim = []

        # TODO add the right amount of Trues in features, this needs to be fixed
        for j in range(ndims_max):
            dim.append([vars_dims[i][j] for i in range(len(vars_dims)) if len(vars_dims[i]) > j])

            dimj_has = len(dim[j]) > 0
            # features.append(dimj_has)

            if dimj_has:
                dimj_same = all([dim_temp == dim[j][0] for dim_temp in dim[j]])
                features.append(dimj_same)
                dimj_max = max(dim[j])
                features.append(dimj_max)
                dimj_min = min(dim[j])
                features.append(dimj_min)
                dimj_avg = mean(dim[j])
                features.append(dimj_avg)
#                dimj_dev = stdev(dim[j])
                dimj_diff = average_difference(dim[j])
                features.append(dimj_diff)

            else:
                features.append(True)
                for i in range(3): features.append(0)
                features.append(0.0)

            # divisors features --- per var_name, per dimension (up to max dimensions), per divisor (up to max divisors)
            # block same, max, min, average, spread
            # for each variable name (type of variable
            for var_name in range(len(self.var_names)):
                # for each dimension of that type of variable
                vars_block = [[dim[j][var] // divisor for var in range(len(scope))
                               if self.var_names[var_name] == get_var_name(scope[var])]
                              for divisor in self.dim_divisors[var_name][j]]

                for l in range(len(self.dim_divisors[var_name][j])):
                    if self.debug_mode:
                        print(vars_block[l])
                    block_same = all([vars_block[l][var] == vars_block[l][0] for var in range(len(vars_block[l]))])
                    features.append(block_same)
                    block_max = max(vars_block[l])
                    features.append(block_max)
                    block_min = min(vars_block[l])
                    features.append(block_min)
                    block_avg = mean(vars_block[l])
                    features.append(block_avg)
                    block_dev = stdev(vars_block[l])
                    features.append(block_dev)

        con_in_gamma = -1
        con_relation = c.get_relation()
        for i in range(len(self.gamma)):
            if self.gamma[i] == con_relation:
                con_in_gamma = i
                break

        if con_in_gamma == -1:
            raise Exception(f"Check why constraint relation was not found in relations: constraint {c}, relation: {con_relation}")
        features.append(con_in_gamma)

        arity = len(scope)
        features.append(arity)

        num = get_constant(c)
        has_const = len(num) > 0
        features.append(has_const)

        if has_const:
            features.append(int(num[0]))
        else:
            features.append(0)

        return features


bench = "exam_timetabling"

if bench == "sudoku":
    constraints = p_load('../data/sudoku/dataset_C.pickle')
    datasetY = p_load('../data/sudoku/dataset_CY.pickle')
elif bench == "jsudoku":
    constraints = p_load('../data/jsudoku/dataset_C.pickle')
    datasetY = p_load('../data/jsudoku/dataset_CY.pickle')
elif bench == "nurse_rostering":
    constraints = p_load('../data/nurse_rostering/dataset_C.pickle')
    datasetY = p_load('../data/nurse_rostering/dataset_CY.pickle')
elif bench == "exam_timetabling":
    constraints = p_load('../data/exam_timetabling/dataset_C.pickle')
    datasetY = p_load('../data/exam_timetabling/dataset_CY.pickle')
else:
    raise NotImplementedError("Benchmark not implemented")


problem = ProblemInstance(B=constraints, X=get_variables_from_constraints(constraints))
problem.generate_datasetX()

datasetX = problem.get_dataset()

classifier = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam',
                                               random_state=1, learning_rate_init=0.01)

print("Starting training ---------")
start_t = time.time()
classifier.fit(datasetX, datasetY)
end_t = time.time()
print("training time: ", end_t-start_t)

scoring = {"Acc": 'accuracy', "Bal_Acc": make_scorer(balanced_accuracy_score), "f1-score": 'f1'}

print("Starting cross validation ---------")
scores = cross_validate(classifier, datasetX, datasetY,
                        scoring=scoring,
                        cv=10)  # fit_params={'sample_weight': compute_sample_weights(Y)})

print(scores)



