from statistics import mean

from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, make_scorer, accuracy_score, f1_score
from sklearn.model_selection import cross_validate

from src.utils.utils import *
from cpmpy import *
from src.utils.memoization import p_load

from sklearn.neural_network import MLPClassifier

from src.utils.feature import Feature

max_dimensions = 4
max_blocks = 4





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
        # for each var name existing, gather the dimensions of each var of the same names

        self.dim_lengths = [
            [np.max([var_dims[i][k][j] for k in range(len(var_dims[i]))]) + 1 for j in range(len(var_dims[i][0]))] for i
            in range(len(var_dims))]
        # for each var name, for each dim get the length of the dim by looking over each variable of the name (for ex for the sudoku = 9)


        self.dim_divisors = []

        # for each var names (each category of vars), an array of for each dim with its 'length', get all divisor of that length
        for i in range(len(self.dim_lengths)):
            dim_divisors = []
            for j in range(len(self.dim_lengths[i])):
                divisors = get_divisors(self.dim_lengths[i][j])
                dim_divisors.append(divisors)

            self.dim_divisors.append(dim_divisors)

    def get_dataset(self):
        if len(self.datasetX) == 0:
            self.generate_datasetX()
        return self.datasetX

    def generate_datasetX(self):

        self.datasetX = []
        for c in self.B:
            self.datasetX.append(self.get_con_features(c))

    def get_con_features(self, c):
        # Returns the features associated with constraint c
        features = []
        scope = get_scope(c)
        arity = len(scope)

        var_names = [self.var_names.index(get_var_name(scope[i])) for i in range(arity)]
        var_name_same = all([var_name == var_names[0] for var_name in var_names])
        features.append(var_name_same)

        # Global var dimension properties
        vars_dims = [get_var_dims(var) for var in scope]
        dim = []

        # TODO add the right amount of Trues in features, this needs to be fixed
        for j in range(max_dimensions):
            dim.append([vars_dims[i][j] for i in range(len(vars_dims)) if len(vars_dims[i]) > j])

            dimj_has = len(dim[j]) > 0
            features.append(dimj_has)
            if not dimj_has:

                # dummy dimension features
                features.append(True)
                for i in range(3): features.append(0)
                features.append(0.0)

                # dummy latent dimension features
                for l in range(max_blocks):
                    features.append(True)
                    for i in range(3): features.append(0)
                    features.append(0.0)
                continue

            dimj_same = all([dim_temp == dim[j][0] for dim_temp in dim[j]])
            features.append(dimj_same)
            dimj_max = max(dim[j])
            features.append(dimj_max)
            dimj_min = min(dim[j])
            features.append(dimj_min)
            dimj_avg = mean(dim[j])
            features.append(dimj_avg)
            dimj_diff = average_difference(dim[j])
            features.append(dimj_diff)

            vars_block = [[dim[j][var] // self.dim_divisors[var_names[var]][j][divisor]
                           if len(self.dim_divisors[var_names[var]][j]) > divisor else -1 for var in range(len(scope))]
                          for divisor in range(max_blocks)]

            # divisors features --- per var_name, per dimension (up to max dimensions), per divisor (up to max divisors)
            # block same, max, min, average, spread
            # for each variable name (type of variable)
            # for var_name in range(len(self.var_names)):
            # for each dimension of that type of variable
            #            vars_block = [[dim[j][var] // divisor for var in range(len(scope))
            #                           if self.var_names[var_name] == get_var_name(scope[var])]
            #                          for divisor in self.dim_divisors[var_name][j]]

            for l in range(max_blocks):
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
                block_diff = average_difference(vars_block[l])
                features.append(block_diff)

        con_in_gamma = -1
        con_relation = c.get_relation()
        for i in range(len(self.gamma)):
            if self.gamma[i] == con_relation:
                con_in_gamma = i
                break

        if con_in_gamma == -1:
            raise Exception(
                f"Check why constraint relation was not found in relations: constraint {c}, relation: {con_relation}")
        features.append(con_in_gamma)
        features.append(arity)

        num = get_constant(c)
        has_const = len(num) > 0
        features.append(has_const)

        if has_const:
            features.append(int(num[0]))
        else:
            features.append(0)

        return features


def benchmark_exp(datasetX, datasetY):

    classifier = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam',
                               random_state=1, learning_rate_init=0.01)

    print("Starting training ---------")
    start_t = time.time()

    print(np.shape(datasetX), np.shape(datasetY))
    classifier.fit(datasetX, datasetY)
    end_t = time.time()
    print("training time: ", end_t - start_t)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(classifier.loss_curve_)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Loss')
    plt.show()

    ### Cross validation scores --------

    scoring = {"Acc": 'accuracy', "Bal_Acc": make_scorer(balanced_accuracy_score), "f1-score": 'f1'}

    print("Starting cross validation ---------")
    scores = cross_validate(classifier, datasetX, datasetY,
                            scoring=scoring,
                            cv=10)  # fit_params={'sample_weight': compute_sample_weights(Y)})

    print(scores)

def combined_exp(X, Y, benchmark_names):

    ### Full training with all (to see time and learning curve ------

    # Concat for full training
    datasetX = []
    datasetY = []
    [datasetX.extend(x) for x in X]
    [datasetY.extend(y) for y in Y]

    classifier = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam',
                               random_state=1, learning_rate_init=0.01)

    print("Starting training ---------")
    start_t = time.time()
    classifier.fit(datasetX, datasetY)
    end_t = time.time()
    print("training time: ", end_t - start_t)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(classifier.loss_curve_)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Loss')
    plt.show()

    print("Starting leave-one-out cross-validation ---------")
    ### Leave one out cross validation
    folds = len(X)
    for fold in range(folds):

        # create train and test datasets for current fold -- train with all except one (the one corresponding to the fold's number), and test on that one
        trainX = []
        [trainX.extend(X[xi]) for xi in range(folds) if xi != fold]
        trainY = []
        [trainY.extend(Y[yi]) for yi in range(folds) if yi != fold]

        testX = X[fold]
        testY = Y[fold]
        print(np.shape(trainX),np.shape(trainY))
        classifier.fit(trainX, trainY)
        y = classifier.predict(testX)

        acc = accuracy_score(testY, y)
        balanced_acc = balanced_accuracy_score(testY, y)
        f1 = f1_score(testY, y)

        labels = ['Accuracy', 'Balanced Accuracy', 'F1 Score']
        scores = [acc, balanced_acc, f1]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, scores, color=['blue', 'green', 'orange'])
        plt.ylabel('Score')
        plt.title(benchmark_names[fold])
        plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
        plt.show()


def main():

    setting = "combined"
    benchmark_names = ["sudoku", "jsudoku", "nurse_rostering", "exam_timetabling"]

    bench = "nurse_rostering"

    if setting == "bench":
        if bench == "sudoku":
            constraints = p_load('../../data/sudoku/dataset_C.pickle')
            datasetY = p_load('../../data/sudoku/dataset_CY.pickle')
        elif bench == "jsudoku":
            constraints = p_load('../../data/jsudoku/dataset_C.pickle')
            datasetY = p_load('../../data/jsudoku/dataset_CY.pickle')
        elif bench == "nurse_rostering":
            constraints = p_load('../../data/nurse_rostering/dataset_C.pickle')
            datasetY = p_load('../../data/nurse_rostering/dataset_CY.pickle')
        elif bench == "exam_timetabling":
            constraints = p_load('../../data/exam_timetabling/dataset_C.pickle')
            datasetY = p_load('../../data/exam_timetabling/dataset_CY.pickle')
        else:
            raise NotImplementedError("Benchmark not implemented")

        problem = ProblemInstance(B=constraints, X=get_variables_from_constraints(constraints))
        datasetX = problem.get_dataset()

        benchmark_exp(datasetX, datasetY)

    elif setting == "combined":

        constraints = p_load('../../data/sudoku/dataset_C.pickle')
        sudokuY = p_load('../../data/sudoku/dataset_CY.pickle')
        sudoku = ProblemInstance(B=constraints, X=get_variables_from_constraints(constraints))
        constraints = p_load('../../data/jsudoku/dataset_C.pickle')
        jsudokuY = p_load('../../data/jsudoku/dataset_CY.pickle')
        jsudoku = ProblemInstance(B=constraints, X=get_variables_from_constraints(constraints))
        constraints = p_load('../../data/nurse_rostering/dataset_C.pickle')
        nurse_rostY = p_load('../../data/nurse_rostering/dataset_CY.pickle')
        nurse_rost = ProblemInstance(B=constraints, X=get_variables_from_constraints(constraints))
        constraints = p_load('../../data/exam_timetabling/dataset_C.pickle')
        exam_ttY = p_load('../../data/exam_timetabling/dataset_CY.pickle')
        exam_tt = ProblemInstance(B=constraints, X=get_variables_from_constraints(constraints))

        sudokuX = sudoku.get_dataset()
        jsudokuX = jsudoku.get_dataset()
        nurse_rostX = nurse_rost.get_dataset()
        exam_ttX = exam_tt.get_dataset()

        # concat X
        X = [sudokuX, jsudokuX, nurse_rostX, exam_ttX]
        # concat Y
        Y = [sudokuY, jsudokuY, nurse_rostY, exam_ttY]

        combined_exp(X, Y, benchmark_names)

if __name__ == "__main__":
    main()

