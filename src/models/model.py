import os.path

from src.utils.memoization.memoization import p_load

from src.utils.utils import get_variables_from_constraints, get_var_name, get_var_dims, get_divisors


class Variables:
    def __init__(self, variables):
        self.variables = variables
        self._var_names = None
        self._var_names_set = None
        self._var_dims = None
        self._var_dims_length = None
        self._var_dims_divisors = None

    @property
    def var_names(self):
        if self._var_names is None:
            vars = self.variables
            self._var_names = [get_var_name(x) for x in vars]
        return self._var_names

    @property
    def var_names_set(self):
        if self._var_names_set is None:
            self._var_names_set = list(set(self.var_names))
        return self._var_names_set

    @property
    def var_dims(self):
        if self._var_dims is None:
            vars = self.variables
            names = self.var_names
            names_set = self.var_names_set
            self._var_dims = [[get_var_dims(vars[i]) for i in range(len(vars)) if names[i] == n] for n in names_set]
        return self._var_dims

    @property
    def var_dims_length(self):
        if self._var_dims_length is None:
            dims = self.var_dims
            self._var_dims_length = [[max([dims[i][j][d] for j in range(len(dims[i]))]) + 1 for d in range(len(dims[i][0]))]
                                     for i in range(len(dims))]
        return self._var_dims_length

    @property
    def var_dims_divisors(self):
        if self._var_dims_divisors is None:
            dims_length = self.var_dims_length
            names = self.var_names_set
            self._var_dims_divisors = {names[i]: [get_divisors(dims_length[i][j]) for j in range(len(dims_length[i]))]
                                       for i in range(len(dims_length))}
        return self._var_dims_divisors

class Model:
    def __init__(self, name: str, family: str, data_dir: str):
        self.name = name
        self.family = family
        self.data_dir = data_dir
        self.data_path = os.path.join(self.data_dir, self.family, self.name)
        self._gamma = None
        self._bias = None
        self._bias_map = None
        self._ground_truth = None
        self._variables = None

    @property
    def gamma(self):
        if self._gamma is None:
            # TODO retrieve it from gamma at creation
            # self._gamma = p_load(os.path.join(self.data_path, "gamma.pickle"))
            bias = self.bias
            self._gamma = list(set([c.get_relation() for c in bias]))
        return self._gamma

    @property
    def bias(self):
        if self._bias is None:
            self._bias = p_load(os.path.join(self.data_path, "bias.pickle"))
        return self._bias

    @property
    def bias_map(self):
        if self._bias_map is None:
            self._bias_map = {}
            b = self.bias
            for i in range(len(b)):
                self._bias_map[str(b[i])] = i
        return self._bias_map

    @property
    def ground_truth(self):
        if self._ground_truth is None:
            self._ground_truth = p_load(os.path.join(self.data_path, "ground_truth.pickle"))
        return self._ground_truth

    @property
    def variables(self):
        if self._variables is None:
            bias = self.bias
            self._variables = Variables(get_variables_from_constraints(bias))
        return self._variables


    def gamma_size(self):
        return len(self.gamma)

    def bias_size(self):
        return len(self.bias)

    def category_count(self):
        b = self.ground_truth
        p = sum(b)
        n = len(b) - p
        return p, n

    def get_stats(self):
        d = {"name": self.name,
             "family": self.family,
             "path": self.data_dir,
             "gamma": self.gamma,
             "gama_size": self.gamma_size(),
             "bias_size": self.bias_size(),
             "positif/negatif": self.category_count()}
        return d


if __name__ == "__main__":
    m = Model("sudoku4", "sudoku", "../../target/data")
    print(m.gamma)
    print(m.bias)
    print(m.bias_map)
    print(m.ground_truth)
    print(m.variables)
    print(m.variables.var_names)
    print(m.variables.var_names_set)
    print(m.variables.var_dims)
    print(m.variables.var_dims_length)
    print(m.variables.var_dims_divisors)
    print(m.get_stats())
