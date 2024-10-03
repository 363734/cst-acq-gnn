from cpmpy import *
from src.utils.utils import * # TODO keep to add the get_relation to the constraint of cpmpy


class Gamma:
    def __init__(self, str_gamma):
        self.str_gamma = str_gamma
        self.size = len(self.str_gamma)
        self.var_count = [s.count("var") for s in self.str_gamma]
        self.relations = [self._get_relation(i) for i in range(len(self.str_gamma))]

    def _get_relation(self, id):
        v = intvar(0, 1, shape=self.var_count[id])
        cst = self.substitute(id, v)
        return cst.get_relation()

    def id_relation(self, rel):
        for i in range(self.size):
            if rel == self.relations[i]:
                return i
        return self.size

    def substitute(self, id, vars):
        cst = self.str_gamma[id]
        for i in range(len(vars)):
            cst = cst.replace(f"var{i + 1}", f"vars[{i}]")
        return eval(cst)

    def generate_all(self, id, vars):
        from itertools import combinations
        return [self.substitute(id, c) for c in combinations(vars, self.var_count[id])]

    def construct_bias(self, vars):
        vars = list(vars.flatten())
        all_cons = []
        for id in range(self.size):
            all_cons.extend(self.generate_all(id, vars))
        return all_cons

    def __str__(self):
        return str(self.str_gamma)


COMPARE_GAMMA = Gamma(["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"])

if __name__ == "__main__":
    print(COMPARE_GAMMA.str_gamma)
    print(COMPARE_GAMMA.var_count)
    print(COMPARE_GAMMA.relations)

    xs = intvar(0, 1, shape=4)
    print(xs)
    print(COMPARE_GAMMA.construct_bias(xs))
