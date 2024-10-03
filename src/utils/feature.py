from statistics import mean
from typing import List

from cpmpy import intvar, AllDifferent

from src.utils.utils import *


class Feature:
    def __init__(self, cst):
        self.cst = cst
        # relation of the constraint
        self.relation = self.cst.get_relation()
        # scope (i.e, list of var) of the constraint
        self.scope = get_scope(self.cst)
        # arity (i.e, number of var) of the constraint
        self.arity = len(self.scope)
        # constants used in the constraint
        self._constants = get_constant(self.cst)
        # do the constraint have constants
        self.has_constant = len(self._constants) > 0
        # names of each of the variables in the scope
        self.scope_name = [get_var_name(s) for s in self.scope]
        # is the name shared by all the variable (true/false)
        self.vars_name_is_shared = all([sn == self.scope_name[0] for sn in self.scope_name])
        # dimensions for each of the variables in the scope
        self.vars_dims = [get_var_dims(var) for var in self.scope]
        # max dimension of the variables
        self.vars_dims_max = max([len(d) for d in self.vars_dims])
        # for each dimension, gather for each variable having this dimension the value of this dimension
        self.dims = [[self.vars_dims[i][d] for i in range(self.arity) if len(self.vars_dims[i]) > d] for d in
                     range(self.vars_dims_max)]
        # for each dimension, is the index for the dim the same for all var
        self._dims_is_shared = [all([d == dim[0] for d in dim]) for dim in self.dims]
        # for each dimension, the maximum index for the dim among the var
        self._dims_max = [max(dim) for dim in self.dims]
        # for each dimension, the minimum index for the dim among the var
        self._dims_min = [min(dim) for dim in self.dims]
        # for each dimension, the mean index for the dim among the var
        self._dims_mean = [mean(dim) for dim in self.dims]
        # for each dimension, the mean difference index for the dim among the var
        self._dims_mean_diff = [average_difference(dim) for dim in self.dims]
        # keep the block stats once they are computed
        self._dims_block = [{} for _ in self.dims]

    # TODO for the moment: do not deal with relation that have variables with different numbers of dimentions yet.
    # TODO also do not deal well with differents variables that have differents divisors
    def _dims_block_compute(self, id_dim: int, block_divider, divisor_id: int):
        # for i in range(self.arity):
        #     print(block_divider[self.scope_name[i]][id_dim])
        #     if len(block_divider[self.scope_name[i]][id_dim]) > divisor_id:
        #         print(self.dims[id_dim][i] // block_divider[self.scope_name[i]][id_dim][divisor_id])
        #     else:
        #         print(-1)
        this_dim_block_values = [self.dims[id_dim][i] // block_divider[self.scope_name[i]][id_dim][divisor_id]
                                 if len(block_divider[self.scope_name[i]][id_dim]) > divisor_id else -1
                                 for i in range(self.arity)]
        # print(this_dim_block_values)
        self._dims_block[id_dim][divisor_id] = {"block_values": this_dim_block_values,
                                                   "is_shared": all(
                                                       [d == this_dim_block_values[0] for d in this_dim_block_values]),
                                                   "min": min(this_dim_block_values),
                                                   "max": max(this_dim_block_values),
                                                   "mean": mean(this_dim_block_values),
                                                   "mean_diff": average_difference(this_dim_block_values)}

    def _safe_select_block(self, id_dim: int, block_divider, divisor_id, default, tag):
        if id_dim >= self.vars_dims_max:
            return default
        else:
            this_dim_block = self._dims_block[id_dim]
            if divisor_id not in this_dim_block:
                self._dims_block_compute(id_dim, block_divider, divisor_id)
            return this_dim_block[divisor_id][tag]

    def dims_block_is_shared(self, id_dim: int, block_divider, divisor_id):
        return self._safe_select_block(id_dim, block_divider, divisor_id, True, "is_shared")

    def dims_block_min(self, id_dim: int, block_divider, divisor_id):
        return self._safe_select_block(id_dim, block_divider, divisor_id, -1, "min")

    def dims_block_max(self, id_dim: int, block_divider, divisor_id):
        return self._safe_select_block(id_dim, block_divider, divisor_id, -1, "max")

    def dims_block_mean(self, id_dim: int, block_divider, divisor_id):
        return self._safe_select_block(id_dim, block_divider, divisor_id, -1, "mean")

    def dims_block_mean_diff(self, id_dim: int, block_divider, divisor_id):
        return self._safe_select_block(id_dim, block_divider, divisor_id, 0.0, "mean_diff")

    def _safe_select(self, id_dim: int, array, default):
        if id_dim >= self.vars_dims_max:
            return default
        else:
            return array[id_dim]

    def dims_is_shared(self, id_dim: int):
        # give for dimension id_dim if the index is the same for all var
        # default value true if asked for a dimension that none of the var in the scope have
        return self._safe_select(id_dim, self._dims_is_shared, True)

    def dims_max(self, id_dim: int):
        # give for dimension id_dim the maximum index for the dim among the var
        # default value 0 if asked for a dimension that none of the var in the scope have
        return self._safe_select(id_dim, self._dims_max, 0)

    def dims_min(self, id_dim):
        # give for dimension id_dim the minimum index for the dim among the var
        # default value 0 if asked for a dimension that none of the var in the scope have
        return self._safe_select(id_dim, self._dims_min, 0)

    def dims_mean(self, id_dim: int):
        # give for dimension id_dim the mean index for the dim among the var
        # default value 0.0 if asked for a dimension that none of the var in the scope have
        return self._safe_select(id_dim, self._dims_mean, 0.0)

    def dims_mean_diff(self, id_dim: int):
        # give for dimension id_dim the mean difference index for the dim among the var
        # default value 0.0 if asked for a dimension that none of the var in the scope have
        return self._safe_select(id_dim, self._dims_mean_diff, 0.0)

    def constants(self, i: int):
        # return constant i used in the constraint
        # default value 0 if asked for an non-existing constrant
        if i >= len(self._constants):
            return 0
        else:
            return self._constants[i]

    def get_gamma_id(self, gamma):
        # return the id of the relation in the gamma
        return gamma.id_relation(self.relation)

    def get_gamma_one_hot(self, gamma: List[str]):
        # return a one-hot encoding of the relation within the gamma
        i = self.get_gamma_id(gamma)
        return [1 if i == j else 0 for j in range(len(gamma))]

# OLD CODE HERE
# def get_con_features(cst):
#     # Returns the features associated with constraint cst
#     features = []
#     scope = get_scope(cst)
#     arity = len(scope)
#
#     var_names = [self.var_names.index(get_var_name(scope[i])) for i in range(arity)]
#     var_name_same = all([var_name == var_names[0] for var_name in var_names])
#     features.append(var_name_same)
#
#     # Global var dimension properties
#     vars_dims = [get_var_dims(var) for var in scope]
#     dim = []
#
#     # TODO add the right amount of Trues in features, this needs to be fixed
#     for j in range(max_dimensions):
#         dim.append([vars_dims[i][j] for i in range(len(vars_dims)) if len(vars_dims[i]) > j])
#
#         dimj_has = len(dim[j]) > 0
#         features.append(dimj_has)
#         if not dimj_has:
#
#             # dummy dimension features
#             features.append(True)
#             for i in range(3): features.append(0)
#             features.append(0.0)
#
#             # dummy latent dimension features
#             for l in range(max_blocks):
#                 features.append(True)
#                 for i in range(3): features.append(0)
#                 features.append(0.0)
#             continue
#
#         dimj_same = all([dim_temp == dim[j][0] for dim_temp in dim[j]])
#         features.append(dimj_same)
#         dimj_max = max(dim[j])
#         features.append(dimj_max)
#         dimj_min = min(dim[j])
#         features.append(dimj_min)
#         dimj_avg = mean(dim[j])
#         features.append(dimj_avg)
#         dimj_diff = average_difference(dim[j])
#         features.append(dimj_diff)
#
#         # for each of the dimension,
#         # for each block allowed, compute for each var, the block id they belong to if there would be a grid with that size of block
#         vars_block = [[dim[j][var] // self.dim_divisors[var_names[var]][j][divisor]
#                        if len(self.dim_divisors[var_names[var]][j]) > divisor else -1 for var in range(len(scope))]
#                       for divisor in range(max_blocks)]
#
#         # divisors features --- per var_name, per dimension (up to max dimensions), per divisor (up to max divisors)
#         # block same, max, min, average, spread
#         # for each variable name (type of variable)
#
#         # for var_name in range(len(self.var_names)):
#         # for each dimension of that type of variable
#         #            vars_block = [[dim[j][var] // divisor for var in range(len(scope))
#         #                           if self.var_names[var_name] == get_var_name(scope[var])]
#         #                          for divisor in self.dim_divisors[var_name][j]]
#
#         for l in range(max_blocks):
#             if self.debug_mode:
#                 print(vars_block[l])
#
#             block_same = all([vars_block[l][var] == vars_block[l][0] for var in range(len(vars_block[l]))])
#             features.append(block_same)
#             block_max = max(vars_block[l])
#             features.append(block_max)
#             block_min = min(vars_block[l])
#             features.append(block_min)
#             block_avg = mean(vars_block[l])
#             features.append(block_avg)
#             block_diff = average_difference(vars_block[l])
#             features.append(block_diff)
#
#     con_in_gamma = -1
#     con_relation = cst.get_relation()
#     for i in range(len(self.gamma)):
#         if self.gamma[i] == con_relation:
#             con_in_gamma = i
#             break
#
#     if con_in_gamma == -1:
#         raise Exception(
#             f"Check why constraint relation was not found in relations: constraint {cst}, relation: {con_relation}")
#     features.append(con_in_gamma)
#     features.append(arity)
#
#     num = get_constant(cst)
#     has_const = len(num) > 0
#     features.append(has_const)
#
#     if has_const:
#         features.append(int(num[0]))
#     else:
#         features.append(0)
#
#     return features


if __name__ == "__main__":
    grid = intvar(0, 9, shape=(9, 9), name='grid')
    cst = AllDifferent(grid[0])
    feat = Feature(cst)
    print(f"Relation: {feat.relation}")
    print(f"Scope: {feat.scope}")
    print(f"Arity: {feat.arity}")
    print(f"Constants: {feat._constants}")
    print(f"Constant at id 0: {feat.constants(0)}")
    print(f"Constant at id 1: {feat.constants(1)}")
    print(f"Has constant: {feat.has_constant}")
    print(f"Names of vars: {feat.scope_name}")
    print(f"Shared names: {feat.vars_name_is_shared}")
    print(f"Var dims: {feat.vars_dims}")
    print(f"Var dims max: {feat.vars_dims_max}")
    print(f"Dims: {feat.dims}")
    for i in range(3):
        print(f"=====Dim id: {i}=====")
        print(f"Dim is shared: {feat.dims_is_shared(i)}")
        print(f"Dim min: {feat.dims_min(i)}")
        print(f"Dim max: {feat.dims_max(i)}")
        print(f"Dim mean: {feat.dims_mean(i)}")
        print(f"Dim diff_mean: {feat.dims_mean_diff(i)}")

        block = {"grid": [[3], [3]]}

        for id_bloc in range(2):
            print(f"-----Block id: {id_bloc}-----")
            print(f"Block is shared: {feat.dims_block_is_shared(i,block,id_bloc)}")
            print(f"Block min: {feat.dims_block_min(i,block,id_bloc)}")
            print(f"Block max: {feat.dims_block_max(i,block,id_bloc)}")
            print(f"Block mean: {feat.dims_block_mean(i,block,id_bloc)}")
            print(f"Block diff_mean: {feat.dims_block_mean_diff(i,block,id_bloc)}")



