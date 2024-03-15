import re
from cpmpy.transformations.get_variables import get_variables

def get_var_name(var):
    name = re.findall("\[\d+[,\d+]*\]", var.name)
    name = var.name.replace(name[0], '')
    return name


def get_var_ndims(var):
    dims = re.findall("\[\d+[,\d+]*\]", var.name)
    dims_str = "".join(dims)
    ndims = len(re.split(",", dims_str))
    return ndims


def get_var_dims(var):
    dims = re.findall("\[\d+[,\d+]*\]", var.name)
    dims_str = "".join(dims)
    dims = re.split("[\[\]]", dims_str)[1]
    dims = [int(dim) for dim in re.split(",", dims)]
    return dims


def get_relation(c, gamma):
    scope = get_variables(c)

    for i in range(len(gamma)):
        relation = gamma[i]

        if relation.count("var") != len(scope):
            continue

        constraint = relation.replace("var1", "scope[0]")
        for j in range(1, len(scope)):
            constraint = constraint.replace("var" + str(j + 1), "scope[" + str(j) + "]")

        constraint = eval(constraint)

        if hash(constraint) == hash(c):
            return i

    return len(gamma)