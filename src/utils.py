import re
from cpmpy.transformations.get_variables import get_variables
import math
import time
from cpmpy.expressions.core import Expression, Comparison, Operator
from cpmpy.expressions.variables import _NumVarImpl, _BoolVarImpl, _IntVarImpl, NDVarArray
from sklearn.utils import class_weight
import numpy as np

import cpmpy
import re


def get_scope(constraint):
    # this code is much more dangerous/too few cases then get_variables()
    if isinstance(constraint, cpmpy.expressions.variables._IntVarImpl):
        return [constraint]
    elif isinstance(constraint, cpmpy.expressions.core.Expression):
        all_variables = []
        for argument in constraint.args:
            if isinstance(argument, cpmpy.expressions.variables._IntVarImpl):
                # non-recursive shortcut
                all_variables.append(argument)
            else:
                all_variables.extend(get_scope(argument))
        return all_variables
    else:
        return []


def get_arity(constraint):
    return len(get_scope(constraint))

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


def get_divisors(n):
    divisors = list()
    for i in range(2, int(n / 2) + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors

def get_variables_from_constraints(constraints):
    def get_variables(expr):
        if isinstance(expr, _IntVarImpl):
            return [expr]
        elif isinstance(expr, _BoolVarImpl):
            return [expr]
        elif isinstance(expr, np.bool_):
            return []
        elif isinstance(expr, np.int_) or isinstance(expr, int):
            return []
        else:
            # Recursively find variables in all arguments of the expression
            return [var for argument in expr.args for var in get_variables(argument)]

    # Create set to hold unique variables
    variable_set = set()
    for constraint in constraints:
        variable_set.update(get_variables(constraint))

    extract_nums = lambda s: list(map(int, s.name[s.name.index("[") + 1:s.name.index("]")].split(',')))

    variable_list = sorted(variable_set, key=extract_nums)
    return variable_list

## to attach to CPMpy expressions!!
def expr_get_relation(self):
    # flatten and replace
    flatargs = []
    for arg in self.args:
        if isinstance(arg, np.ndarray):
            for a in arg.flat:
                if isinstance(a, _NumVarImpl):
                    flatargs.append("var")
                elif isinstance(a, Expression):
                    flatargs.append(a.get_relation())
                else:
                    flatargs.append("const")
        else:
            if isinstance(arg, _NumVarImpl):
                flatargs.append("var")
            elif isinstance(arg, Expression):
                flatargs.append(arg.get_relation())
            else:
                flatargs.append("const")

    if len(flatargs) > 1:
        flatargs = tuple(flatargs)
    else:
        flatargs = flatargs[0] if not isinstance(flatargs[0], _NumVarImpl) else tuple(flatargs)

    return (self.name, flatargs)
Expression.get_relation = expr_get_relation  # attach to CPMpy expressions

def comp_get_relation(self):
    flatargs = []

    for arg in self.args:
        if isinstance(arg, _NumVarImpl):
            flatargs.append("var")
        elif isinstance(arg, Expression):
            flatargs.append(arg.get_relation())
        else:
            flatargs.append("const")

    return (self.name, (flatargs[0], flatargs[1]))
Comparison.get_relation = comp_get_relation  # attach to CPMpy comparisons

def get_constant(constraint):
    # this code is much more dangerous/too few cases then get_variables()
    if isinstance(constraint, cpmpy.expressions.variables._IntVarImpl):
        return []
    elif isinstance(constraint, cpmpy.expressions.core.Expression):
        constants = []
        for argument in constraint.args:
            if not isinstance(argument, cpmpy.expressions.variables._IntVarImpl):
                constants.extend(get_constant(argument))
        return constants
    else:
        return [constraint]

def average_difference(values):
    if len(values) < 2:
        return 0
    differences = [values[i+1] - values[i] for i in range(len(values)-1)]
    return sum(differences) / len(differences)


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