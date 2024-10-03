

import random
from cpmpy import Model

from src.utils.utils import get_scope


def remove_scope_from_bias(B, scope):  # remove constraint network C from B

    prev_B_length = len(B)
    B = [c for c in B if not (set(get_scope(c)) == set(scope))]

    if len(B) == prev_B_length:
        print(B)
        print(scope)
        raise Exception("Removing constraints from Bias did not result in reducing its size")

    return B

def construct_random_problem(Ncons, B):

    model = Model()

    for con in range(Ncons):

        found = False

        while not found:
            r = random.randint(0,len(B)-1)

            model2 = model.copy()

            model2 += B[r]

            if model2.solve():
                model += B[r]
                # print(f"Adding {B[r]} to the model ")
                B = remove_scope_from_bias(B, get_scope(B[r]))
                found = True
            else:
                # print(f"skipping {B[r]}, makes it UNSAT")
                B.pop(r)

    C_T = list(model.constraints)

    if not model.solve():
        print("no solution")
        raise Exception("The problem has no solution")

    return C_T
