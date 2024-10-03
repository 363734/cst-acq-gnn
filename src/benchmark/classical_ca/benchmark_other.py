from cpmpy import *
from cpmpy.transformations.normalize import toplevel_list

def construct_golomb(marks):

    # Variables
    grid = intvar(1, marks*5, shape=(1, marks), name="grid")

    model = Model()

    for i in range(marks-1):
        for j in range(i + 1, marks):
            for x in range(j + 1, marks-1):
                for y in range(x + 1, marks):
                    if (y != i and x != j and x != i and y != j):
                        model += abs(grid[0, i] - grid[0, j]) != abs(grid[0, x] - grid[0, y])

    C_T = list(model.constraints)

    print(len(C_T))

    if not model.solve():
        print("no solution")
        raise Exception("The problem has no solution")

    return grid, C_T

def construct_murder_problem():
    # Variables
    grid = intvar(1, 5, shape=(4, 5), name="grid")

    C_T = list()

    # Constraints on rows and columns
    model = Model([AllDifferent(row).decompose() for row in grid])

    # Additional constraints of the murder problem
    C_T += [grid[0, 1] == grid[1, 2]]
    C_T += [grid[0, 2] != grid[1, 4]]
    C_T += [grid[3, 2] != grid[1, 4]]
    C_T += [grid[0, 2] != grid[1, 0]]
    C_T += [grid[0, 2] != grid[3, 4]]
    C_T += [grid[3, 4] == grid[1, 3]]
    C_T += [grid[1, 1] == grid[2, 1]]
    C_T += [grid[2, 3] == grid[0, 3]]
    C_T += [grid[2, 0] == grid[3, 3]]
    C_T += [grid[0, 0] != grid[2, 4]]
    C_T += [grid[0, 0] != grid[1, 4]]
    C_T += [grid[0, 0] == grid[3, 0]]

    model += C_T

    for row in grid:
        C_T += list(AllDifferent(row).decompose())

    C_T = toplevel_list(C_T)

    if not model.solve():
        print("no solution")
        raise Exception("The problem has no solution")

    return grid, C_T