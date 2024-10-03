from math import sqrt
from cpmpy import *
from cpmpy.transformations.normalize import toplevel_list


def construct_sudoku(grid_size):
    blocks = sqrt(grid_size)
    if blocks.is_integer():
        blocks = int(blocks)
    else:
        raise Exception(f"The grid size of sudoku must have a perfect square, {grid_size} square is {blocks}, "
                        f"which is not integer")

    return construct_rsudoku(blocks, blocks)


def construct_rsudoku(m, n):
    grid_size = m * n

    # Variables
    grid = intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")

    model = Model()

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, grid_size, m):
        for j in range(0, grid_size, n):
            model += AllDifferent(grid[i:i + m, j:j + n]).decompose()  # python's indexing

    C_T = set(toplevel_list(model.constraints))

    if not model.solve():
        print("no solution")
        raise Exception("The problem has no solution")

    return grid, C_T


def construct_jsudoku():
    # Variables
    grid = intvar(1, 9, shape=(9, 9), name="grid")

    model = Model()

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # the 9 blocks of squares in the specific instance of jsudoku
    blocks = [
        [grid[0, 0], grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 1], grid[1, 2], grid[2, 0], grid[2, 1], grid[2, 2]],
        [grid[0, 3], grid[0, 4], grid[0, 5], grid[0, 6], grid[1, 3], grid[1, 4], grid[1, 5], grid[1, 6], grid[2, 4]],
        [grid[0, 7], grid[0, 8], grid[1, 7], grid[1, 8], grid[2, 8], grid[3, 8], grid[4, 8], grid[5, 7], grid[5, 8]],
        [grid[4, 1], grid[4, 2], grid[4, 3], grid[5, 1], grid[5, 2], grid[5, 3], grid[6, 1], grid[6, 2], grid[6, 3]],
        [grid[2, 3], grid[3, 2], grid[3, 3], grid[3, 4], grid[4, 4], grid[5, 4], grid[5, 5], grid[5, 6], grid[6, 5]],
        [grid[2, 5], grid[2, 6], grid[2, 7], grid[3, 5], grid[3, 6], grid[3, 7], grid[4, 5], grid[4, 6], grid[4, 7]],
        [grid[3, 0], grid[3, 1], grid[4, 0], grid[5, 0], grid[6, 0], grid[7, 0], grid[7, 1], grid[8, 0], grid[8, 1]],
        [grid[6, 4], grid[7, 2], grid[7, 3], grid[7, 4], grid[7, 5], grid[8, 2], grid[8, 3], grid[8, 4], grid[8, 5]],
        [grid[6, 6], grid[6, 7], grid[6, 8], grid[7, 6], grid[7, 7], grid[7, 8], grid[8, 6], grid[8, 7], grid[8, 8]]
    ]

    # Constraints on blocks
    for i in range(0, 9):
        model += AllDifferent(blocks[i][:]).decompose()  # python's indexing

    C_T = set(toplevel_list(model.constraints))

    if not model.solve():
        print("no solution")
        raise Exception("The problem has no solution")

    return grid, C_T
