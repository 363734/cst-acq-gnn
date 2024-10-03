from cpmpy import *
from cpmpy.transformations.normalize import toplevel_list


def construct_nurse_rostering(num_nurses, shifts_per_day, num_days):
    # Define the variables
    roster_matrix = intvar(1, num_nurses, shape=(num_days, shifts_per_day), name="shifts")
    print(roster_matrix)

    # Define the constraints
    model = Model()

    # Constraint: Each shift in a day must be assigned to a different nurse
    for day in range(num_days):
        model += AllDifferent(roster_matrix[day, :]).decompose()

    # Constraint: The last shift of a day cannot have the same nurse as the first shift of the next day
    for day in range(num_days - 1):
        model += (roster_matrix[day, shifts_per_day - 1] != roster_matrix[day + 1, 0])

    print(model)

    if not model.solve():
        print("no solution")
        raise Exception("The problem has no solution")

    C = list(model.constraints)

    C_T = set(toplevel_list(C))

    return roster_matrix, C_T


def construct_nurse_rostering_advanced(num_nurses, shifts_per_day, nurses_per_shift, num_days):
    # Define the variables
    roster_matrix = intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="shifts")

    # Define the constraints
    model = Model()

    # Constraint: Each shift in a day must be assigned to a different nurse
    for day in range(num_days):
        model += AllDifferent(roster_matrix[day, ...]).decompose()

    # Constraint: The last shift of a day cannot have the same nurse as the first shift of the next day
    for day in range(num_days - 1):
        model += AllDifferent(roster_matrix[day, shifts_per_day - 1], roster_matrix[day + 1, 0]).decompose()

    if not model.solve():
        print("no solution")
        raise Exception("The problem has no solution")

    C = list(model.constraints)

    C_T = set(toplevel_list(C))

    return roster_matrix, C_T
