import argparse
import pickle
import random
from itertools import combinations
from math import sqrt

import cpmpy as cp
import numpy as np
from cpmpy import *
from cpmpy.transformations.normalize import toplevel_list


def all_combinations(args, combos):
    """ returns all pairwise combinations of elements in args
    """
    return list(combinations(args, combos))

# Benchmark construction
def construct_sudoku(grid_size):


    blocks = sqrt(grid_size)
    if blocks.is_integer():
        blocks = int(blocks)
    else:
        raise Exception(f"The grid size of sudoku must have a perfect square, {grid_size} square is {blocks}, "
                        f"which is not integer")

    # Variables
    grid = intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")

    model = Model()

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, grid_size, blocks):
        for j in range(0, grid_size, blocks):
            model += AllDifferent(grid[i:i + blocks, j:j + blocks]).decompose()  # python's indexing

    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    C_T = set(toplevel_list(C))

    return grid, C_T


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

    if model.solve():
        print(roster_matrix.value())
    else:
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

    if model.solve():
        print("solution exists")
    else:
        print("no solution")
        raise Exception("The problem has no solution")

    C = list(model.constraints)

    C_T = set(toplevel_list(C))

    return roster_matrix, C_T


def construct_examtt_simple(NSemesters=9, courses_per_semester=6, rooms=3, timeslots_per_day=3, days_for_exams=14):
    total_courses = NSemesters * courses_per_semester
    slots_per_day = rooms * timeslots_per_day
    total_slots = slots_per_day * days_for_exams

    # Variables
    courses = intvar(1, total_slots, shape=(NSemesters, courses_per_semester), name="courses")
    all_courses = set(courses.flatten())

    model = Model()

    model += AllDifferent(all_courses).decompose()

    # Constraints on courses of same semester
    for row in courses:
        model += AllDifferent(row // slots_per_day).decompose()

    C = list(model.constraints)

    print(model)

    if model.solve():
        print(courses.value())
    else:
        print("no solution")

    C2 = [c for c in model.constraints if not isinstance(c, list)]

    C_T = set(toplevel_list(C))

    print("new model: ----------------------------------\n", C_T)
    print(C2)

    return courses, C_T


def construct_examtt_advanced(NSemesters=9, courses_per_semester=6, rooms=3, timeslots_per_day=3, days_for_exams=14,
                              NProfessors=30):
    total_courses = NSemesters * courses_per_semester
    slots_per_day = rooms * timeslots_per_day
    total_slots = slots_per_day * days_for_exams

    # Variables
    courses = intvar(1, total_slots, shape=(NSemesters, courses_per_semester), name="courses")
    all_courses = set(courses.flatten())

    model = Model()

    model += AllDifferent(all_courses).decompose()

    # Constraints on courses of same semester
    for row in courses:
        model += AllDifferent(row // slots_per_day).decompose()

    C = list(model.constraints)

    # Constraints of Professors - instance specific -------------------------------

    # first define the courses each professor is assigned to
    # this can be given, or random generated!!

    assert NProfessors <= total_courses
    courses_per_professor = total_courses // NProfessors
    remaining_courses = total_courses % NProfessors  # will assign 1 per professor to some professors

    # probabilities of additional constraints to be introduced
    pcon_close = 0.3  # probability of professor constraint to have his courses on close days
    # (e.g. because he lives in another city and has to come for the exams)

    # pcon_diff = 0.2  # probability of professor constraint to not have his exams in a certain day

    Prof_courses = list()
    for i in range(NProfessors):

        prof_courses = list()

        for j in range(courses_per_professor):  # assign the calculated number of courses to the professors
            prof_courses.append(all_courses.pop())  # it is a set, so it will pop a random one (there is no order)

        if i < remaining_courses:  # # assign the remaining courses to the professors
            prof_courses.append(all_courses.pop())  # it is a set, so it will pop a random one (there is no order)

        Prof_courses.append(prof_courses)

        if len(prof_courses) > 1:

            r = random.uniform(0, 1)

            if r < pcon_close:
                for c1, c2 in all_combinations(prof_courses, 2):
                    model += abs(c1 - c2) // slots_per_day <= 2  # all her courses in 2 days

    print(model)

    if model.solve():
        print(courses.value())
    else:
        print("no solution")

    C_T = set(toplevel_list(C))

    return courses, C_T


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

    print(len(C_T))

    return grid, C_T


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

    return grid, C_T


def construct_job_shop_scheduling_problem(n_jobs, machines, horizon, seed=0):
    random.seed(seed)
    max_time = horizon // n_jobs

    duration = [[0] * machines for i in range(0, n_jobs)]
    for i in range(0, n_jobs):
        for j in range(0, machines):
            duration[i][j] = random.randint(1, max_time)

    task_to_mach = [list(range(0, machines)) for i in range(0, n_jobs)]

    for i in range(0, n_jobs):
        random.shuffle(task_to_mach[i])

    precedence = [[(i, j) for j in task_to_mach[i]] for i in range(0, n_jobs)]

    # convert to numpy
    task_to_mach = np.array(task_to_mach)
    duration = np.array(duration)
    precedence = np.array(precedence)

    machines = set(task_to_mach.flatten().tolist())

    model = cp.Model()

    # decision variables
    start = cp.intvar(1, horizon, shape=task_to_mach.shape, name="start")
    end = cp.intvar(1, horizon, shape=task_to_mach.shape, name="end")

    grid = cp.cpm_array(np.expand_dims(np.concatenate([start.flatten(), end.flatten()]), 0))

    # precedence constraints
    for chain in precedence:
        for (j1, t1), (j2, t2) in zip(chain[:-1], chain[1:]):
            model += end[j1, t1] <= start[j2, t2]

    # duration constraints
    model += (start + duration == end)

    # non_overlap constraints per machine
    for m in machines:
        tasks_on_mach = np.where(task_to_mach == m)
        for (j1, t1), (j2, t2) in all_combinations(zip(*tasks_on_mach),2):
            m += (end[j1, t1] <= start[j2, t2]) | (end[j2, t2] <= start[j1, t1])

    C = list(model.constraints)

    temp = []
    for c in C:
        if isinstance(c, cp.expressions.core.Comparison):
            temp.append(c)
        elif isinstance(c, cp.expressions.variables.NDVarArray):
            _c = c.flatten()
            for __c in _c:
                temp.append(__c)
    # [temp.append(c) for c in C]
    C_T = set(temp)

    max_duration = max(duration)
    return grid, C_T, max_duration


def construct_bias(X, gamma):
    all_cons = []

    X = list(X)

    for relation in gamma:

        if relation.count("var") == 2:

            for v1, v2 in all_combinations(X,2):
                constraint = relation.replace("var1", "v1")
                constraint = constraint.replace("var2", "v2")
                constraint = eval(constraint)

                all_cons.append(constraint)

        elif relation.count("var") == 4:

            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    for x in range(j + 1, len(X) - 1):
                        for y in range(x + 1, len(X)):
                            if (y != i and x != j and x != i and y != j):
                                #            for v1, v2 in all_pairs(X):
                                #                for v3, v4 in all_pairs(X):
                                constraint = relation.replace("var1", "X[i]")
                                constraint = constraint.replace("var2", "X[j]")
                                constraint = constraint.replace("var3", "X[x]")
                                constraint = constraint.replace("var4", "X[y]")
                                constraint = eval(constraint)

                                all_cons.append(constraint)

    return all_cons

def construct_benchmark(args):
    if args.benchmark == "sudoku":
        grid, C_T = construct_sudoku(args.grid_size)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "jsudoku":
        grid, C_T = construct_jsudoku()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "golomb":
        grid, C_T = construct_golomb(args.marks)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2",
                 "abs(var1 - var2) != abs(var3 - var4)"]
    #            "abs(var1 - var2) == abs(var3 - var4)"]

    elif args.benchmark == "murder":
        grid, C_T = construct_murder_problem()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "job_shop_scheduling":
        grid, C_T, max_duration = construct_job_shop_scheduling_problem(args.num_jobs, args.num_machines,
                                                                        args.horizon, args.seed)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"var1 + {i} == var2" for i in range(1, max_duration + 1)] + \
                [f"var2 + {i} == var1" for i in range(1, max_duration + 1)]

    elif args.benchmark == "exam_timetabling" or args.benchmark == "exam_timetabling_simple":
        slots_per_day = args.num_rooms * args.num_timeslots_per_day

        grid, C_T = construct_examtt_simple(args.num_semesters, args.num_courses_per_semester, args.num_rooms,
                                            args.num_timeslots_per_day, args.num_days_for_exams)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"(var1 // {slots_per_day}) != (var2 // {slots_per_day})",
                 f"(var1 // {slots_per_day}) == (var2 // {slots_per_day})"]

    elif args.benchmark == "exam_timetabling_adv" or args.benchmark == "exam_timetabling_advanced":
        slots_per_day = args.num_rooms * args.num_timeslots_per_day

        grid, C_T = construct_examtt_advanced(args.num_semesters, args.num_courses_per_semester, args.num_rooms,
                                              args.num_timeslots_per_day, args.num_days_for_exams, args.num_professors)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"abs(var1 - var2) // {slots_per_day} <= 2"] + \
                [f"(var1 // {slots_per_day}) != (var2 // {slots_per_day})"]
        # [f"var1 // {slots_per_day} != {d}" for d in range(num_days_for_exams)]
    elif args.benchmark == "nurse_rostering":

        grid, C_T = construct_nurse_rostering(args.num_nurses, args.num_shifts_per_day,
                                              args.num_days_for_schedule)

        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "nurse_rostering_adv" or args.benchmark == "nurse_rostering_advanced":

        grid, C_T = construct_nurse_rostering_advanced(args.num_nurses, args.num_shifts_per_day,
                                                       args.nurses_per_shift, args.num_days_for_schedule)

        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    else:
        raise NotImplementedError(f'benchmark {args.benchmark} not implemented yet')

    B = construct_bias(grid.flatten(), gamma)

    return args.benchmark, grid, C_T, B


def parse_args():
    parser = argparse.ArgumentParser()

    # Parsing benchmark
    parser.add_argument("-b", "--benchmark", type=str, required=True,
                        choices=["sudoku", "jsudoku", "golomb", "murder", "job_shop_scheduling",
                                 "exam_timetabling", "exam_timetabling_simple", "exam_timetabling_adv",
                                 "exam_timetabling_advanced", "nurse_rostering", "nurse_rostering_simple",
                                 "nurse_rostering_advanced", "nurse_rostering_adv"], help="The name of the benchmark to use")

    # Parsing specific to Sudoku benchmark
    parser.add_argument("-gs", "--grid-size", type=int, required=False,
                        help="Only relevant when the chosen benchmark is sudoku - the grid size")

    # Parsing specific to Golomb benchmark
    parser.add_argument("-m", "--marks", type=int, required=False,
                        help="Only relevant when the chosen benchmark is golomb - the amount of marks")

    # Parsing specific to job-shop scheduling benchmark
    parser.add_argument("-nj", "--num-jobs", type=int, required=False,
                        help="Only relevant when the chosen benchmark is job-shop scheduling - the number of jobs")
    parser.add_argument("-nm", "--num-machines", type=int, required=False,
                        help="Only relevant when the chosen benchmark is job-shop scheduling - the number of machines")
    parser.add_argument("-hor", "--horizon", type=int, required=False,
                        help="Only relevant when the chosen benchmark is job-shop scheduling - the horizon")
    parser.add_argument("-s", "--seed", type=int, required=False,
                        help="Only relevant when the chosen benchmark is job-shop scheduling - the seed")

    # Parsing specific to nurse rostering benchmark
    parser.add_argument("-nspd", "--num-shifts-per-day", type=int, required=False,
                        help="Only relevant when the chosen benchmark is nurse rostering - the number of shifts per day")
    parser.add_argument("-ndfs", "--num-days-for-schedule", type=int, required=False,
                        help="Only relevant when the chosen benchmark is nurse rostering - the number of days for the schedule")
    parser.add_argument("-nn", "--num-nurses", type=int, required=False,
                        help="Only relevant when the chosen benchmark is nurse rostering - the number of nurses")
    parser.add_argument("-nps", "--nurses-per-shift", type=int, required=False,
                        help="Only relevant when the chosen benchmark is nurse rostering (advanced) - "
                             "the number of nurses per shift")

    # Parsing specific to exam timetabling benchmark
    parser.add_argument("-ns", "--num-semesters", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of semesters")
    parser.add_argument("-ncps", "--num-courses-per-semester", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of courses per semester")
    parser.add_argument("-nr", "--num-rooms", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of rooms")
    parser.add_argument("-ntpd", "--num-timeslots-per-day", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of timeslots per day")
    parser.add_argument("-ndfe", "--num-days-for-exams", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of days for exams")
    parser.add_argument("-np", "--num-professors", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of professors")
    args = parser.parse_args()

    # Additional validity checks
    if args.benchmark == "sudoku" and \
            (args.grid_size is None):
        parser.error("When Sudoku is chosen as benchmark, the grid size must be specified with the parameter -gs")
    if args.benchmark == "golomb" and \
            (args.marks is None):
        parser.error("When Golomb is chosen as benchmark, the amount of marks must be specified with the parameter -m")
    if args.benchmark == "job_shop_scheduling" and \
            (args.num_jobs is None or args.num_machines is None or args.horizon is None or args.seed is None):
        parser.error("When job-shop-scheduling is chosen as benchmark, a number of jobs, a number of machines,"
                     "a horizon and a seed must be specified")
    if (args.benchmark == "exam_timetabling" or args.benchmark == "exam_timetabling_simple") and\
            (args.num_semesters is None or args.num_courses_per_semester is None or args.num_rooms is None or
             args.num_timeslots_per_day is None or args.num_days_for_exams is None):
        parser.error("When exam-timetabling is chosen as benchmark, a number of semesters, a number of courses per"
                     "semester, a number of rooms, a number of timeslots per day and a number of days for exams"
                     " must be specified")
    if (args.benchmark == "exam_timetabling_adv" or args.benchmark == "exam_timetabling_advanced") and\
            (args.num_semesters is None or args.num_courses_per_semester is None or args.num_rooms is None or
             args.num_timeslots_per_day is None or args.num_days_for_exams is None or args.num_professors is None):
        parser.error("When exam-timetabling is chosen as benchmark, a number of semesters, a number of courses per"
                     "semester, a number of rooms, a number of timeslots per day, a number of days for exams"
                     " and a number of professors must be specified")

    return args

if __name__ == "__main__":

    # Setup
    args = parse_args()

    benchmark_name, grid, C_T, B = construct_benchmark(args)

    print(f"benchmark name: {benchmark_name}, constriant network size: {len(C_T)}, bias size: {len(B)}")

    # Create dataset
    datasetC = []
    datasetCY = []

    setModel = set(C_T)

    for c in B:
        datasetC.append(c)
        if c in setModel:
            datasetCY.append(1)
        else:
            datasetCY.append(0)

    out_file_X = f"benchmarks/{args.benchmark}/{args.benchmark}_{len(C_T)}_C.pickle"
    out_file_Y = f"benchmarks/{args.benchmark}/{args.benchmark}_{len(C_T)}_CY.pickle"

    with open(out_file_X, 'wb') as handle:
        pickle.dump(datasetC, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_file_Y, 'wb') as handle:
        pickle.dump(datasetCY, handle, protocol=pickle.HIGHEST_PROTOCOL)