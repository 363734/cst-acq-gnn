import os

from cpmpy import *

from src.benchmark.classical_ca.benchmark_job_shop_scheduling import construct_job_shop_scheduling_problem
from src.benchmark.classical_ca.benchmark_nurse_rostering import construct_nurse_rostering, \
    construct_nurse_rostering_advanced
from src.benchmark.classical_ca.benchmark_other import construct_golomb, construct_murder_problem
from src.benchmark.classical_ca.benchmark_random import construct_random_problem
from src.benchmark.classical_ca.benchmark_sudoku import construct_sudoku, construct_jsudoku, construct_rsudoku
from src.benchmark.classical_ca.benchmark_time_tabling import construct_examtt_simple, construct_examtt_advanced
from src.benchmark.create_opt import parse_args
from src.models.gamma import COMPARE_GAMMA, Gamma
from src.utils.memoization.memoization import p_save, p_load
from src.utils.utils import get_variables_from_constraints


def construct_benchmark(args):
    if args.benchmark == "random":
        gamma = COMPARE_GAMMA
        grid = intvar(1, args.size_domain, shape=(1, args.number_vars), name="grid")
        B = gamma.construct_bias(grid)
        C_T = construct_random_problem(args.number_cst, B)
    else:
        if args.benchmark == "sudoku":
            grid, C_T = construct_sudoku(args.grid_size)
            gamma = COMPARE_GAMMA

        elif args.benchmark == "rsudoku":
            grid, C_T = construct_rsudoku(args.grid_size_dim1, args.grid_size_dim2)
            gamma = COMPARE_GAMMA

        elif args.benchmark == "jsudoku":
            grid, C_T = construct_jsudoku()
            gamma = COMPARE_GAMMA

        elif args.benchmark == "golomb":
            grid, C_T = construct_golomb(args.marks)
            gamma = Gamma(["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2",
                           "abs(var1 - var2) != abs(var3 - var4)"])
        #            "abs(var1 - var2) == abs(var3 - var4)"]

        elif args.benchmark == "murder":
            grid, C_T = construct_murder_problem()
            gamma = COMPARE_GAMMA

        elif args.benchmark == "job_shop_scheduling":
            grid, C_T, max_duration = construct_job_shop_scheduling_problem(args.num_jobs, args.num_machines,
                                                                            args.horizon, args.seed)
            gamma = Gamma(
                ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"var1 + {i} == var2" for i in range(1, max_duration + 1)] + \
                [f"var2 + {i} == var1" for i in range(1, max_duration + 1)])

        elif args.benchmark == "exam_timetabling" or args.benchmark == "exam_timetabling_simple":
            slots_per_day = args.num_rooms * args.num_timeslots_per_day

            grid, C_T = construct_examtt_simple(args.num_semesters, args.num_courses_per_semester, args.num_rooms,
                                                args.num_timeslots_per_day, args.num_days_for_exams)
            gamma = Gamma(
                ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"(var1 // {slots_per_day}) != (var2 // {slots_per_day})",
                 f"(var1 // {slots_per_day}) == (var2 // {slots_per_day})"])

        elif args.benchmark == "exam_timetabling_adv" or args.benchmark == "exam_timetabling_advanced":
            slots_per_day = args.num_rooms * args.num_timeslots_per_day

            grid, C_T = construct_examtt_advanced(args.num_semesters, args.num_courses_per_semester, args.num_rooms,
                                                  args.num_timeslots_per_day, args.num_days_for_exams,
                                                  args.num_professors)
            gamma = Gamma(
                ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"abs(var1 - var2) // {slots_per_day} <= 2"] + \
                [f"(var1 // {slots_per_day}) != (var2 // {slots_per_day})"])
            # [f"var1 // {slots_per_day} != {d}" for d in range(num_days_for_exams)]
        elif args.benchmark == "nurse_rostering":

            grid, C_T = construct_nurse_rostering(args.num_nurses, args.num_shifts_per_day,
                                                  args.num_days_for_schedule)

            gamma = COMPARE_GAMMA

        elif args.benchmark == "nurse_rostering_adv" or args.benchmark == "nurse_rostering_advanced":

            grid, C_T = construct_nurse_rostering_advanced(args.num_nurses, args.num_shifts_per_day,
                                                           args.nurses_per_shift, args.num_days_for_schedule)

            gamma = COMPARE_GAMMA
        elif args.benchamrk == "fromfile":  # TODO not tested yet but should work
            import cpmpy as cp
            model = p_load(args.model_from_file)
            C = list(model.constraints)
            temp = []
            for c in C:
                if isinstance(c, cp.expressions.core.Comparison):
                    temp.append(c)
                elif isinstance(c, cp.expressions.variables.NDVarArray):
                    temp.extend(c.flatten())
            C_T = set(temp)
            grid = []
            for c in C_T:
                grid.extend(get_variables_from_constraints(c))
            grid = set(grid)
            gamma = COMPARE_GAMMA
        else:
            raise NotImplementedError(f'benchmark {args.benchmark} not implemented yet')

        B = gamma.construct_bias(grid)

    return args.benchmark, grid, C_T, B, gamma


if __name__ == "__main__":
    # Setup
    args = parse_args()

    benchmark_name, grid, C_T, B, gamma = construct_benchmark(args)

    print(f"benchmark name: {benchmark_name}, constraint network size: {len(C_T)}, bias size: {len(B)}")

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

    out_file_X = f"{args.directory}/{args.benchmark}/{args.name}/bias.pickle"
    out_file_Y = f"{args.directory}/{args.benchmark}/{args.name}/ground_truth.pickle"
    out_file_G = f"{args.directory}/{args.benchmark}/{args.name}/gamma.pickle"

    os.makedirs(os.path.dirname(out_file_X), exist_ok=True)
    os.makedirs(os.path.dirname(out_file_Y), exist_ok=True)
    os.makedirs(os.path.dirname(out_file_G), exist_ok=True)

    p_save(out_file_X, datasetC)
    p_save(out_file_Y, datasetCY)
    p_save(out_file_G, gamma)
