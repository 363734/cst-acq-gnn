import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--directory", type=str, required=True, help="storage directory")
    parser.add_argument("-n", "--name", type=str, required=True, help="usual name of benchmark")

    # Parsing benchmark
    parser.add_argument("-b", "--benchmark", type=str, required=True,
                        choices=["fromfile", "sudoku", "rsudoku", "jsudoku", "golomb", "murder", "job_shop_scheduling",
                                 "exam_timetabling", "exam_timetabling_simple", "exam_timetabling_adv",
                                 "exam_timetabling_advanced", "nurse_rostering", "nurse_rostering_simple",
                                 "nurse_rostering_advanced", "nurse_rostering_adv", "random"],
                        help="The name of the benchmark to use")

    parser.add_argument("-mff", "--model-from-file", type=int, required=False,
                        help="Only relevant when the chosen benchmark is fromfile - the file storing the cpmpy model")

    # Parsing specific to Sudoku benchmark
    parser.add_argument("-gs", "--grid-size", type=int, required=False,
                        help="Only relevant when the chosen benchmark is sudoku - the grid size")

    # Parsing specific to Sudoku benchmark
    parser.add_argument("-dim1", "--grid-size-dim1", type=int, required=False,
                        help="Only relevant when the chosen benchmark is rsudoku - the grid size")
    parser.add_argument("-dim2", "--grid-size-dim2", type=int, required=False,
                        help="Only relevant when the chosen benchmark is rsudoku - the grid size")

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

    # Parsing specific to random benchmark
    parser.add_argument("-sd", "--size_domain", type=int, required=False,
                        help="Only relevant when the chosen benchmark is random - "
                             "size domain")
    parser.add_argument("-nv", "--number_vars", type=int, required=False,
                        help="Only relevant when the chosen benchmark is random - "
                             "number of variables")
    parser.add_argument("-nc", "--number_cst", type=int, required=False,
                        help="Only relevant when the chosen benchmark is random - "
                             "number of constraints")

    args = parser.parse_args()

    # Additional validity checks
    if args.benchmark == "fromfile" and (args.model_from_file is None):
        parser.error("When fromfile is used, a file to fetch the model from should be specified with -mff")

    if args.benchmark == "sudoku" and (args.grid_size is None):
        parser.error("When Sudoku is chosen as benchmark, the grid size must be specified with the parameter -gs")

    if args.benchmark == "rsudoku" and (args.grid_size_dim1 is None or args.grid_size_dim2 is None):
        parser.error(
            "When RSudoku is chosen as benchmark, the grid size must be specified with the parameter -dim1 and -dim2")

    if args.benchmark == "golomb" and (args.marks is None):
        parser.error("When Golomb is chosen as benchmark, the amount of marks must be specified with the parameter -m")

    if args.benchmark == "job_shop_scheduling" and \
            (args.num_jobs is None or args.num_machines is None or args.horizon is None or args.seed is None):
        parser.error("When job-shop-scheduling is chosen as benchmark, a number of jobs, a number of machines,"
                     "a horizon and a seed must be specified")

    if (args.benchmark == "exam_timetabling" or args.benchmark == "exam_timetabling_simple") and \
            (args.num_semesters is None or args.num_courses_per_semester is None or args.num_rooms is None or
             args.num_timeslots_per_day is None or args.num_days_for_exams is None):
        parser.error("When exam-timetabling is chosen as benchmark, a number of semesters, a number of courses per"
                     "semester, a number of rooms, a number of timeslots per day and a number of days for exams"
                     " must be specified")

    if (args.benchmark == "exam_timetabling_adv" or args.benchmark == "exam_timetabling_advanced") and \
            (args.num_semesters is None or args.num_courses_per_semester is None or args.num_rooms is None or
             args.num_timeslots_per_day is None or args.num_days_for_exams is None or args.num_professors is None):
        parser.error("When exam-timetabling is chosen as benchmark, a number of semesters, a number of courses per"
                     "semester, a number of rooms, a number of timeslots per day, a number of days for exams"
                     " and a number of professors must be specified")

    return args
