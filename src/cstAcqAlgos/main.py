import os
import argparse

from cpmpy.transformations.normalize import toplevel_list

from src.cstAcqAlgos.QuAcq import QuAcq
from src.cstAcqAlgos.MQuAcq import MQuAcq
from src.cstAcqAlgos.MQuAcq2 import MQuAcq2
from src.cstAcqAlgos.GrowAcq import GrowAcq

from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.cstAcqAlgos.benchmarks import construct_9sudoku, construct_4sudoku, construct_jsudoku, construct_random122, \
    construct_new_random, construct_random495, construct_golomb8, construct_murder_problem, \
    construct_job_shop_scheduling_problem, construct_examtt_simple, construct_examtt_advanced, \
    construct_nurse_rostering, construct_nurse_rostering_advanced
from src.prior.prior import Prior
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(prog='PROG', usage='%(prog)s [options]')

    # Parsing algorithm
    parser.add_argument("-a", "--algorithm", type=str, choices=["quacq", "mquacq", "mquacq2", "mquacq2-a", "growacq"],
                        required=True,
                        help="The name of the algorithm to use")
    # Parsing specific to GrowAcq
    parser.add_argument("-ia", "--inner-algorithm", type=str, choices=["quacq", "mquacq", "mquacq2", "mquacq2-a"],
                        required=False,
                        help="Only relevant when the chosen algorithm is GrowAcq - "
                             "the name of the inner algorithm to use")

    # Parsing query generation method
    parser.add_argument("-qg", "--query-generation", type=str, choices=["baseline", "base", "tqgen", "pqgen"],
                        help="The version of the query generation method to use", default="pqgen")
    parser.add_argument("-o", "--objective", type=str, choices=["max", "sol", "p", "prob", "proba", "class"],
                        help="The objective function used in query generation", default="max")
    # Parsing findscope method
    parser.add_argument("-fs", "--findscope", type=int, choices=[1, 2], required=False,
                        help="The version of the findscope method to use", default=2)
    # Parsing findc method
    parser.add_argument("-fc", "--findc", type=int, choices=[1, 2], required=False,
                        help="The version of the findc method to use", default=1)

    # Parsing if query generation, FindScope and/or FindC should be guided
    parser.add_argument("-gqg", "--guide-qgen", action='store_true',
                        help="Use this to guide query generation")  # query generation
    parser.add_argument("-gfs", "--guide-findscope", action='store_true',
                        help="Use this to guide FindScope")  # FindScope
    parser.add_argument("-gfc", "--guide-findc", action='store_true', help="Use this to guide FindC")  # FindC

    # Parsing specific to guiding
    parser.add_argument("-c", "--classifier", type=str, choices=["counts", "random_forest", "MLP", "GaussianNB",
                                                                 "CategoricalNB", "SVM"], required=False,
                        help="Only relevant when at least one component of Constraint Acquisition is guided - "
                             "the machine learning classifier to use")

    # Parsing specific to MLP classifier
    parser.add_argument("-hls", "--hidden-layers", type=int, required=False, nargs='+',
                        help="Only relevant when the chosen classifier is "
                             "MLP - the hidden layer shapes, e.g. -hls 16 32 64 creates 3 hidden layers of sizes 16, "
                             "32 and 64, respectively")
    parser.add_argument("-lr", "--learning-rate", type=float, required=False,
                        help="Only relevant when the chosen classifier is "
                             "MLP - the learning rate")

    # Parsing time limit - will default to None if none is provided
    parser.add_argument("-t", "--time-limit", type=float, help="An optional time limit")

    # Parsing benchmark
    parser.add_argument("-b", "--benchmark", type=str, required=True,
                        choices=["9sudoku", "4sudoku", "jsudoku", "random122", "random495", "new_random",
                                 "golomb8", "murder", "job_shop_scheduling",
                                 "exam_timetabling", "exam_timetabling_simple", "exam_timetabling_adv",
                                 "exam_timetabling_advanced", "nurse_rostering", "nurse_rostering_simple",
                                 "nurse_rostering_advanced", "nurse_rostering_adv"],
                        help="The name of the benchmark to use")

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

    # NEW ARGS:
    parser.add_argument("-pf", "--prior-file", type=str, required=False,
                        help="", default="")

    group_architecture = parser.add_mutually_exclusive_group(required=False)
    group_architecture.add_argument("-paf", action="store_true", dest="prior_as_feature", help="")
    group_architecture.add_argument("-pam", action="store_true", dest="prior_as_multiplier", help="")
    group_architecture.add_argument("-pamd", action="store_true", dest="prior_as_multiplier_dyn", help="")
    parser.set_defaults(prior_as_feature=False,
                        prior_as_multiplier=False,
                        prior_as_multiplier_dyn=False)

    parser.add_argument("-lam", type=float, required=False, dest="prior_as_multiplier_lambda", help="",
                                    default=0.5)

    parser.add_argument("-dec", type=float, required=False, dest="prior_as_multiplier_decay", help="",
                                    default=0.99)

    args = parser.parse_args()

    # Additional validity checks
    if args.algorithm == "growacq" and args.inner_algorithm is None:
        parser.error("When GrowAcq is chosen as main algorithm, an inner algorithm must be specified")
    if args.query_generation in ["baseline", "base"]:
        args.query_generation = "base"
    if not (args.guide_qgen or args.guide_findscope or args.guide_findc):
        if args.objective not in ["max", "sol"]:
            parser.error(
                "When no component of Constraint Acquisition is guided, then this objective function is not allowed")
    else:
        if args.objective in ["max", "sol"]:
            parser.error(
                "When at least one component of Constraint Acquisition is guided, this objective function is not "
                "allowed")
    if args.objective in ["p", "prob", "proba"]:
        args.objective = "proba"
        if args.classifier is None:
            parser.error("When the objective function using probabilities is chosen, a classifier must be specified")
    if args.objective in ["class"] and args.classifier is None:
        parser.error("When the objective function using classification is chosen, a classifier must be specified")
    if args.classifier == "MLP" and args.hidden_layers is None:
        parser.error("When an MLP classifier is used, the hidden layer"
                     "structure must be specified, e.g., -hls 16 32 64 creates 3 hidden layers of sizes 16, "
                     "32 and 64, respectively")
    if args.classifier == "MLP" and args.learning_rate is None:
        parser.error("When an MLP classifier is used, the learning rate must"
                     "be specified")
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

    # NEW ARGS CHECKS
    if args.prior_as_feature or args.prior_as_multiplier or args.prior_as_multiplier_dyn:
        if len(args.prior_file) == 0:
            parser.error("When using a prior, a prior file needs to be given")

    return args


def construct_benchmark():
    if args.benchmark == "9sudoku":
        grid, C_T, oracle = construct_9sudoku()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "4sudoku":
        grid, C_T, oracle = construct_4sudoku()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "jsudoku":
        grid, C_T, oracle = construct_jsudoku()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "random122":
        grid, C_T, oracle = construct_random122()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "new_random":
        grid, C_T, oracle = construct_new_random()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "random495":
        grid, C_T, oracle = construct_random495()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "golomb8":
        grid, C_T, oracle = construct_golomb8()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2",
                 "abs(var1 - var2) != abs(var3 - var4)"]
    #            "abs(var1 - var2) == abs(var3 - var4)"]

    elif args.benchmark == "murder":
        grid, C_T, oracle = construct_murder_problem()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "job_shop_scheduling":
        grid, C_T, oracle, max_duration = construct_job_shop_scheduling_problem(args.num_jobs, args.num_machines,
                                                                                args.horizon, args.seed)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"var1 + {i} == var2" for i in range(1, max_duration + 1)] + \
                [f"var2 + {i} == var1" for i in range(1, max_duration + 1)]

    elif args.benchmark == "exam_timetabling" or args.benchmark == "exam_timetabling_simple":
        slots_per_day = args.num_rooms * args.num_timeslots_per_day

        grid, C_T, oracle = construct_examtt_simple(args.num_semesters, args.num_courses_per_semester, args.num_rooms,
                                                    args.num_timeslots_per_day, args.num_days_for_exams)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"(var1 // {slots_per_day}) != (var2 // {slots_per_day})",
                 f"(var1 // {slots_per_day}) == (var2 // {slots_per_day})"]

    elif args.benchmark == "exam_timetabling_adv" or args.benchmark == "exam_timetabling_advanced":
        slots_per_day = args.num_rooms * args.num_timeslots_per_day

        grid, C_T, oracle = construct_examtt_advanced(args.num_semesters, args.num_courses_per_semester, args.num_rooms,
                                                      args.num_timeslots_per_day, args.num_days_for_exams,
                                                      args.num_professors)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"abs(var1 - var2) // {slots_per_day} <= 2"] + \
                [f"(var1 // {slots_per_day}) != (var2 // {slots_per_day})"]
        # [f"var1 // {slots_per_day} != {d}" for d in range(num_days_for_exams)]
    elif args.benchmark == "nurse_rostering":

        grid, C_T, oracle = construct_nurse_rostering(args.num_nurses, args.num_shifts_per_day,
                                                      args.num_days_for_schedule)

        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "nurse_rostering_adv" or args.benchmark == "nurse_rostering_advanced":

        grid, C_T, oracle = construct_nurse_rostering_advanced(args.num_nurses, args.num_shifts_per_day,
                                                               args.nurses_per_shift, args.num_days_for_schedule)

        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    else:
        raise NotImplementedError(f'Benchmark {args.benchmark} not implemented yet')

    return args.benchmark, grid, C_T, oracle, gamma


def construct_classifier():
    classifier = None

    if args.classifier == "random_forest":
        classifier = RandomForestClassifier()
    elif args.classifier == "MLP":
        classifier = MLPClassifier(hidden_layer_sizes=tuple(args.hidden_layers), activation='relu', solver='adam',
                                   random_state=1, learning_rate_init=args.learning_rate)
    elif args.classifier == "CategoricalNB":
        classifier = CategoricalNB(min_categories=5)
    elif args.classifier == "GaussianNB":
        classifier = GaussianNB()
    elif args.classifier == "SVM":
        classifier = SVC(kernel='rbf', C=100, gamma='scale', probability=True)

    return classifier


def save_results(alg=None, inner_alg=None, qg=None, tl=None, t=None, fs=None, fc=None, clf=None, hl=None,
                 lr=None, bench=None, start_time=None, conacq=None):
    if conacq is None: conacq = ca_system
    if alg is None: alg = args.algorithm
    if qg is None: qg = args.query_generation
    if fs is None: fs = args.findscope
    if fc is None: fc = args.findc
    if bench is None: bench = benchmark_name
    if start_time is None: start_time = start

    end = time.time()  # to measure the total time of the acquisition process
    total_time = end - start_time

    print("\n\nConverged ------------------------")

    print("Total number of queries: ", conacq.metrics.queries_count)

    avg_size = conacq.metrics.average_size_queries / conacq.metrics.queries_count if conacq.metrics.queries_count > 0 else 0
    print("Average size of queries: ", avg_size)

    print("Total time: ", total_time)
    average_waiting_time = total_time / conacq.metrics.queries_count if conacq.metrics.queries_count > 0 else 0

    print("Average waiting time for a query: ", average_waiting_time)
    print("Maximum waiting time for a query: ", conacq.metrics.max_waiting_time)

    print("C_L size: ", len(toplevel_list(conacq.C_l.constraints)))

    # res_name = ["results/results"]
    # res_name.append(alg)
    #
    # # results_file = "results/results_" + args.algorithm + "_"
    # if alg == "growacq":
    #     if inner_alg is None: inner_alg = args.inner_algorithm
    #     res_name.append(inner_alg)
    #     # results_file += args.inner_algorithm + "_"

    # res_name.append(f"{str(qg)}")
    #
    # if qg == "tqgen":
    #     if tl is None: tl = args.time_limit
    #     if t is None: t = 0.1
    #     res_name.append(f"tl{str(tl)}")
    #     res_name.append(f"t{str(t)}")
    #
    # res_name.append(f"fs{str(fs)}")
    #
    # res_name.append(f"fc{str(fc)}")
    #
    # if conacq.obj == "proba" or conacq.obj == "class":
    #
    #     res_name.append(str(conacq.obj))
    #
    #     if clf is None: clf = args.classifier
    #     res_name.append(str(clf))
    #
    #     if clf == "MLP":
    #         if hl is None: hl = args.hidden_layers
    #         if lr is None: lr = args.learning_rate
    #         res_name.append(str(hl))
    #         res_name.append(str(lr))
    #
    # res_name.append(bench)

    # results_file = "_".join(res_name)

    # file_exists = os.path.isfile(results_file)
    # f = open(results_file, "a")

    # if not file_exists:
    results = "CL\tTot_q\ttop_lvl_q\tgen_q\tfs_q\tfc_q\tavg|q|\tgen_time\tavg_t\tmax_t\ttot_t\tconv\n"
    # else:
    #     results = ""

    results += str(len(toplevel_list(conacq.C_l.constraints))) + "\t" + str(conacq.metrics.queries_count) + "\t" + str(
        conacq.metrics.top_lvl_queries) \
               + "\t" + str(conacq.metrics.generated_queries) + "\t" + str(
        conacq.metrics.findscope_queries) + "\t" + str(conacq.metrics.findc_queries)

    avg_size = round(conacq.metrics.average_size_queries / conacq.metrics.queries_count,
                     4) if conacq.metrics.queries_count > 0 else 0

    avg_qgen_time = round(conacq.metrics.generation_time / conacq.metrics.generated_queries,
                          4) if conacq.metrics.generated_queries > 0 else 0
    results += "\t" + str(avg_size) + "\t" + str(avg_qgen_time) \
               + "\t" + str(round(average_waiting_time, 4)) + "\t" + str(
        round(conacq.metrics.max_waiting_time, 4)) + "\t" + \
               str(round(total_time, 4))

    results += "\t" + str(conacq.metrics.converged) + "\n"

    print("")
    print(results)
    # f.write(results)
    # f.close()


if __name__ == "__main__":

    # Setup
    args = parse_args()
    print(args)

    benchmark_name, grid, C_T, oracle, gamma = construct_benchmark()
    grid.clear()
    print("Size of C_T: ", len(C_T))

    classifier = construct_classifier()

    if args.findscope is None:
        fs_version = 2
    else:
        fs_version = args.findscope

    if args.findc is None:
        fc_version = 1
    else:
        fc_version = args.findc

    start = time.time()
    if args.algorithm == "quacq":
        ca_system = QuAcq(gamma, grid, C_T, qg=args.query_generation, obj=args.objective, classifier=classifier,
                          gqg=args.guide_qgen, gfs=args.guide_findscope, gfc=args.guide_findc,
                          classifier_name=args.classifier, time_limit=args.time_limit, findscope_version=fs_version,
                          findc_version=fc_version)
    elif args.algorithm == "mquacq":
        ca_system = MQuAcq(gamma, grid, C_T, qg=args.query_generation, obj=args.objective, classifier=classifier,
                           gqg=args.guide_qgen, gfs=args.guide_findscope, gfc=args.guide_findc,
                           classifier_name=args.classifier, time_limit=args.time_limit, findscope_version=fs_version,
                           findc_version=fc_version)
    elif args.algorithm == "mquacq2":
        ca_system = MQuAcq2(gamma, grid, C_T, qg=args.query_generation, obj=args.objective, classifier=classifier,
                            gqg=args.guide_qgen, gfs=args.guide_findscope, gfc=args.guide_findc,
                            classifier_name=args.classifier, time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version)
    elif args.algorithm == "mquacq2-a":
        ca_system = MQuAcq2(gamma, grid, C_T, qg=args.query_generation, obj=args.objective, classifier=classifier,
                            gqg=args.guide_qgen, gfs=args.guide_findscope, gfc=args.guide_findc,
                            classifier_name=args.classifier, time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version, perform_analyzeAndLearn=True)
    elif args.algorithm == "growacq":
        ca_system = GrowAcq(gamma, grid, C_T, qg=args.query_generation, obj=args.objective, classifier=classifier,
                            gqg=args.guide_qgen, gfs=args.guide_findscope, gfc=args.guide_findc,
                            classifier_name=args.classifier, time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version)
    else:
        raise Exception("Algorithm not implemented")

    # Load prior
    prior = None
    if "prior_file" in args and not args.prior_file == "":
        prior = Prior(args.prior_file)
        prior.variables(grid)
        if args.prior_as_feature:
            ca_system.prior_use = "feat"
        if args.prior_as_multiplier:
            ca_system.prior_use = "mul"
            ca_system.prior_param["lam"] = args.prior_as_multiplier_lambda
        if args.prior_as_multiplier_dyn:
            ca_system.prior_use = "muldyn"
            ca_system.prior_param["lam"] = args.prior_as_multiplier_lambda
            ca_system.prior_param["dec"] = args.prior_as_multiplier_decay
        ca_system.prior = prior

    ca_system.learn()

    save_results()
