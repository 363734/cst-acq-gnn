import random
from itertools import combinations

from cpmpy import *
from cpmpy.transformations.normalize import toplevel_list


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

    if not model.solve():
        print("no solution")
        raise Exception("The problem has no solution")

    C_T = set(toplevel_list(C))

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
                for c1, c2 in combinations(prof_courses, 2):
                    model += abs(c1 - c2) // slots_per_day <= 2  # all her courses in 2 days

    if not model.solve():
        print("no solution")
        raise Exception("The problem has no solution")

    C_T = set(toplevel_list(C))

    return courses, C_T
