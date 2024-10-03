import random
from itertools import combinations

from cpmpy import *
import cpmpy as cp
import numpy as np


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
        for (j1, t1), (j2, t2) in combinations(zip(*tasks_on_mach), 2):
            m += (end[j1, t1] <= start[j2, t2]) | (end[j2, t2] <= start[j1, t1])

    C = list(model.constraints)

    temp = []
    for c in C:
        if isinstance(c, cp.expressions.core.Comparison):
            temp.append(c)
        elif isinstance(c, cp.expressions.variables.NDVarArray):
            temp.extend(c.flatten())
    C_T = set(temp)

    max_duration = max(duration)

    if not model.solve():
        print("no solution")
        raise Exception("The problem has no solution")

    return grid, C_T, max_duration
