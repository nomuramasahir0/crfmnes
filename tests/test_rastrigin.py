#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from crfmnes import CRFMNES
import math


def rastrigin(x):
    x = x.reshape(-1)
    dim = len(x)
    return 10.0 * dim + np.sum([x[i] ** 2 - 10.0 * math.cos(2.0 * math.pi * x[i]) for i in range(dim)])


def test_run_d40_rastrigin():
    print("test_run_d40:")
    dim = 40
    mean = np.ones([dim, 1]) * 3
    sigma = 2.0
    lamb = 1130 # note that lamb (sample size) should be even number
    allowable_evals = (148 + 8.8*3) * 1e3 # 2 sigma
    iteration_number = int(allowable_evals / lamb) + 1

    cr = CRFMNES(dim, rastrigin, mean, sigma, lamb, use_constraint_violation=False)
    x_best, f_best = cr.optimize(iteration_number)
    print("f_best:{}".format(f_best))
    assert f_best < 1e-12


def test_run_d80_rastrigin():
    print("test_run_d80:")
    dim = 80
    mean = np.ones([dim, 1]) * 3
    sigma = 2.0
    lamb = 1600 # note that lamb (sample size) should be even number
    allowable_evals = (296 + 8.0*3) * 1e3 # 2 sigma
    iteration_number = int(allowable_evals / lamb) + 1

    cr = CRFMNES(dim, rastrigin, mean, sigma, lamb, use_constraint_violation=False)
    x_best, f_best = cr.optimize(iteration_number)
    print("f_best:{}".format(f_best))
    assert f_best < 1e-12
