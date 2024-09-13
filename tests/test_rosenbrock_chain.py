#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from crfmnes import CRFMNES


def rosenbrock_chain(x):
    x = x.reshape(-1)
    return np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

def test_run_d40_rosen():
    print("test_run_d40:")
    dim = 40
    mean = np.zeros([dim, 1])
    sigma = 2.0
    lamb = 40 # note that lamb (sample size) should be even number
    allowable_evals = (52.2 + 1.5*3) * 1e3 # 2 sigma
    iteration_number = int(allowable_evals / lamb) + 1

    cr = CRFMNES(dim, rosenbrock_chain, mean, sigma, lamb, dtype=np.float128)
    x_best, f_best = cr.optimize(iteration_number)
    print("f_best:{}".format(f_best))
    assert f_best < 1e-12


def test_run_d80_rosen():
    print("test_run_d80:")
    dim = 80
    mean = np.zeros([dim, 1])
    sigma = 2.0
    lamb = 64 # note that lamb (sample size) should be even number
    allowable_evals = (192 + 2.0*3) * 1e3 # 2 sigma
    iteration_number = int(allowable_evals / lamb) + 1

    cr = CRFMNES(dim, rosenbrock_chain, mean, sigma, lamb, dtype=np.float128)
    x_best, f_best = cr.optimize(iteration_number)
    print("f_best:{}".format(f_best))
    assert f_best < 1e-12
