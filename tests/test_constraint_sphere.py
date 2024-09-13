#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from crfmnes import CRFMNES


def const_sphere(x):
    x = x.reshape(-1)
    if np.sum(x < 0) > 0:
        return np.inf
    return np.sum(x**2)


def test_run_d40_const_sphere():
    print("test_run_d40:")
    dim = 40
    mean = np.ones([dim, 1]) * 10
    sigma = 2.0
    lamb = 16 # note that lamb (sample size) should be even number
    allowable_evals = (19.4 + 1.1*3) * 1e3 # 2 sigma
    iteration_number = int(allowable_evals / lamb) + 1

    cr = CRFMNES(dim, const_sphere, mean, sigma, lamb, dtype=np.float128)
    x_best, f_best = cr.optimize(iteration_number)
    print("f_best:{}".format(f_best))
    assert f_best < 1e-12


def test_run_d80_const_sphere():
    print("test_run_d80:")
    dim = 80
    mean = np.ones([dim, 1]) * 10
    sigma = 2.0
    lamb = 32 # note that lamb (sample size) should be even number
    allowable_evals = (48.8 + 1.4*3) * 1e3 # 2 sigma
    iteration_number = int(allowable_evals / lamb) + 1

    cr = CRFMNES(dim, const_sphere, mean, sigma, lamb, dtype=np.float128)
    x_best, f_best = cr.optimize(iteration_number)
    print("f_best:{}".format(f_best))
    assert f_best < 1e-12
