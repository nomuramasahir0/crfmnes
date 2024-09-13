#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from crfmnes import CRFMNES


def ellipsoid(x):
    x = x.reshape(-1)
    dim = len(x)
    return np.sum([(1000**(i / (dim-1)) * x[i])**2 for i in range(dim)])


def test_run_d40_ellipsoid():
    print("test_run_d40:")
    dim = 40
    mean = np.ones([dim, 1]) * 3
    sigma = 2.0
    lamb = 16 # note that lamb (sample size) should be even number
    allowable_evals = (8.8 + 0.5*3) * 1e3 # 2 sigma
    iteration_number = int(allowable_evals / lamb) + 1

    cr = CRFMNES(dim, ellipsoid, mean, sigma, lamb, dtype=np.float128)
    x_best, f_best = cr.optimize(iteration_number)
    print("f_best:{}".format(f_best))
    assert f_best < 1e-12


def test_run_d80_ellipsoid():
    print("test_run_d80:")
    dim = 80
    mean = np.ones([dim, 1]) * 3
    sigma = 2.0
    lamb = 16 # note that lamb (sample size) should be even number
    allowable_evals = (17.5 + 0.6*3) * 1e3 # 2 sigma
    iteration_number = int(allowable_evals / lamb) + 1

    cr = CRFMNES(dim, ellipsoid, mean, sigma, lamb, dtype=np.float128)
    x_best, f_best = cr.optimize(iteration_number)
    print("f_best:{}".format(f_best))
    assert f_best < 1e-12
