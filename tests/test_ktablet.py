#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from crfmnes import CRFMNES


def ktablet(x):
    x = x.reshape(-1)
    dim = len(x)
    k = int(dim / 4)
    return np.sum(x[0:k]**2) + np.sum((100*x[k:dim])**2)


def test_run_d40_ktablet():
    print("test_run_d40:")
    dim = 40
    mean = np.ones([dim, 1]) * 3
    sigma = 2.0
    lamb = 16 # note that lamb (sample size) should be even number
    allowable_evals = (9.1 + 0.6*3) * 1e3 # 2 sigma
    iteration_number = int(allowable_evals / lamb) + 1

    cr = CRFMNES(dim, ktablet, mean, sigma, lamb, dtype=np.float128)
    x_best, f_best = cr.optimize(iteration_number)
    print("f_best:{}".format(f_best))
    assert f_best < 1e-12


def test_run_d80_ktablet():
    print("test_run_d80:")
    dim = 80
    mean = np.ones([dim, 1]) * 3
    sigma = 2.0
    lamb = 16 # note that lamb (sample size) should be even number
    allowable_evals = (18.6 + 0.8*3) * 1e3 # 2 sigma
    iteration_number = int(allowable_evals / lamb) + 1

    cr = CRFMNES(dim, ktablet, mean, sigma, lamb, dtype=np.float128)
    x_best, f_best = cr.optimize(iteration_number)
    print("f_best:{}".format(f_best))
    assert f_best < 1e-12
