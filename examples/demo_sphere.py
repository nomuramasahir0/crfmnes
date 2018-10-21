#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from crfmnes import CRFMNES


def sphere(x):
    x = x.reshape(-1)
    return np.sum(x**2)


def main():
    dim = 40
    mean = np.ones([dim, 1]) * 2
    sigma = 0.5
    lamb = 16 # note that lamb (sample size) should be even number
    iteration_number = 500

    cr = CRFMNES(dim, sphere, mean, sigma, lamb)
    x_best, f_best = cr.optimize(iteration_number)
    print("x_best:{}, f_best:{}".format(x_best, f_best))


if __name__ == '__main__':
    main()