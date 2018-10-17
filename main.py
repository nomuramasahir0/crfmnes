#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from crfmnes.alg import CRFMNES


def sphere(x):
    return np.sum(x**2)


def main():
    dim = 3
    f = sphere
    m = np.ones([dim, 1]) * 0.5
    sigma = 0.2
    lamb = 6
    crfmnes = CRFMNES(dim, f, m, sigma, lamb)

    x_best, f_best = crfmnes.optimize(100)
    print("x_best:{}, f_best:{}".format(x_best, f_best))


if __name__ == '__main__':
    main()
