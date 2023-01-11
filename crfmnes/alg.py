#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import copy

# evaluation value of the infeasible solution
INFEASIBLE = np.inf


def get_h_inv(dim):
    f = lambda a, b: ((1. + a * a) * math.exp(a * a / 2.) / 0.24) - 10. - dim
    f_prime = lambda a: (1. / 0.24) * a * math.exp(a * a / 2.) * (3. + a * a)
    h_inv = 6.0
    while abs(f(h_inv, dim)) > 1e-10:
        last = h_inv
        h_inv = h_inv - 0.5 * (f(h_inv, dim) / f_prime(h_inv))
        if abs(h_inv - last) < 1e-16:
            # Exit early since no further improvements are happening
            break
    return h_inv


def sort_indices_by(evals, z):
    lam = evals.size
    sorted_indices = np.argsort(evals)
    sorted_evals = evals[sorted_indices]
    no_of_feasible_solutions = np.where(sorted_evals != INFEASIBLE)[0].size
    if no_of_feasible_solutions != lam:
        infeasible_z = z[:, np.where(evals == INFEASIBLE)[0]]
        distances = np.sum(infeasible_z ** 2, axis=0)
        infeasible_indices = sorted_indices[no_of_feasible_solutions:]
        indices_sorted_by_distance = np.argsort(distances)
        sorted_indices[no_of_feasible_solutions:] = infeasible_indices[indices_sorted_by_distance]
    return sorted_indices


class CRFMNES:
    def __init__(self, dim, f, m, sigma, lamb, **kwargs):

        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])
        self.dim = dim
        self.f = f
        self.m = m
        self.sigma = sigma
        self.lamb = lamb
        assert (lamb > 0 and lamb % 2 == 0), f"The value of 'lamb' must be an even, positive integer greater than 0"

        self.v = kwargs.get('v', np.random.randn(dim, 1) / np.sqrt(dim))
        self.D = np.ones([dim, 1])
        self.constraint = kwargs.get('constraint', [[- np.inf, np.inf] for _ in range(dim)])
        self.penalty_coef = kwargs.get('penalty_coef', 1e5)
        self.use_constraint_violation = kwargs.get('use_constraint_violation', True)

        self.w_rank_hat = (np.log(self.lamb / 2 + 1) - np.log(np.arange(1, self.lamb + 1))).reshape(self.lamb, 1)
        self.w_rank_hat[np.where(self.w_rank_hat < 0)] = 0
        self.w_rank = self.w_rank_hat / sum(self.w_rank_hat) - (1. / self.lamb)
        self.mueff = 1 / ((self.w_rank + (1 / self.lamb)).T @ (self.w_rank + (1 / self.lamb)))[0][0]
        self.cs = (self.mueff + 2.) / (self.dim + self.mueff + 5.)
        self.cc = (4. + self.mueff / self.dim) / (self.dim + 4. + 2. * self.mueff / self.dim)
        self.c1_cma = 2. / (math.pow(self.dim + 1.3, 2) + self.mueff)
        # initialization
        self.chiN = np.sqrt(self.dim) * (1. - 1. / (4. * self.dim) + 1. / (21. * self.dim * self.dim))
        self.pc = np.zeros([self.dim, 1])
        self.ps = np.zeros([self.dim, 1])
        # distance weight parameter
        self.h_inv = get_h_inv(self.dim)
        self.alpha_dist = lambda lambF: self.h_inv * min(1., math.sqrt(float(self.lamb) / self.dim)) * math.sqrt(
            float(lambF) / self.lamb)
        self.w_dist_hat = lambda z, lambF: math.exp(self.alpha_dist(lambF) * np.linalg.norm(z))
        # learning rate
        self.eta_m = 1.0
        self.eta_move_sigma = 1.
        self.eta_stag_sigma = lambda lambF: math.tanh((0.024 * lambF + 0.7 * self.dim + 20.) / (self.dim + 12.))
        self.eta_conv_sigma = lambda lambF: 2. * math.tanh((0.025 * lambF + 0.75 * self.dim + 10.) / (self.dim + 4.))
        self.c1 = lambda lambF: self.c1_cma * (self.dim - 5) / 6 * (float(lambF) / self.lamb)
        self.eta_B = lambda lambF: np.tanh((min(0.02 * lambF, 3 * np.log(self.dim)) + 5) / (0.23 * self.dim + 25))

        self.g = 0
        self.no_of_evals = 0

        self.idxp = np.arange(self.lamb / 2, dtype=int)
        self.idxm = np.arange(self.lamb / 2, self.lamb, dtype=int)
        self.z = np.zeros([self.dim, self.lamb])

        self.f_best = float('inf')
        self.x_best = np.empty(self.dim)

    def calc_violations(self, x):
        violations = np.zeros(self.lamb)
        for i in range(self.lamb):
            for j in range(self.dim):
                violations[i] += (- min(0, x[j][i] - self.constraint[j][0]) + max(0, x[j][i] - self.constraint[j][1])) * self.penalty_coef
        return violations

    def optimize(self, iterations):
        for _ in range(iterations):
            # print("f_best:{}".format(self.f_best))
            _ = self.one_iteration()
        return self.x_best, self.f_best

    def one_iteration(self):
        d = self.dim
        lamb = self.lamb
        zhalf = np.random.randn(d, int(lamb / 2))  # dim x lamb/2
        self.z[:, self.idxp] = zhalf
        self.z[:, self.idxm] = -zhalf
        normv = np.linalg.norm(self.v)
        normv2 = normv ** 2
        vbar = self.v / normv
        y = self.z + (np.sqrt(1 + normv2) - 1) * vbar @ (vbar.T @ self.z)
        x = self.m + self.sigma * y * self.D
        evals_no_sort = np.array([self.f(np.array(x[:, i].reshape(self.dim, 1))) for i in range(self.lamb)])
        xs_no_sort = [x[:, i] for i in range(lamb)]

        violations = np.zeros(lamb)
        if self.use_constraint_violation:
            violations = self.calc_violations(x)
            sorted_indices = sort_indices_by(evals_no_sort + violations, self.z)
        else:
            sorted_indices = sort_indices_by(evals_no_sort, self.z)
        best_eval_id = sorted_indices[0]
        f_best = evals_no_sort[best_eval_id]
        x_best = x[:, best_eval_id]
        self.z = self.z[:, sorted_indices]
        y = y[:, sorted_indices]
        x = x[:, sorted_indices]

        self.no_of_evals += self.lamb
        self.g += 1
        if f_best < self.f_best:
            self.f_best = f_best
            self.x_best = x_best

        # This operation assumes that if the solution is infeasible, infinity comes in as input.
        lambF = np.sum(evals_no_sort < np.finfo(float).max)

        # evolution path p_sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2. - self.cs) * self.mueff) * (self.z @ self.w_rank)
        ps_norm = np.linalg.norm(self.ps)
        # distance weight
        w_tmp = np.array(
            [self.w_rank_hat[i] * self.w_dist_hat(np.array(self.z[:, i]), lambF) for i in range(self.lamb)]).reshape(
            self.lamb, 1)
        weights_dist = w_tmp / sum(w_tmp) - 1. / self.lamb
        # switching weights and learning rate
        weights = weights_dist if ps_norm >= self.chiN else self.w_rank
        eta_sigma = self.eta_move_sigma if ps_norm >= self.chiN else self.eta_stag_sigma(
            lambF) if ps_norm >= 0.1 * self.chiN else self.eta_conv_sigma(lambF)
        # update pc, m
        wxm = (x - self.m) @ weights
        self.pc = (1. - self.cc) * self.pc + np.sqrt(self.cc * (2. - self.cc) * self.mueff) * wxm / self.sigma
        self.m += self.eta_m * wxm
        # calculate s, t
        # step1
        normv4 = normv2 ** 2
        exY = np.append(y, self.pc / self.D, axis=1)  # dim x lamb+1
        yy = exY * exY  # dim x lamb+1
        ip_yvbar = vbar.T @ exY
        yvbar = exY * vbar  # dim x lamb+1. exYのそれぞれの列にvbarがかかる
        gammav = 1. + normv2
        vbarbar = vbar * vbar
        alphavd = np.min(
            [1, np.sqrt(normv4 + (2 * gammav - np.sqrt(gammav)) / np.max(vbarbar)) / (2 + normv2)])  # scalar
        t = exY * ip_yvbar - vbar * (ip_yvbar ** 2 + gammav) / 2  # dim x lamb+1
        b = -(1 - alphavd ** 2) * normv4 / gammav + 2 * alphavd ** 2
        H = np.ones([self.dim, 1]) * 2 - (b + 2 * alphavd ** 2) * vbarbar  # dim x 1
        invH = H ** (-1)
        s_step1 = yy - normv2 / gammav * (yvbar * ip_yvbar) - np.ones([self.dim, self.lamb + 1])  # dim x lamb+1
        ip_vbart = vbar.T @ t  # 1 x lamb+1
        s_step2 = s_step1 - alphavd / gammav * ((2 + normv2) * (t * vbar) - normv2 * vbarbar @ ip_vbart)  # dim x lamb+1
        invHvbarbar = invH * vbarbar
        ip_s_step2invHvbarbar = invHvbarbar.T @ s_step2  # 1 x lamb+1
        s = (s_step2 * invH) - b / (
                    1 + b * vbarbar.T @ invHvbarbar) * invHvbarbar @ ip_s_step2invHvbarbar  # dim x lamb+1
        ip_svbarbar = vbarbar.T @ s  # 1 x lamb+1
        t = t - alphavd * ((2 + normv2) * (s * vbar) - vbar @ ip_svbarbar)  # dim x lamb+1
        # update v, D
        exw = np.append(self.eta_B(lambF) * weights, np.array([self.c1(lambF)]).reshape(1, 1),
                        axis=0)  # lamb+1 x 1
        self.v = self.v + (t @ exw) / normv
        self.D = self.D + (s @ exw) * self.D
        # calculate detA
        nthrootdetA = np.exp(np.sum(np.log(self.D)) / self.dim + np.log(1 + self.v.T @ self.v) / (2 * self.dim))[0][0]
        self.D = self.D / nthrootdetA
        # update sigma
        G_s = np.sum((self.z * self.z - np.ones([self.dim, self.lamb])) @ weights) / self.dim
        self.sigma = self.sigma * np.exp(eta_sigma / 2 * G_s)

        return xs_no_sort, evals_no_sort, violations
