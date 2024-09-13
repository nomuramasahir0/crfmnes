#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

# evaluation value of the infeasible solution
INFEASIBLE = np.inf


def get_h_inv(dim):
    f = lambda a, b: ((1. + a**2.) * math.exp(a**2. / 2.) / .24) - 10. - b
    f_prime = lambda a: a * math.exp(a**2. / 2.) * (3. + a**2.) / .24
    h_inv = 6.
    while abs(f(h_inv, dim)) > 1e-10:
        last = h_inv
        h_inv = h_inv - .5 * (f(h_inv, dim) / f_prime(h_inv))
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
        distances = np.sum(infeasible_z**2, axis=0)
        infeasible_indices = sorted_indices[no_of_feasible_solutions:]
        indices_sorted_by_distance = np.argsort(distances)
        sorted_indices[no_of_feasible_solutions:] = infeasible_indices[indices_sorted_by_distance]
    return sorted_indices


class CRFMNES:
    def __init__(self, dim, f, m, sigma, lamb, **kwargs):

        self.t = kwargs.get('dtype', np.float64)
        t = lambda v: self.t(v)
        self._1 = t(1.)
        self._2 = t(2.)
        self._dim = t(dim)

        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])
        self.dim = dim
        self.f = f
        self.m = t(m)
        self.sigma = t(sigma)
        self.lamb = lamb
        assert (lamb > 0 and lamb % 2 == 0), f"The value of 'lamb' must be an even, positive integer greater than 0"

        self.v = kwargs.get('v', np.random.randn(dim, 1) / np.sqrt(dim)).astype(self.t)
        self.D = np.ones([dim, 1], dtype=self.t)
        self.constraint = np.asarray(kwargs.get('constraint', [[-t('inf'), +t('inf')] for _ in range(dim)]), dtype=self.t)
        self.penalty_coef = t(kwargs.get('penalty_coef', 1e5))
        self.use_constraint_violation = kwargs.get('use_constraint_violation', True)

        self.w_rank_hat = (np.log(self.lamb / 2 + 1, dtype=self.t) - np.log(np.arange(1, self.lamb + 1), dtype=self.t)).reshape(self.lamb, 1)
        self.w_rank_hat[np.where(self.w_rank_hat < 0.)] = 0.
        self.w_rank = self.w_rank_hat / np.sum(self.w_rank_hat) - (1. / self.lamb)
        self.mueff = np.reciprocal(((self.w_rank + (1. / self.lamb)).T @ (self.w_rank + (1. / self.lamb)))[0][0])
        self.cs = (self.mueff + self._2) / (self.mueff + self._dim + t(5.))
        self.cc = (self.mueff / self._dim + t(4.)) / (self.mueff * t(2.) / self._dim + self._dim + t(4.))
        self.c1_cma = np.reciprocal(np.power(self._dim + 1.3, 2, dtype=self.t) + self.mueff) * self._2
        # initialization
        self.chiN = np.sqrt(self._dim, dtype=self.t) * (np.reciprocal(21 * self.dim**2, dtype=self.t) - np.reciprocal(4 * self.dim, dtype=self.t) + self._1)
        self.pc = np.zeros([self.dim, 1], dtype=self.t)
        self.ps = np.zeros([self.dim, 1], dtype=self.t)
        # distance weight parameter
        self.h_inv = t(get_h_inv(self.dim))
        self.alpha_dist = lambda lambF: self.h_inv * min(1., np.sqrt(t(self.lamb) / self._dim)) * np.sqrt(
            t(lambF) / self.lamb)
        self.w_dist_hat = lambda z, lambF: np.exp(self.alpha_dist(lambF) * np.linalg.norm(z))
        # learning rate
        self.eta_m = self._1
        self.eta_move_sigma = self._1
        self.eta_stag_sigma = lambda lambF: np.tanh((t(lambF) * .024 + t(.70) * self._dim + 20.) / t(self.dim + 12), dtype=self.t)
        self.eta_conv_sigma = lambda lambF: np.tanh((t(lambF) * .025 + t(.75) * self._dim + 10.) / t(self.dim +  4), dtype=self.t) * self._2
        self.c1 = lambda lambF: self.c1_cma * (self.dim - 5) / 6. * (t(lambF) / self.lamb)
        self.eta_B = lambda lambF: np.tanh((min(t(lambF) * .02, np.log(self._dim, dtype=self.t) * 3.) + 5.) / t(.23 * self._dim + 25.), dtype=self.t)

        self.g = 0
        self.no_of_evals = 0

        self.idxp = np.arange(self.lamb / 2, dtype=int)
        self.idxm = np.arange(self.lamb / 2, self.lamb, dtype=int)
        self.z = np.zeros([self.dim, self.lamb], dtype=self.t)

        self.f_best = t('inf')
        self.x_best = np.empty(self.dim, dtype=self.t)

    def calc_violations(self, x):
        violations = np.zeros(self.lamb, dtype=self.t)
        for i in range(self.lamb):
            for j in range(self.dim):
                violations[i] += (-min(0, x[j][i] - self.constraint[j][0]) + max(0, x[j][i] - self.constraint[j][1])) * self.penalty_coef
        return violations

    def optimize(self, iterations):
        for _ in range(iterations):
            # print("f_best:{}".format(self.f_best))
            _ = self.one_iteration()
        return self.x_best, self.f_best

    def one_iteration(self):
        d = self.dim
        lamb = self.lamb
        zhalf = np.random.randn(d, int(lamb / 2)).astype(self.t)  # dim x lamb/2
        self.z[:, self.idxp] = +zhalf
        self.z[:, self.idxm] = -zhalf
        normv = self.t(np.linalg.norm(self.v))
        normv2 = np.square(normv, dtype=self.t)
        vbar = self.v / normv
        y = self.z + (np.sqrt(normv2 + self._1) - self._1) * vbar @ (vbar.T @ self.z)
        x = self.m + self.sigma * y * self.D
        evals_no_sort = np.array(
            [self.f(np.array(x[:, i].reshape(self.dim, 1), dtype=self.t)) for i in range(self.lamb)],
            dtype=self.t)
        xs_no_sort = [x[:, i] for i in range(lamb)]

        violations = np.zeros(lamb, dtype=self.t)
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
        lambF = np.sum(evals_no_sort < np.finfo(self.t).max)

        # evolution path p_sigma
        self.ps = (self._1 - self.cs) * self.ps + np.sqrt(self.cs * (self._2 - self.cs) * self.mueff) * (self.z @ self.w_rank)
        ps_norm = np.linalg.norm(self.ps)
        # distance weight
        w_tmp = np.array(
            [self.w_rank_hat[i] * self.w_dist_hat(np.array(self.z[:, i], dtype=self.t), lambF) for i in range(self.lamb)],
            dtype=self.t).reshape(self.lamb, 1)
        weights_dist = w_tmp / np.sum(w_tmp) - self._1 / self.lamb
        # switching weights and learning rate
        weights = weights_dist if ps_norm >= self.chiN else self.w_rank
        eta_sigma = self.eta_move_sigma if ps_norm >= self.chiN else self.eta_stag_sigma(
            lambF) if ps_norm >= .1 * self.chiN else self.eta_conv_sigma(lambF)
        # update pc, m
        wxm = (x - self.m) @ weights
        self.pc = (self._1 - self.cc) * self.pc + np.sqrt(self.cc * (self._2 - self.cc) * self.mueff) * wxm / self.sigma
        self.m += self.eta_m * wxm
        # calculate s, t
        # step1
        normv4 = np.square(normv2, dtype=self.t)
        exY = np.append(y, self.pc / self.D, axis=1)  # dim x lamb+1
        yy = exY * exY  # dim x lamb+1
        ip_yvbar = vbar.T @ exY
        yvbar = exY * vbar  # dim x lamb+1. exYのそれぞれの列にvbarがかかる
        gammav = normv2 + self._1
        vbarbar = vbar * vbar
        alphavd = np.min(
            [self._1, np.sqrt(normv4 + (gammav * self._2 - np.sqrt(gammav)) / np.max(vbarbar)) / (normv2 + self._2)])  # scalar
        t = exY * ip_yvbar - vbar * (ip_yvbar ** self._2 + gammav) / self._2  # dim x lamb+1
        b = (alphavd ** self._2 - self._1) * normv4 / gammav + self._2 * alphavd ** self._2
        H = np.full([self.dim, 1], self._2, dtype=self.t) - (b + self._2 * alphavd ** self._2) * vbarbar  # dim x 1
        invH = H ** -self._1
        s_step1 = yy - normv2 / gammav * (yvbar * ip_yvbar) - np.ones([self.dim, self.lamb + 1], dtype=self.t)  # dim x lamb+1
        ip_vbart = vbar.T @ t  # 1 x lamb+1
        s_step2 = s_step1 - alphavd / gammav * ((self._2 + normv2) * (t * vbar) - normv2 * vbarbar @ ip_vbart)  # dim x lamb+1
        invHvbarbar = invH * vbarbar
        ip_s_step2invHvbarbar = invHvbarbar.T @ s_step2  # 1 x lamb+1
        s = (s_step2 * invH) - b / (
                    self._1 + b * vbarbar.T @ invHvbarbar) * invHvbarbar @ ip_s_step2invHvbarbar  # dim x lamb+1
        ip_svbarbar = vbarbar.T @ s  # 1 x lamb+1
        t = t - alphavd * ((self._2 + normv2) * (s * vbar) - vbar @ ip_svbarbar)  # dim x lamb+1
        # update v, D
        exw = np.append(self.eta_B(lambF) * weights, np.array([self.c1(lambF)], dtype=self.t).reshape(1, 1),
                        axis=0)  # lamb+1 x 1
        self.v = self.v + (t @ exw) / normv
        self.D = self.D + (s @ exw) * self.D
        # calculate detA
        nthrootdetA = np.exp(np.sum(np.log(self.D)) / self._dim + np.log(1 + self.v.T @ self.v) / (2 * self.dim))[0][0]
        self.D = self.D / nthrootdetA
        # update sigma
        G_s = np.sum((self.z * self.z - np.ones([self.dim, self.lamb], dtype=self.t)) @ weights) / self._dim
        self.sigma = self.sigma * np.exp(eta_sigma / self._2 * G_s, dtype=self.t)

        return xs_no_sort, evals_no_sort, violations
