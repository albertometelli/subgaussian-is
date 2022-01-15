# Copyright (C) 2021 Alberto Maria Metelli, Alessio Russo, and Politecnico di Milano. All rights reserved.
# Licensed under the Apache 2.0 License.

from scipy.optimize import root_scalar
import numpy as np
import matplotlib.pyplot as plt
from ..dataset.regression import ActionDist


def estimate_lambda(weights, n, delta, alpha=2, return_info=False):

    eta = estimate_lambda2(weights, np.sqrt(n), delta, alpha, return_info)
    if return_info:
        return eta[0] / n ** .25, eta[1]
    return eta / n ** .25

def estimate_lambda2(weights, n, delta, alpha=2, return_info=False):

    if alpha != 2:
        raise NotImplementedError

    rhs = 2 * np.log(1 / delta) / (3 * n)

    def fun(x):
        return x ** 2 * np.mean(weights ** 2 / (1 - x + x * weights) ** 2) - rhs

    def deriv(x):
        return 2 * x * np.mean(weights ** 2 / (1 - x + x * weights) ** 3)

    def f(x):
        if x < 0 or x > 1:
            return np.inf, np.inf
        #print(x, fun(x))
        return fun(x), deriv(x)

    guess1 = 1
    res1 = root_scalar(f, x0=guess1, fprime=True)



    if return_info:
        if res1.converged:
            return np.clip(res1.root, 0, 1), res1.converged
        return 0, res1.converged
    else:
        if res1.converged:
            return np.clip(res1.root, 0, 1)
        return 0


def compute_renyi_divergence(action_dist, behav_action_dist, alpha=2):

    if alpha != 2:
        raise NotImplementedError

    if isinstance(action_dist, ActionDist): #We assume it's Gaussian
        mean_i = action_dist.preds
        mean_j = behav_action_dist.preds
        var_i = action_dist.sigma_e ** 2
        var_j = behav_action_dist.sigma_e ** 2
        var_star = alpha*var_j + (1 - alpha)*var_i
        contextual_non_exp_Renyi = np.log(var_j ** .5 / var_i ** .5) + 1 / (2 * (alpha - 1)) * np.log(var_j / var_star) + alpha * (
                    mean_i - mean_j) ** 2 / (2 * var_star)
        non_exp_Renyi = np.mean(contextual_non_exp_Renyi)
        exp_Renyi = np.exp(non_exp_Renyi)
    elif isinstance(action_dist, np.ndarray): #We assume it's categorical
        action_dist2 = np.power(action_dist, 2)
        contextual_Renyi = np.sum(action_dist2 / behav_action_dist, axis=1)
        exp_Renyi = np.mean(contextual_Renyi)
    else:
        raise ValueError

    return exp_Renyi

def find_minimum_bound(d, n, delta):

    t = np.log(1 / delta)

    def fun(x):
        el = d + x - d * x
        den = 6 * n * x ** 2 * np.sqrt(el)
        num = 3 * np.sqrt(2 * n * t) * (1 - d) * x ** 2 + \
                3 * n * np.sqrt(d - 1) * (1 - d) * x ** 3 - \
                4 * t * np.sqrt(el) + 6 * n * x ** 2 * el * np.sqrt(d - 1)

        return num / den

    def deriv(x):
        el = d + x - d * x
        n1 = 4 * t
        d1 = 3 * n * x ** 3
        n2 = - (d - 1) ** 2 * t
        d2 = 2 * np.sqrt(2 * n * t) * el ** (3 / 2)
        n3 = - (d - 1) ** (5 / 2) * x
        d3 = 4 * el ** (3 / 2)
        n4 = - (d - 1) ** (3 / 2)
        d4 = np.sqrt(el)

        return n1 / d1 + n2 / d2 + n3 / d3 + n4 / d4

    def f(x):
        if x < 0 or x > 1:
            return np.inf, np.inf
        #print(x, fun(x))
        return fun(x), deriv(x)

    guess1 = np.sqrt(2 * t / (3 * d * n))
    res1 = root_scalar(f, x0=guess1, fprime=True)

    # guess2 = 1
    # res2 = root_scalar(f, x0=guess2, fprime=True)

    if res1.converged:
        root = res1.root
        if deriv(root) > 0:
            return root

    return None

    # print(res1)
    # print(res1.root, deriv(res1.root))
    # print(res2.root, deriv(res2.root))

def find_optimal_lambda(d, n, delta):

    t = np.log(1 / delta)

    def bound(x):
        el = d + x - d * x
        return 2 * t / (3 * n * x) + x * np.sqrt((d - 1) * el) + np.sqrt(2 * t * el / n)

    lambda_wmin = find_minimum_bound(d, n, delta)

    if lambda_wmin is None:
        return 1.
    else:
        if bound(lambda_wmin) < bound(1.):
            return np.clip(lambda_wmin, 0, 1)
        else:
            return 1.
