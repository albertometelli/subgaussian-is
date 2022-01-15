# Copyright (C) 2021 Alberto Maria Metelli, Alessio Russo, and Politecnico di Milano. All rights reserved.
# Licensed under the Apache 2.0 License.

import sys
sys.path = ['.'] + sys.path
import numpy as np
import scipy.stats as stats

# import open bandit pipeline (obp)
from obp.ope.utils import find_optimal_lambda, estimate_lambda
from prettytable import PrettyTable
from scipy.stats import t
from scipy.optimize import minimize

SIGMA2_B = [1]
SIGMA2_E = [1.5, 1.9, 1.99, 1.999]
N = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
N = [10, 20, 50, 100, 200, 500, 1000]
N_ref = 10000000    #samples to compute all things
mu_b, mu_e = 0., 0.5
n_runs = 60
significance = 0.1


def f(x):
    return 100*np.cos(2*np.pi*x)

def generate_dataset(n, mu, sigma2):
    generated_samples = stats.norm.rvs(size=n, loc=mu, scale=np.sqrt(sigma2))
    return generated_samples, f(generated_samples)

def compute_renyi_divergence(mu_b, sigma2_b, mu_e, sigma2_e, alpha=2):
    var_star = alpha * sigma2_b + (1 - alpha) * sigma2_e
    contextual_non_exp_Renyi = np.log(sigma2_b ** .5 / sigma2_e ** .5) + 1 / (2 * (alpha - 1)) * np.log(
        sigma2_b / var_star) + alpha * (mu_e - mu_b) ** 2 / (2 * var_star)
    non_exp_Renyi = np.mean(contextual_non_exp_Renyi)
    exp_Renyi = np.exp(non_exp_Renyi)
    return exp_Renyi

def welch_test(res):
    res_mean = np.mean(res, axis=0)
    res_std = np.var(res, axis=0) ** .5
    n = res.shape[0]

    confidence = .98
    ll = []
    for i in range(len(N)):
        optimal = np.argmin(res_mean[i][:-1])
        t_stat = -(res_mean[i][optimal] - res_mean[i]) / np.sqrt(res_std[i][optimal] ** 2 / n + res_std[i] ** 2 / n)
        dof = (res_std[i][optimal] ** 2 / n + res_std[i] ** 2 / n) ** 2 / (res_std[i][optimal] ** 4 / (n ** 2 * (n - 1)) + res_std[i] ** 4 / (n ** 2 * (n - 1)))
        dof = dof.astype(int)
        c = t.ppf(confidence, dof)
        ll.append(t_stat < c)
    print(ll)

    return np.array(ll).T

for sigma2_b in SIGMA2_B:
    for sigma2_e in SIGMA2_E:

        # On policy sampling for reference
        X_on, f_on = generate_dataset(N_ref, mu_e, sigma2_e)
        mu_f_e = np.mean(f_on)
        sigma2_f_e = np.var(f_on)

        d2_Renyi = compute_renyi_divergence(mu_b, sigma2_b, mu_e, sigma2_e)
        print('Reference: ', mu_f_e, sigma2_f_e)
        print('Renyi: ', d2_Renyi)

        target_dist = stats.norm(mu_e, sigma2_e)
        behav_dist = stats.norm(mu_b, sigma2_b)

        res = np.zeros((n_runs, len(N), 8))
        lambdas = np.zeros((n_runs, len(N), 3))
        ess = np.zeros((n_runs, len(N), 8))

        for run in range(n_runs):
            X_all, f_all = generate_dataset(max(N), mu_b, sigma2_b)
            X_on_all, f_on_all = generate_dataset(max(N), mu_e, sigma2_e)
            pdf_e_all = target_dist.pdf(X_all)
            pdf_b_all = behav_dist.pdf(X_all)

            for i, n in enumerate(N):
                X_n, f_n = X_all[:n], f_all[:n]
                X_on_n, f_on_n = X_on_all[:n], f_on_all[:n]
                pdf_e = pdf_e_all[:n]
                pdf_b = pdf_b_all[:n]

                iw = pdf_e / pdf_b
                
                lambda_optimal = np.sqrt(np.log(1 / significance) / (3 * d2_Renyi * n))
                lambda_superoptimal = find_optimal_lambda(d2_Renyi, n, significance)
                lambda_estimated, conv = estimate_lambda(iw, n, significance, return_info=True)
                threshold = np.sqrt((3 * d2_Renyi * n)/(2 * np.log(1 / significance)))

                iw_optimal = iw / ((1 - lambda_optimal) + lambda_optimal * iw)
                iw_superoptimal = iw / ((1 - lambda_superoptimal) + lambda_superoptimal * iw)
                iw_estimated = iw / ((1 - lambda_estimated) + lambda_estimated * iw)
                iw_trucnated = np.clip(iw, 0, threshold)
                iw_sn = iw / np.sum(iw) * n


                def obj(lambda_):
                    shrinkage_weight = (lambda_ * iw) / (iw ** 2 + lambda_)
                    estimated_rewards_ = shrinkage_weight * f_n
                    variance = np.var(estimated_rewards_)
                    bias = np.sqrt(np.mean((iw - shrinkage_weight) ** 2)) * max(f_n)
                    return bias ** 2 + variance
                
                lambda_opt = minimize(obj, x0=np.array([1]), bounds=[(0., np.inf)], method='Powell').x
                iw_os = (lambda_opt * iw) / (iw ** 2 + lambda_opt)

                ess[run, i, 0] = np.sum(iw) ** 2 / np.sum(iw ** 2)
                ess[run, i, 4] = np.sum(iw_optimal) ** 2 / np.sum(iw_optimal ** 2)
                ess[run, i, 5] = np.sum(iw_superoptimal) ** 2 / np.sum(iw_superoptimal ** 2)
                ess[run, i, 6] = np.sum(iw_estimated) ** 2 / np.sum(iw_estimated ** 2)
                ess[run, i, 2] = np.sum(iw_trucnated) ** 2 / np.sum(iw_trucnated ** 2)
                ess[run, i, 1] = np.sum(iw_sn) ** 2 / np.sum(iw_sn ** 2)
                ess[run, i, 7] = n
                ess[run, i, 3] = np.sum(iw_os) ** 2 / np.sum(iw_os ** 2)

                f_iw = np.mean(iw * f_n)
                f_iw_optimal = np.mean(iw_optimal * f_n)
                f_iw_superoptimal = np.mean(iw_superoptimal * f_n)
                f_iw_estimated = np.mean(iw_estimated * f_n)
                f_iw_trucnated = np.mean(iw_trucnated * f_n)
                f_iw_wn = np.mean(iw_sn * f_n)
                f_on_nn = np.mean(f_on_n)  
                f_iw_os = np.mean(iw_os * f_n)
                
                error_iw = np.abs(f_iw - mu_f_e)
                error_optimal = np.abs(f_iw_optimal - mu_f_e)
                error_superoptimal = np.abs(f_iw_superoptimal - mu_f_e)
                error_estimated = np.abs(f_iw_estimated - mu_f_e)
                error_trucnated = np.abs(f_iw_trucnated - mu_f_e)
                error_wn = np.abs(f_iw_wn - mu_f_e)
                error_on = np.abs(f_on_nn - mu_f_e)
                error_os = np.abs(f_iw_os - mu_f_e)

                res[run, i, 0] = error_iw
                res[run, i, 1] = error_wn
                res[run, i, 2] = error_trucnated
                res[run, i, 3] = error_os
                res[run, i, 4] = error_optimal
                res[run, i, 5] = error_superoptimal
                res[run, i, 6] = error_estimated
                res[run, i, 7] = error_on


                lambdas[run, i, 0] = lambda_optimal
                lambdas[run, i, 1] = lambda_superoptimal
                lambdas[run, i, 2] = lambda_estimated

        test = welch_test(res)
        res_mean = np.mean(res, axis=0)
        res_std = np.var(res, axis=0) ** .5
        res_low, res_high = t.interval(0.95, n_runs-1, loc=res_mean, scale=res_std / np.sqrt(n_runs))

        lambdas_mean = np.mean(lambdas, axis=0)
        lambdas_std = np.var(lambdas, axis=0) ** .5
        lambdas_low, lambdas_high = t.interval(0.95, n_runs - 1, loc=lambdas_mean, scale=lambdas_std / np.sqrt(n_runs))

        ess_mean = np.mean(ess, axis=0)
        ess_std = np.var(ess, axis=0) ** .5
        ess_low, ess_high = t.interval(0.95, n_runs - 1, loc=ess_mean, scale=ess_std / np.sqrt(n_runs))

        res_all = np.hstack([np.array(N)[:, None], res_mean, res_std, res_low, res_high])
        algs = ['IS', 'SN-IS', 'IS-Trunc', 'IS-OS', "IS-$\\lambda^*$", "IS-$\\lambda^{**}$", "IS-$\\widehat{\\lambda}$", 'OnPolicy']
        col_names = ['N'] + [alg + h for h in ['_mean', '_std', '_low', '_high'] for alg in algs ]
        np.savetxt('res_%s.csv' % sigma2_e, res_all, delimiter=',', header=','.join(col_names), comments='')

        lambdas_all = np.hstack([np.array(N)[:, None], lambdas_mean, lambdas_std, lambdas_low, lambdas_high])
        algs = ['LambdaOptimal', 'LambdaSuperOptimal', 'LambdaEstimated']
        col_names = ['N'] + [alg + h for h in ['_mean', '_std', '_low', '_high'] for alg in algs ]
        np.savetxt('lambda_%s.csv' % sigma2_e, lambdas_all, delimiter=',', header=','.join(col_names), comments='')

        ess_all = np.hstack([np.array(N)[:, None], ess_mean, ess_std, ess_low, ess_high])
        algs = ['IS', 'SN-IS', 'IS-Trunc', 'IS-OS', "IS-LambdaStar", "IS-LambdaStarStar", "IS-LambdaHat$", 'OnPolicy']
        col_names = ['N'] + [alg + h for h in ['_mean', '_std', '_low', '_high'] for alg in algs]
        np.savetxt('ess_%s.csv' % sigma2_e, ess_all, delimiter=',', header=','.join(col_names), comments='')

        x = PrettyTable([''] + N)
        x.add_row(['IW'] + res_mean[:, 0].tolist())
        x.add_row(['SN-IS'] + res_mean[:, 1].tolist())
        x.add_row(['IS-Trunc'] + res_mean[:, 2].tolist())
        x.add_row(['IS-OS'] + res_mean[:, 3].tolist())
        x.add_row(["IS-$\\lambda^*$"] + res_mean[:, 4].tolist())
        x.add_row(["IS-$\\lambda^{**}$"] + res_mean[:, 5].tolist())
        x.add_row(["IS-$\\widehat{\\lambda}$"] + res_mean[:, 6].tolist())
        x.add_row(['OnPolicy'] + res_mean[:, 7].tolist())
        print(x)

        x = PrettyTable([''] + N)
        x.add_row(['LambdaOptimal'] + lambdas_mean[:, 0].tolist())
        x.add_row(['LambdaSuperOptimal'] + lambdas_mean[:, 1].tolist())
        x.add_row(['LambdaEstimated'] + lambdas_mean[:, 2].tolist())
        print(x)
