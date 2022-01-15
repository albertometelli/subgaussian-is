# Copyright (C) 2021 Alberto Maria Metelli, Alessio Russo, and Politecnico di Milano. All rights reserved.
# Licensed under the Apache 2.0 License.

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path = ['.'] + sys.path
import evaluate_off_policy_estimators_no_main
import obp.dataset.uci_datasets
import warnings
from sklearn.exceptions import ConvergenceWarning


datasets = ['letter']
num_samples_list = [30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
alpha_b_list = [0.5, 0.9]
alpha_e_list = [0.99]
n_runs = 10

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for alpha_b in alpha_b_list:
            for alpha_e in alpha_e_list:
                for num_samples in num_samples_list:
                    print(alpha_b, alpha_e, num_samples)
                    evaluate_off_policy_estimators_no_main.run(
                        dataset_name=datasets[0],
                        num_samples=num_samples,
                        n_runs=n_runs,
                        alpha_b=alpha_b,
                        alpha_e=alpha_e,
                        reward_noise=0.,
                        n_jobs=1,
                    )
