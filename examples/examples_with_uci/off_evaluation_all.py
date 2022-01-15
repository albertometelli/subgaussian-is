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

datasets = obp.dataset.uci_datasets.DATASETS
num_samples_list = [float('inf')]
alpha_b_list = [0.8, 0.9]
alpha_e_list = [0.8, 0.85, 0.9, 0.95, 0.99]
n_runs = 5

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for dataset in datasets:
            for alpha_b in alpha_b_list:
                for alpha_e in alpha_e_list:
                    for num_samples in num_samples_list:
                        evaluate_off_policy_estimators_no_main.run(
                            dataset_name=dataset,
                            num_samples=num_samples,
                            n_runs=n_runs,
                            alpha_b=alpha_b,
                            alpha_e=alpha_e,
                            n_jobs=-1,
                        )
