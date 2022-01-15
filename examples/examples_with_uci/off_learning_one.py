# Copyright (C) 2021 Alberto Maria Metelli, Alessio Russo, and Politecnico di Milano. All rights reserved.
# Licensed under the Apache 2.0 License.

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path = ['.'] + sys.path
import learn_off_policy_estimators_no_main
from obp.dataset.uci_datasets import DATASETS

dataset = "ecoli"
num_samples = 20000
alpha_b = 0.
iterations = 1000
batch_size = 32
eta = 0.1
learning_rate = 0.05

if __name__ == "__main__":

    for dataset in DATASETS:
        print(dataset)
        learn_off_policy_estimators_no_main.run(
            n_runs=10,
            dataset_name=dataset,
            num_samples=num_samples,
            batch_size=batch_size,
            iterations=iterations,
            alpha_b=alpha_b,
            reward_noise=0.0,
            eta=eta,
            learning_rate=learning_rate,
            values_from_epoch=0,
            n_jobs=1,
        )

