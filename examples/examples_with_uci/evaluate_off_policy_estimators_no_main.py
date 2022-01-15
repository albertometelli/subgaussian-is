# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

# Modifications copyright (C) 2021 Alberto Maria Metelli, Alessio Russo, and Politecnico di Milano. All rights reserved.
# Licensed under the Apache 2.0 License.

import sys
import os
import argparse
import yaml
from pathlib import Path

import numpy as np
from pandas import DataFrame
from joblib import Parallel, delayed
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from obp.dataset import MultiClassToBanditReduction
from obp.dataset.uci_datasets import load_uci_dataset


from obp.policy import IPWLearner
from obp.ope import (
    DoublyRobust,
    DirectMethod,
    RegressionModel,
    OffPolicyEvaluation,
    TransformedImportanceWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    TransformedDoublyRobust,
    SelfNormalizedDoublyRobust,
    DoublyRobustWithShrinkage,
    DoublyRobustWithShrinkageOptimal,
    InverseProbabilityWeightingShrinkageOptimal,
    TruncatedImportanceWeighting,
    TruncatedDoublyRobust,
    SwitchDoublyRobustOptimal,
)


# hyperparameter for the regression model used in model dependent OPE estimators
with open("./examples/examples_with_uci/conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

# compared OPE estimators
ope_estimators = [
    DoublyRobustWithShrinkageOptimal(estimator_name='dr-os-optimal'),
    SelfNormalizedDoublyRobust(),
    DirectMethod(),
    SelfNormalizedInverseProbabilityWeighting(),
    TransformedImportanceWeighting(lambda_='optimal', estimator_name='transf_ipw (lambda=optimal)'),
    TransformedImportanceWeighting(lambda_=0.0, estimator_name='iw'),
    TransformedDoublyRobust(lambda_='optimal', estimator_name='dr-transf_ipw (lambda=optimal)'),
    TransformedDoublyRobust(lambda_=0.0, estimator_name='dr-iw'),
    InverseProbabilityWeightingShrinkageOptimal(estimator_name="is-os-optimal"),
    TransformedImportanceWeighting(lambda_="estimated", estimator_name='transf_ipw (lambda=estimated)'),
    TransformedDoublyRobust(lambda_="estimated", estimator_name='dr-transf_ipw (lambda=estimated)'),
    TruncatedDoublyRobust(),
    TruncatedImportanceWeighting(),
    SwitchDoublyRobustOptimal(),
]
def process(i, dataset_name, num_samples, reward_noise, alpha_b, eval_size, base_model_for_evaluation_policy,
            base_model_for_reg_model, alpha_e, random_state):
    np.random.seed(i)

    # synthetic data generator
    # convert the raw classification data into the logged bandit dataset
    base_path = Path(".") / "datasets"
    X, y, n_actions = load_uci_dataset(dataset_name, base_path, num_samples)

    dataset = MultiClassToBanditReduction(
        X=X,
        y=y,
        reward_noise=reward_noise,
        n_actions=n_actions,
        base_classifier_b=LogisticRegression(solver='lbfgs', random_state=random_state),  # solver='lbfgs'
        alpha_b=alpha_b,
        dataset_name=dataset_name,
    )
    dataset.split_train_eval(eval_size=eval_size, random_state=random_state)

    bandit_feedback = dataset.obtain_batch_bandit_feedback(random_state=i)

    # predict the action decisions for the test set of the synthetic logged bandit feedback
    action_dist = dataset.obtain_action_dist_by_eval_policy(
        base_classifier_e=base_model_dict[base_model_for_evaluation_policy](
            **hyperparams[base_model_for_evaluation_policy]
        ),
        alpha_e=alpha_e,
    )
    # estimate the ground-truth policy values of the evaluation policy
    # using the full expected reward contained in the test set of synthetic bandit feedback
    ground_truth = dataset.calc_ground_truth_policy_value(action_dist=action_dist)

    # estimate the mean reward function of the test set of synthetic bandit feedback with ML model
    regression_model = RegressionModel(
        n_actions=dataset.n_actions,
        len_list=dataset.len_list,
        base_model=base_model_dict[base_model_for_reg_model](
            **hyperparams[base_model_for_reg_model]
        ),
    )

    if np.unique(bandit_feedback["reward"]).shape[0] == 1:
        estimated_rewards_by_reg_model = np.ones(
            (dataset.n_rounds, dataset.n_actions, 1)
        )
    else:

        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=bandit_feedback["context"],
            action=bandit_feedback["action"],
            reward=bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )
    # evaluate estimators' performances using relative estimation error (relative-ee)
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=ope_estimators,
    )
    relative_ee_i = ope.evaluate_performance_of_estimators(
        ground_truth_policy_value=ground_truth,
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    return relative_ee_i

def run(n_runs=2,
        dataset_name="letter",
        num_samples=1000,
        eval_size=0.7,
        alpha_b=0.8,
        alpha_e=0.9,
        reward_noise=0,
        n_jobs=1,
        random_state=12345,
        base_model_for_evaluation_policy="logistic_regression",
        base_model_for_reg_model="logistic_regression"
        ):

    assert(base_model_for_evaluation_policy=="logistic_regression" or
            base_model_for_evaluation_policy=="lightgbm" or
            base_model_for_evaluation_policy=="random_forest"
    ), f"base ML model for evaluation policy, logistic_regression, random_forest or lightgbm."

    assert(base_model_for_reg_model=="logistic_regression" or
            base_model_for_reg_model=="lightgbm" or
            base_model_for_reg_model=="random_forest"
    ), f"base ML model for evaluation policy, logistic_regression, random_forest or lightgbm."




    processed = Parallel(
        backend="multiprocessing",
        n_jobs=n_jobs,
        verbose=50,
    )([delayed(process)(i, dataset_name, num_samples, reward_noise, alpha_b, eval_size, base_model_for_evaluation_policy,
                        base_model_for_reg_model, alpha_e, random_state) for i in np.arange(n_runs)])
    relative_ee_dict = {est.estimator_name: dict() for est in ope_estimators}
    for i, relative_ee_i in enumerate(processed):
        for (
            estimator_name,
            relative_ee_,
        ) in relative_ee_i.items():
            relative_ee_dict[estimator_name][i] = relative_ee_
    relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)

    s = f"num_samples= {num_samples}, n_runs={n_runs}, alpha_b={alpha_b}, alpha_e={alpha_e}, reward_noise={reward_noise}, random_state={random_state}\n"
    s = s + "=" * 45 + "\n"
    s = s + str(relative_ee_df[["mean", "std"]]) + "\n"
    s = s + "=" * 45 + "\n\n\n"

    # save results of the evaluation of off-policy estimators in './logs' directory.

    log_path = os.path.join(os.getcwd(), "logs")
    dataset_dir = os.path.join(log_path, dataset_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    log_name = f"test_alpha_b={alpha_b}_alpha_e={alpha_e}_reward_noise={reward_noise}.txt"
    file_path = os.path.join(dataset_dir, log_name)
    file = open(file_path, "a+")
    file.write(s)
    file.close()
    #relative_ee_df.to_csv(log_path / ("alpha_b=%s_alpha_e=%s.csv" % (alpha_b, alpha_e)))
