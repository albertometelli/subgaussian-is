# Copyright (C) 2021 Alberto Maria Metelli, Alessio Russo, and Politecnico di Milano. All rights reserved.
# Licensed under the Apache 2.0 License.

import sys
sys.path = ['.'] + sys.path
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
)



# hyperparameter for the regression model used in model dependent OPE estimators
with open("./examples/examples_with_uci/conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

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
]



def _initialize_theta(values_from_epoch, size):
    if values_from_epoch == 0:
        theta = np.zeros((size, 1))#np.random.randn(size, 1) * 0.01
        print(theta.shape)
        return theta
    else:
        print(Path(".").resolve().parents)
        file_path = Path(".").resolve() / "logs" / "offpolicy_learning" / "offpolicy_learning.txt"
        f = open(file_path)
        content = f.read()
        index = content.index(f"Epoch number {values_from_epoch}")
        content = content[index:]
        start = content.find("[[")
        end = content.find("]]")
        content = content[start+2:end]
        content = content.replace("\n", "")
        values = content.split("  ")
        params = []
        for item in values:
            nums = item.split(" ")
            [params.append(val) for val in nums]
        params = [item for item in params if item != ""]
        float_params = [float(item) for item in params]
        theta = np.array(float_params)
        theta = np.expand_dims(theta, 1)
        return theta

def process(i, dataset, base_model_for_reg_model, values_from_epoch, batch_size, eta, learning_rate, iterations):

    bandit_feedback = dataset.obtain_batch_bandit_feedback(learning=True, random_state=i)

    regression_model = RegressionModel(
        n_actions=dataset.n_actions,
        len_list=dataset.len_list,
        base_model=base_model_dict[base_model_for_reg_model](
            **hyperparams[base_model_for_reg_model]
        ),
    )
    estimated_rewards_by_reg_model = regression_model.fit_predict(
        context=bandit_feedback["context"],
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        n_folds=3,  # 3-fold cross-fitting
        random_state=i,
    )

    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=ope_estimators,
    )

    # definition of theta
    theta = _initialize_theta(values_from_epoch, dataset.X.shape[1] * dataset.n_actions)

    res = ope.learn_policy(
        dataset=dataset,
        batch_size=batch_size,
        theta=theta,
        eta=eta,
        learning_rate=learning_rate,
        iterations=iterations,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    return res

def run(n_runs=1,
        iterations=100,
        batch_size=100,
        dataset_name="letter",
        num_samples=1000,
        eval_size=0.4,
        alpha_b=0.8,
        reward_noise=0,
        n_jobs=1,
        random_state=12345,
        eta=0.00,
        learning_rate=0.05,
        values_from_epoch=0,
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

    np.random.seed(random_state)

    # synthetic data generator
    # convert the raw classification data into the logged bandit dataset
    base_path = Path(".") / "datasets"
    X, y, n_actions = load_uci_dataset(dataset_name, base_path, num_samples)

    X = (X - np.min(X, axis=0)[None, :]) / (np.max(X, axis=0)[None, :] - np.min(X, axis=0)[None, :])

    dataset = MultiClassToBanditReduction(
        X=X,
        y=y,
        reward_noise=reward_noise,
        n_actions=n_actions,
        base_classifier_b=LogisticRegression(solver='lbfgs', random_state=random_state),  #solver='lbfgs'
        alpha_b=alpha_b,
        dataset_name=dataset_name,
    )
    dataset.split_train_eval(eval_size=eval_size, random_state=random_state)

    processed = Parallel(
        backend="multiprocessing",
        n_jobs=n_jobs,
        verbose=50,
    )([delayed(process)(i, dataset, base_model_for_reg_model, values_from_epoch, batch_size, eta, learning_rate, iterations) for i in np.arange(n_runs)])
    res_dict = {est.estimator_name: dict() for est in ope_estimators}
    res_dict_train = {est.estimator_name: [] for est in ope_estimators}
    res_dict_eval = {est.estimator_name: [] for est in ope_estimators}

    for i, res_i in enumerate(processed):
        for (
                estimator_name,
                res_,
        ) in res_i.items():
            res_dict[estimator_name][i] = res_
            res_dict_train[estimator_name].append(res_['history_train'])
            res_dict_eval[estimator_name].append(res_['history_eval'])

    df = DataFrame()
    for k in res_dict_train.keys():
        histories_train = np.array(res_dict_train[k])
        histories_eval = np.array(res_dict_eval[k])
        histories_train_mean = np.mean(histories_train, axis=0)
        histories_eval_mean = np.mean(histories_eval, axis=0)
        histories_train_std = np.var(histories_train, axis=0) ** .5
        histories_eval_std = np.var(histories_eval, axis=0) ** .5
        df[k + '_train_mean'] = histories_train_mean
        df[k + '_train_std'] = histories_train_std
        df[k + '_eval_mean'] = histories_eval_mean
        df[k + '_eval_std'] = histories_eval_std

    df.to_csv("%s_res_learning_eta.csv" % dataset_name)

    #print(df)


    '''
    import pickle
    f = open("%d.pkl" % dataset_name, "wb")
    pickle.dump(res_dict, f)
    f.close()
    '''
    #res_df = DataFrame(res_dict).describe().T.round(6)
    #print([(r, res_dict[r][0]['ground_truth']) for r in res_dict.keys()])

    '''
    for i in range(0, n_jobs):
        process(i, dataset, iterations, batch_size, eta, learning_rate, lambda_, values_from_epoch)
    '''
