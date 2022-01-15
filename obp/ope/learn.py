# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

# Modifications copyright (C) 2021 Alberto Maria Metelli, Alessio Russo, and Politecnico di Milano. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

from .estimators import BaseOffPolicyEstimator
from obp.types import BanditFeedback
from obp.ope.utils import compute_renyi_divergence

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import optimizers
import itertools

logger = getLogger(__name__)

def learn(loss, init_params, num_train, step_size = 0.001, num_epochs = 10, batch_size = 128):

    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = np.random.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield batch_idx

    batches = data_stream()

    opt_init, opt_update, get_params = optimizers.rmsprop(step_size=step_size)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))


@dataclass
class OffPolicyLearning:
    """Class to conduct off-policy evaluation by multiple off-policy estimators simultaneously.

    Parameters
    -----------
    bandit_feedback: BanditFeedback
        Logged bandit feedback data used for off-policy evaluation.

    ope_estimators: List[BaseOffPolicyEstimator]
        List of OPE estimators used to evaluate the policy value of evaluation policy.
        Estimators must follow the interface of `obp.ope.BaseOffPolicyEstimator`.

    Examples
    ----------

    .. code-block:: python

        # a case for implementing OPE of the BernoulliTS policy
        # using log data generated by the Random policy
        >>> from obp.dataset import OpenBanditDataset
        >>> from obp.policy import BernoulliTS
        >>> from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting as IPW

        # (1) Data loading and preprocessing
        >>> dataset = OpenBanditDataset(behavior_policy='random', campaign='all')
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback()
        >>> bandit_feedback.keys()
        dict_keys(['n_rounds', 'n_actions', 'action', 'position', 'reward', 'pscore', 'context', 'action_context'])

        # (2) Off-Policy Learning
        >>> evaluation_policy = BernoulliTS(
            n_actions=dataset.n_actions,
            len_list=dataset.len_list,
            is_zozotown_prior=True, # replicate the policy in the ZOZOTOWN production
            campaign="all",
            random_state=12345
        )
        >>> action_dist = evaluation_policy.compute_batch_action_dist(
            n_sim=100000, n_rounds=bandit_feedback["n_rounds"]
        )

        # (3) Off-Policy Evaluation
        >>> ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback, ope_estimators=[IPW()])
        >>> estimated_policy_value = ope.estimate_policy_values(action_dist=action_dist)
        >>> estimated_policy_value
        {'ipw': 0.004553...}

        # policy value improvement of BernoulliTS over the Random policy estimated by IPW
        >>> estimated_policy_value_improvement = estimated_policy_value['ipw'] / bandit_feedback['reward'].mean()
        # our OPE procedure suggests that BernoulliTS improves Random by 19.81%
        >>> print(estimated_policy_value_improvement)
        1.198126...

    """

    bandit_feedback: BanditFeedback
    ope_estimators: List[BaseOffPolicyEstimator]

    def __post_init__(self) -> None:
        """Initialize class."""
        for key_ in ["action", "position", "reward", "pscore", "context"]:
            if key_ not in self.bandit_feedback:
                raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")
        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

    def _create_estimator_inputs(
        self, policy: None, estimated_rewards_by_reg_model: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Create input dictionary to estimate policy value by subclasses of `BaseOffPolicyEstimator`"""
        estimator_inputs = {
            input_: self.bandit_feedback[input_]
            for input_ in ["reward", "action", "position", "pscore", "behav_action_dist"]
        }
        estimator_inputs["policy"] = policy
        behav_action_dist = self.bandit_feedback["behav_action_dist"]
        estimator_inputs["d2_Renyi"] = compute_renyi_divergence(action_dist, behav_action_dist)
        estimator_inputs[
            "estimated_rewards_by_reg_model"
        ] = estimated_rewards_by_reg_model

        return estimator_inputs

    def optimize_policy(
        self,
        policy: None,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Estimate policy value of an evaluation policy.

        Parameters
        ------------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When None is given, model-dependent estimators such as DM and DR cannot be used.

        Returns
        ----------
        policy_value_dict: Dict[str, float]
            Dictionary containing estimated policy values by OPE estimators.

        """
        #assert isinstance(action_dist, np.ndarray), "action_dist must be ndarray"
        #assert action_dist.ndim == 3, "action_dist must be 3-dimensional"
        if estimated_rewards_by_reg_model is None:
            logger.warning(
                "`estimated_rewards_by_reg_model` is not given; model dependent estimators such as DM or DR cannot be used."
            )

        policy_value_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            policy=policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_dict[estimator_name] = estimator.optimize_policy(
                **estimator_inputs
            )

        return policy_value_dict

    def estimate_intervals(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate confidence intervals of estimated policy values using a nonparametric bootstrap procedure.

        Parameters
        ------------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        policy_value_interval_dict: Dict[str, Dict[str, float]]
            Dictionary containing confidence intervals of estimated policy value estimated
            using a nonparametric bootstrap procedure.

        """
        assert isinstance(action_dist, np.ndarray), "action_dist must be ndarray"
        assert action_dist.ndim == 3, "action_dist must be 3-dimensional"
        if estimated_rewards_by_reg_model is None:
            logger.warning(
                "`estimated_rewards_by_reg_model` is not given; model dependent estimators such as DM or DR cannot be used."
            )

        policy_value_interval_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_interval_dict[estimator_name] = estimator.estimate_interval(
                **estimator_inputs,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

        return policy_value_interval_dict

    def summarize_off_policy_estimates(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Summarize policy values estimated by OPE estimators and their confidence intervals.

        Parameters
        ------------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        (policy_value_df, policy_value_interval_df): Tuple[DataFrame, DataFrame]
            Estimated policy values and their confidence intervals by OPE estimators.

        """
        assert isinstance(action_dist, np.ndarray), "action_dist must be ndarray"
        assert action_dist.ndim == 3, "action_dist must be 3-dimensional"

        policy_value_df = DataFrame(
            self.estimate_policy_values(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            ),
            index=["estimated_policy_value"],
        )
        policy_value_interval_df = DataFrame(
            self.estimate_intervals(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )
        )

        return policy_value_df.T, policy_value_interval_df.T

    def visualize_off_policy_estimates(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize policy values estimated by OPE estimators.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        is_relative: bool, default=False,
            If True, the method visualizes the estimated policy values of evaluation policy
            relative to the ground-truth policy value of behavior policy.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If 'None' is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        assert isinstance(action_dist, np.ndarray), "action_dist must be ndarray"
        assert action_dist.ndim == 3, "action_dist must be 3-dimensional"
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "fig_dir must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "fig_dir must be a string"
        if estimated_rewards_by_reg_model is None:
            logger.warning(
                "`estimated_rewards_by_reg_model` is not given; model dependent estimators such as DM or DR cannot be used."
            )

        estimated_round_rewards_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_round_rewards_dict[
                estimator_name
            ] = estimator._estimate_round_rewards(**estimator_inputs)
        estimated_round_rewards_df = DataFrame(estimated_round_rewards_dict)
        estimated_round_rewards_df.rename(
            columns={key: key.upper() for key in estimated_round_rewards_dict.keys()},
            inplace=True,
        )
        if is_relative:
            estimated_round_rewards_df /= self.bandit_feedback["reward"].mean()

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=estimated_round_rewards_df,
            ax=ax,
            ci=100 * (1 - alpha),
            n_boot=n_bootstrap_samples,
            seed=random_state,
        )
        plt.xlabel("OPE Estimators", fontsize=25)
        plt.ylabel(
            f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=20
        )
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=25 - 2 * len(self.ope_estimators))

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def evaluate_performance_of_estimators(
        self,
        ground_truth_policy_value: float,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        metric: str = "relative-ee",
    ) -> Dict[str, float]:
        """Evaluate estimation performances of OPE estimators.

        Note
        ------
        Evaluate the estimation performances of OPE estimators by relative estimation error (relative-EE) or squared error (SE):

        .. math ::

            \\text{Relative-EE} (\\hat{V}; \\mathcal{D}) = \\left|  \\frac{\\hat{V}(\\pi; \\mathcal{D}) - V(\\pi)}{V(\\pi)} \\right|,

        .. math ::

            \\text{SE} (\\hat{V}; \\mathcal{D}) = \\left(\\hat{V}(\\pi; \\mathcal{D}) - V(\\pi) \\right)^2,

        where :math:`V({\\pi})` is the ground-truth policy value of the evalation policy :math:`\\pi_e` (often estimated using on-policy estimation).
        :math:`\\hat{V}(\\pi; \\mathcal{D})` is an estimated policy value by an OPE estimator :math:`\\hat{V}` and logged bandit feedback :math:`\\mathcal{D}`.

        Parameters
        ----------
        ground_truth policy value: float
            Ground_truth policy value of an evaluation policy, i.e., :math:`V(\\pi)`.
            With Open Bandit Dataset, in general, we use an on-policy estimate of the policy value as its ground-truth.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        metric: str, default="relative-ee"
            Evaluation metric to evaluate and compare the estimation performance of OPE estimators.
            Must be "relative-ee" or "se".

        Returns
        ----------
        eval_metric_ope_dict: Dict[str, float]
            Dictionary containing evaluation metric for evaluating the estimation performance of OPE estimators.

        """
        # assert isinstance(action_dist, np.ndarray), "action_dist must be ndarray"
        # assert action_dist.ndim == 3, "action_dist must be 3-dimensional"
        assert isinstance(
            ground_truth_policy_value, float
        ), "ground_truth_policy_value must be a float"
        assert metric in [
            "relative-ee",
            "se",
        ], "metric must be either 'relative-ee' or 'se'"
        if estimated_rewards_by_reg_model is None:
            logger.warning(
                "`estimated_rewards_by_reg_model` is not given; model dependent estimators such as DM or DR cannot be used."
            )

        eval_metric_ope_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_policy_value = estimator.estimate_policy_value(**estimator_inputs)
            if metric == "relative-ee":
                relative_ee_ = estimated_policy_value - ground_truth_policy_value
                relative_ee_ /= ground_truth_policy_value
                eval_metric_ope_dict[estimator_name] = np.abs(relative_ee_)
            elif metric == "se":
                se_ = (estimated_policy_value - ground_truth_policy_value) ** 2
                eval_metric_ope_dict[estimator_name] = se_
        return eval_metric_ope_dict

    def summarize_estimators_comparison(
        self,
        ground_truth_policy_value: float,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        metric: str = "relative-ee",
    ) -> DataFrame:
        """Summarize performance comparisons of OPE estimators.

        Parameters
        ----------
        ground_truth policy value: float
            Ground_truth policy value of an evaluation policy, i.e., :math:`V(\\pi)`.
            With Open Bandit Dataset, in general, we use an on-policy estimate of the policy value as ground-truth.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        metric: str, default="relative-ee"
            Evaluation metric to evaluate and compare the estimation performance of OPE estimators.
            Must be either "relative-ee" or "se".

        Returns
        ----------
        eval_metric_ope_df: DataFrame
            Evaluation metric for evaluating the estimation performance of OPE estimators.

        """
        assert isinstance(action_dist, np.ndarray), "action_dist must be ndarray"
        assert action_dist.ndim == 3, "action_dist must be 3-dimensional"
        assert metric in [
            "relative-ee",
            "se",
        ], "metric must be either 'relative-ee' or 'se'"

        eval_metric_ope_df = DataFrame(
            self.evaluate_performance_of_estimators(
                ground_truth_policy_value=ground_truth_policy_value,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                metric=metric,
            ),
            index=[metric],
        )
        return eval_metric_ope_df.T