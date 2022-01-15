# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

# Modifications copyright (C) 2021 Alberto Maria Metelli, Alessio Russo, and Politecnico di Milano. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from ..utils import estimate_confidence_interval_by_bootstrap
from .utils import find_optimal_lambda, estimate_lambda

from ..utils import estimate_confidence_interval_by_bootstrap
from .utils import find_optimal_lambda
import math

from scipy.optimize import minimize

def compute_estimator_gradient_over_actions(
        context: np.ndarray,  # this represents only one sample
        n_actions: int,
        theta: np.ndarray,
) -> np.ndarray:
    context_size = context.shape[0]
    gradient_estimator_over_actions = np.zeros((n_actions * context_size, n_actions), dtype=float)
    phi = np.zeros((n_actions * context_size, n_actions), dtype=float)
    for j in range(0, n_actions):
        phi[j * context_size:(j + 1) * context_size, j] = context
    general_numerator = np.exp(np.dot(theta.T, phi))
    softmax_denominator = np.sum(general_numerator)
    prob_context = general_numerator / softmax_denominator
    for i in range(0, n_actions):
        curr_action = i
        phi_curr_action = phi[:, curr_action]
        phi_curr_action_full = np.tile(phi_curr_action, (n_actions, 1)).T
        diff = phi_curr_action_full - phi
        weighted_diff = prob_context * diff
        sum_of_weighted_diff = np.sum(weighted_diff, axis=1)
        gradient = prob_context[0, curr_action] * sum_of_weighted_diff
        gradient_estimator_over_actions[:, i] = gradient

    return gradient_estimator_over_actions, prob_context


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

    @abstractmethod
    def _estimate_round_rewards(self) -> np.ndarray:
        """Estimate rewards for each round."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate policy value of an evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure."""
        raise NotImplementedError


@dataclass
class ReplayMethod(BaseOffPolicyEstimator):
    """Estimate the policy value by Relpay Method (RM).

    Note
    -------
    Replay Method (RM) estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{RM}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{\\mathcal{D}}[\\mathbb{I} \\{ \\pi_e (x_t) = a_t \\} r_t ]}{\\mathbb{E}_{\\mathcal{D}}[\\mathbb{I} \\{ \\pi_e (x_t) = a_t \\}]},

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`\\pi_e: \\mathcal{X} \\rightarrow \\mathcal{A}` is the function
    representing action choices by the evaluation policy realized during offline bandit simulation.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.

    Parameters
    ----------
    estimator_name: str, default='rm'.
        Name of off-policy estimator.

    References
    ------------
    Lihong Li, Wei Chu, John Langford, and Xuanhui Wang.
    "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms.", 2011.

    """

    estimator_name: str = "rm"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the Replay Method for each round.

        """
        action_match = np.array(
            action_dist[np.arange(action.shape[0]), action, position] == 1
        )
        estimated_rewards = np.zeros_like(action_match)
        if action_match.sum() > 0.0:
            estimated_rewards = action_match * reward / action_match.mean()
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        action_dist: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class InverseProbabilityWeighting(BaseOffPolicyEstimator):
    """Estimate the policy value by Inverse Probability Weighting (IPW).

    Note
    -------
    Inverse Probability Weighting (IPW) estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{IPW}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{\\mathcal{D}} [ w(x_t,a_t) r_t],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.

    IPW re-weights the rewards by the ratio of the evaluation policy and behavior policy (importance weight).
    When the behavior policy is known, IPW is unbiased and consistent for the true policy value.
    However, it can have a large variance, especially when the evaluation policy significantly deviates from the behavior policy.

    Parameters
    ------------
    estimator_name: str, default='ipw'.
        Name of off-policy estimator.

    References
    ------------
    Alex Strehl, John Langford, Lihong Li, and Sham M Kakade.
    "Learning from Logged Implicit Exploration Data"., 2010.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "ipw"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by IPW for each round.

        """
        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        return reward * iw

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
        ).mean()

    def obtain_loss(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: None,
        **kwargs,
    ) -> np.ndarray:

        iw = action_dist / pscore
        return reward * iw

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities
            by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

@dataclass
class SelfNormalizedInverseProbabilityWeighting(InverseProbabilityWeighting):
    """Estimate the policy value by Self-Normalized Inverse Probability Weighting (SNIPW).

    Note
    -------
    Self-Normalized Inverse Probability Weighting (SNIPW) estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SNIPW}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{\\mathcal{D}} [w(x_t,a_t) r_t]}{ \\mathbb{E}_{\\mathcal{D}} [w(x_t,a_t)]},

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.

    SNIPW re-weights the observed rewards by the self-normalized importance weihgt.
    This estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and increases the stability in some senses.
    See the references for the detailed discussions.

    Parameters
    ----------
    estimator_name: str, default='snipw'.
        Name of off-policy estimator.

    References
    ----------
    Adith Swaminathan and Thorsten Joachims.
    "The Self-normalized Estimator for Counterfactual Learning.", 2015.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "snipw"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the SNIPW estimator for each round.

        """
        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        return reward * iw / iw.mean()

    def estimate_optimization_function_gradient_(
                self,
                context: np.ndarray,
                reward: np.ndarray,
                action: np.ndarray,
                pscore: np.ndarray,
                behav_action_dist: np.ndarray,
                eta: float,
                theta: np.ndarray,  #size of theta is 16 * n_actions
                **kwargs,
            ) -> np.ndarray:

        num_samples = context.shape[0]
        context_size = context.shape[1]
        n_actions = behav_action_dist.shape[1]
        estimator_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)
        renyi_div_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)

        iw = np.zeros((num_samples, ))

        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            curr_behav_dist = behav_action_dist[i]
            curr_reward = reward[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            renyi_div_gradient_term = np.sum(2*prob_context*gradient_over_actions/curr_behav_dist.T, axis=1)
            renyi_div_gradient_terms[:, i] = renyi_div_gradient_term
            iw[i] = prob_context[:, curr_action] / curr_score

            estimator_gradient_term = gradient_over_actions[:, curr_action] / curr_score
            estimator_gradient_terms[:, i] = estimator_gradient_term

        optimization_function_gradient = (np.sum(estimator_gradient_terms * reward[None, :], axis=1) * np.sum(iw)  \
            - np.sum(iw * reward) * np.sum(estimator_gradient_terms, axis=1)) / np.sum(iw ** 2)

        optimization_function_gradient = optimization_function_gradient - eta * np.mean(renyi_div_gradient_terms, axis=1)
        return optimization_function_gradient


@dataclass
class DirectMethod(BaseOffPolicyEstimator):
    """Estimate the policy value by Direct Method (DM).

    Note
    -------
    DM first learns a supervised machine learning model, such as ridge regression and gradient boosting,
    to estimate the mean reward function (:math:`q(x,a) = \\mathbb{E}[r|x,a]`).
    It then uses it to estimate the policy value as follows.

    .. math::

        \\hat{V}_{\\mathrm{DM}} (\\pi_e; \\mathcal{D}, \\hat{q})
        &:= \\mathbb{E}_{\\mathcal{D}} \\left[ \\sum_{a \\in \\mathcal{A}} \\hat{q} (x_t,a) \\pi_e(a|x_t) \\right],    \\\\
        & =  \\mathbb{E}_{\\mathcal{D}}[\\hat{q} (x_t,\\pi_e)],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`, which supports several fitting methods specific to OPE.

    If the regression model (:math:`\\hat{q}`) is a good approximation to the true mean reward function,
    this estimator accurately estimates the policy value of the evaluation policy.
    If the regression function fails to approximate the mean reward function well,
    however, the final estimator is no longer consistent.

    Parameters
    ----------
    estimator_name: str, default='dm'.
        Name of off-policy estimator.

    References
    ----------
    Alina Beygelzimer and John Langford.
    "The offset tree for learning with partial labels.", 2009.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "dm"

    def estimate_optimization_function_gradient_(
                self,
                context: np.ndarray,
                eta: float,
                theta: np.ndarray,  #size of theta is 16 * n_actions
                estimated_rewards_by_reg_model: np.ndarray,
                **kwargs,
            ) -> np.ndarray:

        num_samples = context.shape[0]
        context_size = context.shape[1]
        n_actions = estimated_rewards_by_reg_model.shape[1]
        estimator_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)

        for i in range(0, num_samples):
            curr_context = context[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            estimator_gradient_term = np.sum(gradient_over_actions * estimated_rewards_by_reg_model[i, :, 0][None, :], axis=1)
            estimator_gradient_terms[:, i] = estimator_gradient_term

        optimization_function_gradient = np.mean(estimator_gradient_terms, axis=1)

        regularizer = 0.

        optimization_function_gradient = optimization_function_gradient - eta * regularizer
        return optimization_function_gradient

    def _estimate_round_rewards(
        self,
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DM estimator for each round.

        """
        n_rounds = position.shape[0]
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        return np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )

    def estimate_policy_value(
        self,
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        return self._estimate_round_rewards(
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

@dataclass
class TransformedImportanceWeighting(InverseProbabilityWeighting):

    lambda_: Union[str, float] = 'optimal'
    significance: float = 0.1
    estimator_name: str = "transf_ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        if isinstance(self.lambda_, str) and self.lambda_ in ['optimal', 'super-optimal', 'estimated']:
            """ if lambda value is optimal, its optimal value will be computed through the compute_lambda_value function"""
            return
        elif isinstance(self.lambda_, float):
            assert self.lambda_ >= 0.0, f"lambda hyperparameter must be larger than zero, but {self.lambda_} is given"
            assert self.lambda_ <= 1.0, f"lambda hyperparameter must be smaller than one, but {self.lambda_} is given"
        else:
            raise ValueError

    def _compute_lambda_value(
                self,
                action_dist: np.ndarray,
                d2_Renyi: float,
                weights: Optional[np.array] = None,
        ) -> float:
        if isinstance(self.lambda_, str):

            n = action_dist.shape[0]

            if self.lambda_ == 'optimal':
                lambda_ = np.sqrt(np.log(1 / self.significance) / (3 * d2_Renyi * n))
            elif self.lambda_ == 'super-optimal':
                lambda_ = find_optimal_lambda(d2_Renyi, n, self.significance)
            elif self.lambda_ == 'estimated':
                lambda_ = estimate_lambda(weights, n, self.significance)


            #print('Lambda %s' % lambda_)
            return lambda_
        else:
            #print('Lambda %s' % self.lambda_)
            return self.lambda_

    def estimate_optimization_function_gradient_(
                self,
                context: np.ndarray,
                reward: np.ndarray,
                action: np.ndarray,
                pscore: np.ndarray,
                behav_action_dist: np.ndarray,
                #lambda_: float,
                eta: float,
                theta: np.ndarray,  #size of theta is 16 * n_actions
                d2Renyi: float,
                **kwargs,
            ) -> np.ndarray:

        #lambda_ = self._compute_lambda_value(
        #    action_dist=action, #whatever, just needed for n
        #    d2_Renyi=d2Renyi,
        #)

        num_samples = context.shape[0]
        context_size = context.shape[1]
        n_actions = behav_action_dist.shape[1]
        estimator_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)
        renyi_div_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)

        iw = np.zeros((num_samples, ))

        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            curr_behav_dist = behav_action_dist[i]
            curr_reward = reward[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            #renyi_div_gradient_term = np.sum(2*prob_context*gradient_over_actions/curr_behav_dist.T, axis=1)
            #renyi_div_gradient_terms[:, i] = renyi_div_gradient_term
            iw[i] = prob_context[:, curr_action] / curr_score

        lambda_ = self._compute_lambda_value(
            action_dist=action, #whatever, just needed for n
            d2_Renyi=d2Renyi,
            weights=iw,
        )


        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            curr_behav_dist = behav_action_dist[i]
            curr_reward = reward[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            renyi_div_gradient_term = np.sum(2*prob_context*gradient_over_actions/curr_behav_dist.T, axis=1)
            renyi_div_gradient_terms[:, i] = renyi_div_gradient_term
            iw[i] = prob_context[:, curr_action] / curr_score


            numerator_estimator_gradient_term = (1 - lambda_)*curr_score*gradient_over_actions[:, curr_action]
            denominator_estimator_gradient_term = np.power((1 - lambda_)*curr_score + lambda_*prob_context[:, curr_action], 2)
            estimator_gradient_term = numerator_estimator_gradient_term/denominator_estimator_gradient_term
            w = prob_context[:, curr_action] / curr_score
            wt = w / ((1 - lambda_) + lambda_ * w)
            estimator_gradient_terms[:, i] = estimator_gradient_term * curr_reward - 2 * eta * wt * estimator_gradient_term

        optimization_function_gradient = np.average(estimator_gradient_terms, axis=1)

        return optimization_function_gradient

    def _estimate_round_rewards(
                self,
                reward: np.ndarray,
                action: np.ndarray,
                position: np.ndarray,
                pscore: np.ndarray,
                action_dist: np.ndarray,
                d2_Renyi: float,
                **kwargs,
        ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the SNIPW estimator for each round.

        """

        #print("lambda value is " + str(lambda_))
        n_rounds = action.shape[0]
        evaluation_dist = action_dist[np.arange(n_rounds), action, position]

        weights = evaluation_dist / pscore
        lambda_ = self._compute_lambda_value(
            action_dist=action_dist,
            d2_Renyi=d2_Renyi,
            weights=weights,
        )

        transformed_iw = evaluation_dist / ((1 - lambda_) * pscore + lambda_ * evaluation_dist)
        estimated_rewards = reward*transformed_iw
        return estimated_rewards

    def estimate_policy_value(
            self,
            reward: np.ndarray,
            action: np.ndarray,
            position: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            d2_Renyi: float,
            **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            d2_Renyi=d2_Renyi,
        ).mean()


@dataclass
class TruncatedImportanceWeighting(InverseProbabilityWeighting):

    threshold_: Union[str, float] = 'optimal'
    significance: float = 0.1
    estimator_name: str = "truncated_ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        if isinstance(self.threshold_, str) and self.threshold_ == 'optimal':
            """ if threshold value is -1, the optimal value of threshold will be computed through the compute_threshold_value function"""
            return
        else:
            assert self.threshold_ >= 0.0, f"threshold hyperparameter must be larger than zero, but {self.threshold_} is given"

    def _compute_threshold_value(
            self,
            action_dist: np.ndarray,
            d2_Renyi: float,
    ) -> float:
        if self.threshold_ == 'optimal':
            threshold_ = np.sqrt((3 * d2_Renyi*action_dist.shape[0])/(2 * np.log(1 / self.significance)))
            return threshold_
        else:
            return self.threshold_

    def _estimate_round_rewards(
            self,
            reward: np.ndarray,
            action: np.ndarray,
            position: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            d2_Renyi: float,
            **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the SNIPW estimator for each round.

        """
        threshold_ = self._compute_threshold_value(
            action_dist=action_dist,
            d2_Renyi=d2_Renyi,
        )
        #print("threshold value is " + str(threshold_))
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        transformed_iw = np.minimum(threshold_, iw)
        estimated_rewards = reward*transformed_iw
        return estimated_rewards

    def estimate_policy_value(
            self,
            reward: np.ndarray,
            action: np.ndarray,
            position: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            d2_Renyi: float,
            **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        d2_Renyi: computed value of exponentiated Renyi divergence
        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.
        """
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            d2_Renyi=d2_Renyi,
        ).mean()


@dataclass
class DoublyRobust(InverseProbabilityWeighting):
    """Estimate the policy value by Doubly Robust (DR).

    Note
    -------
    Similar to DM, DR first learns a supervised machine learning model, such as ridge regression and gradient boosting,
    to estimate the mean reward function (:math:`q(x,a) = \\mathbb{E}[r|x,a]`).
    It then uses it to estimate the policy value as follows.

    .. math::

        \\hat{V}_{\\mathrm{DR}} (\\pi_e; \\mathcal{D}, \\hat{q})
        := \\mathbb{E}_{\\mathcal{D}}[\\hat{q}(x_t,\\pi_e) +  w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.

    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    DR mimics IPW to use a weighted version of rewards, but DR also uses the estimated mean reward
    function (the regression model) as a control variate to decrease the variance.
    It preserves the consistency of IPW if either the importance weight or
    the mean reward estimator is accurate (a property called double robustness).
    Moreover, DR is semiparametric efficient when the mean reward estimator is correctly specified.

    Parameters
    ----------
    estimator_name: str, default='dr'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    """

    estimator_name: str = "dr"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DR estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += iw * (reward - q_hat_factual)
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value by the DR estimator.

        """
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SelfNormalizedDoublyRobust(DoublyRobust):
    """Estimate the policy value by Self-Normalized Doubly Robust (SNDR).

    Note
    -------
    Self-Normalized Doubly Robust estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SNDR}} (\\pi_e; \\mathcal{D}, \\hat{q}) :=
        \\mathbb{E}_{\\mathcal{D}} \\left[\\hat{q}(x_t,\\pi_e) +  \\frac{w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t))}{\\mathbb{E}_{\\mathcal{D}}[ w(x_t,a_t) ]} \\right],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    Similar to Self-Normalized Inverse Probability Weighting, SNDR estimator applies the self-normalized importance weighting technique to
    increase the stability of the original Doubly Robust estimator.

    Parameters
    ----------
    estimator_name: str, default='sndr'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "sndr"

    def estimate_optimization_function_gradient_(
            self,
            context: np.ndarray,
            reward: np.ndarray,
            action: np.ndarray,
            pscore: np.ndarray,
            behav_action_dist: np.ndarray,
            # lambda_: float,
            eta: float,
            theta: np.ndarray,  # size of theta is 16 * n_actions
            d2Renyi: float,
            estimated_rewards_by_reg_model: np.ndarray,
            **kwargs,
    ) -> np.ndarray:

        num_samples = context.shape[0]
        context_size = context.shape[1]
        n_actions = behav_action_dist.shape[1]
        estimator_gradient_terms = np.zeros((n_actions * context_size, num_samples), dtype=float)
        estimator_gradient_terms2 = np.zeros((n_actions * context_size, num_samples), dtype=float)
        renyi_div_gradient_terms = np.zeros((n_actions * context_size, num_samples), dtype=float)

        iw = np.zeros((num_samples,))

        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(num_samples), action, 0
        ]

        diff = reward - q_hat_factual

        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            curr_behav_dist = behav_action_dist[i]
            curr_reward = reward[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions,
                                                                                          theta)
            renyi_div_gradient_term = np.sum(2 * prob_context * gradient_over_actions / curr_behav_dist.T, axis=1)
            renyi_div_gradient_terms[:, i] = renyi_div_gradient_term
            iw[i] = prob_context[:, curr_action] / curr_score

            estimator_gradient_term = gradient_over_actions[:, curr_action] / curr_score

            estimator_gradient_term2 = np.sum(gradient_over_actions * estimated_rewards_by_reg_model[i, :, 0][None, :],
                                              axis=1)
            estimator_gradient_terms2[:, i] = estimator_gradient_term2
            estimator_gradient_terms[:, i] = estimator_gradient_term

        optimization_function_gradient = (np.sum(estimator_gradient_terms * diff[None, :], axis=1) * np.sum(iw)\
                            - np.sum(iw * diff) * np.sum(estimator_gradient_terms, axis=1)) / np.sum(iw ** 2)\
                            + np.mean(estimator_gradient_terms2, axis=1) - eta * np.mean(renyi_div_gradient_terms)

        return optimization_function_gradient

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the SNDR estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        estimated_rewards += iw * (reward - q_hat_factual) / iw.mean()
        return estimated_rewards


@dataclass
class SwitchInverseProbabilityWeighting(DoublyRobust):
    """Estimate the policy value by Switch Inverse Probability Weighting (Switch-IPW).

    Note
    -------
    Switch-IPW aims to reduce the variance of the IPW estimator by using direct method
    when the importance weight is large. This estimator estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        & \\hat{V}_{\\mathrm{SwitchIPW}} (\\pi_e; \\mathcal{D}, \\tau) \\\\
        & := \\mathbb{E}_{\\mathcal{D}} \\left[ \\sum_{a \\in \\mathcal{A}} \\hat{q} (x_t, a) \\pi_e (a|x_t) \\mathbb{I} \\{ w(x_t, a) > \\tau \\}
         + w(x_t,a_t) r_t \\mathbb{I} \\{ w(x_t,a_t) \\le \\tau \\} \\right],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\tau (\\ge 0)` is a switching hyperparameter, which decides the threshold for the importance weight.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    Parameters
    ----------
    tau: float, default=1
        Switching hyperparameter. When importance weight is larger than this parameter, the DM estimator is applied, otherwise the IPW estimator is applied.
        This hyperparameter should be larger than 1., otherwise it is meaningless.

    estimator_name: str, default='switch-ipw'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    """

    tau: float = 1
    estimator_name: str = "switch-ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert (
            self.tau >= 0.0
        ), f"switching hyperparameter should be larger than 1, but {self.tau} is given"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the Switch-IPW estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        switch_indicator = np.array(iw <= self.tau, dtype=int)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = (1 - switch_indicator) * np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += switch_indicator * iw * reward
        return estimated_rewards


@dataclass
class SwitchDoublyRobust(DoublyRobust):
    """Estimate the policy value by Switch Doubly Robust (Switch-DR).

    Note
    -------
    Switch-DR aims to reduce the variance of the DR estimator by using direct method
    when the importance weight is large. This estimator estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SwitchDR}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\tau)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t)) \\mathbb{I} \\{ w(x_t,a_t) \\le \\tau \\}],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\tau (\\ge 0)` is a switching hyperparameter, which decides the threshold for the importance weight.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    Parameters
    ----------
    tau: float, default=1
        Switching hyperparameter. When importance weight is larger than this parameter, the DM estimator is applied, otherwise the DR estimator is applied.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='switch-dr'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    """

    tau: float = 1
    estimator_name: str = "switch-dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert (
            self.tau >= 0.0
        ), f"switching hyperparameter must be larger than or equal to zero, but {self.tau} is given"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the Switch-DR estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        switch_indicator = np.array(iw <= self.tau, dtype=int)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += switch_indicator * iw * (reward - q_hat_factual)
        return estimated_rewards


@dataclass
class DoublyRobustWithShrinkage(DoublyRobust):
    """Estimate the policy value by Doubly Robust with optimistic shrinkage (DRos).

    Note
    ------
    DR with (optimistic) shrinkage replaces the importance weight in the original DR estimator with a new weight mapping
    found by directly optimizing sharp bounds on the resulting MSE.

    .. math::

        \\hat{V}_{\\mathrm{DRos}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w_o(x_t,a_t;\\lambda) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    :math:`w_{o} (x_t,a_t;\\lambda)` is a new weight by the shrinkage technique which is defined as

    .. math::

        w_{o} (x_t,a_t;\\lambda) := \\frac{\\lambda}{w^2(x_t,a_t) + \\lambda} w(x_t,a_t).

    When :math:`\\lambda=0`, we have :math:`w_{o} (x,a;\\lambda)=0` corresponding to the DM estimator.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, :math:`w_{o} (x,a;\\lambda)` increases and in the limit becomes equal to
    the original importance weight, corresponding to the standard DR estimator.


    Parameters
    ----------
    lambda_: float
        Shrinkage hyperparameter.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='dr-os'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambda_: float = 0.0
    estimator_name: str = "dr-os"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert (
            self.lambda_ >= 0.0
        ), f"shrinkage hyperparameter must be larger than or equal to zero, but {self.lambda_} is given"

    def estimate_optimization_function_gradient_(
                self,
                context: np.ndarray,
                reward: np.ndarray,
                action: np.ndarray,
                pscore: np.ndarray,
                behav_action_dist: np.ndarray,
                #lambda_: float,
                eta: float,
                theta: np.ndarray,  #size of theta is 16 * n_actions
                d2Renyi: float,
                estimated_rewards_by_reg_model: np.ndarray,
                **kwargs,
            ) -> np.ndarray:


        num_samples = context.shape[0]
        context_size = context.shape[1]
        n_actions = behav_action_dist.shape[1]
        estimator_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)
        renyi_div_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)

        iw = np.zeros((num_samples, ))

        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(num_samples), action, 0
        ]

        diff = reward - q_hat_factual

        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            curr_behav_dist = behav_action_dist[i]
            curr_reward = reward[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            renyi_div_gradient_term = np.sum(2*prob_context*gradient_over_actions/curr_behav_dist.T, axis=1)
            renyi_div_gradient_terms[:, i] = renyi_div_gradient_term
            iw[i] = prob_context[:, curr_action] / curr_score

            w = prob_context[:, curr_action] / curr_score
            deriv_w = self.lambda_ * (self.lambda_ - w ** 2) / (self.lambda_ + w ** 2) ** 2

            estimator_gradient_term = deriv_w / curr_score * gradient_over_actions[:, curr_action]
            estimator_gradient_term2 = np.sum(gradient_over_actions * estimated_rewards_by_reg_model[i, :, 0][None, :], axis=1)

            estimator_gradient_terms[:, i] = estimator_gradient_term2 + estimator_gradient_term * diff[i] - eta * renyi_div_gradient_term

        # optimization_function_gradient = np.average(estimator_gradient_terms - eta * renyi_div_gradient_terms, axis=1)
        optimization_function_gradient = np.average(estimator_gradient_terms, axis=1)

        return optimization_function_gradient

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DRos estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        shrinkage_weight = (self.lambda_ * iw) / (iw ** 2 + self.lambda_)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += shrinkage_weight * (reward - q_hat_factual)
        return estimated_rewards



@dataclass
class TransformedDoublyRobust(DoublyRobust):
    """Estimate the policy value by Doubly Robust with optimistic shrinkage (DRos).

    Note
    ------
    DR with (optimistic) shrinkage replaces the importance weight in the original DR estimator with a new weight mapping
    found by directly optimizing sharp bounds on the resulting MSE.

    .. math::

        \\hat{V}_{\\mathrm{DRos}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w_o(x_t,a_t;\\lambda) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    :math:`w_{o} (x_t,a_t;\\lambda)` is a new weight by the shrinkage technique which is defined as

    .. math::

        w_{o} (x_t,a_t;\\lambda) := \\frac{\\lambda}{w^2(x_t,a_t) + \\lambda} w(x_t,a_t).

    When :math:`\\lambda=0`, we have :math:`w_{o} (x,a;\\lambda)=0` corresponding to the DM estimator.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, :math:`w_{o} (x,a;\\lambda)` increases and in the limit becomes equal to
    the original importance weight, corresponding to the standard DR estimator.


    Parameters
    ----------
    lambda_: float
        Shrinkage hyperparameter.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='dr-os'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambda_: Union[float, str] = 'optimal'
    significance: float = 0.1
    estimator_name: str = "transf_dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        if isinstance(self.lambda_, str) and self.lambda_ in ['optimal', 'super-optimal', 'estimated']:
            """ if lambda value is optimal, its optimal value will be computed through the compute_lambda_value function"""
            return
        elif isinstance(self.lambda_, float):
            assert self.lambda_ >= 0.0, f"lambda hyperparameter must be larger than zero, but {self.lambda_} is given"
            assert self.lambda_ <= 1.0, f"lambda hyperparameter must be smaller than one, but {self.lambda_} is given"
        else:
            raise ValueError

    def _compute_lambda_value(
            self,
            action_dist: np.ndarray,
            d2_Renyi: float,
            weights: Optional[np.ndarray] = None,
    ) -> float:
        if isinstance(self.lambda_, str):

            n = action_dist.shape[0]

            if self.lambda_ == 'optimal':
                lambda_ = np.sqrt(np.log(1 / self.significance) / (3 * d2_Renyi * n))
            elif self.lambda_ == 'super-optimal':
                lambda_ = find_optimal_lambda(d2_Renyi, n, self.significance)
            elif self.lambda_ == 'estimated':
                lambda_ = estimate_lambda(weights, n, self.significance)

            return lambda_
        else:
            return self.lambda_

    def estimate_optimization_function_gradient_(
                self,
                context: np.ndarray,
                reward: np.ndarray,
                action: np.ndarray,
                pscore: np.ndarray,
                behav_action_dist: np.ndarray,
                #lambda_: float,
                eta: float,
                theta: np.ndarray,  #size of theta is 16 * n_actions
                d2Renyi: float,
                estimated_rewards_by_reg_model: np.ndarray,
                **kwargs,
            ) -> np.ndarray:

        #lambda_ = self._compute_lambda_value(
        #    action_dist=action, #whatever, just needed for n
        #    d2_Renyi=d2Renyi,
        #)

        num_samples = context.shape[0]
        context_size = context.shape[1]
        n_actions = behav_action_dist.shape[1]
        estimator_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)
        renyi_div_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)

        iw = np.zeros((num_samples, ))

        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            curr_behav_dist = behav_action_dist[i]
            curr_reward = reward[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            #renyi_div_gradient_term = np.sum(2*prob_context*gradient_over_actions/curr_behav_dist.$
            #renyi_div_gradient_terms[:, i] = renyi_div_gradient_term
            iw[i] = prob_context[:, curr_action] / curr_score


        lambda_ = self._compute_lambda_value(
            action_dist=action, #whatever, just needed for n
            d2_Renyi=d2Renyi,
            weights=iw,
        )



        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(num_samples), action, 0
        ]

        diff = reward - q_hat_factual

        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            curr_behav_dist = behav_action_dist[i]
            curr_reward = reward[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            renyi_div_gradient_term = np.sum(2*prob_context*gradient_over_actions/curr_behav_dist.T, axis=1)
            renyi_div_gradient_terms[:, i] = renyi_div_gradient_term
            iw[i] = prob_context[:, curr_action] / curr_score


            numerator_estimator_gradient_term = (1 - lambda_)*curr_score*gradient_over_actions[:, curr_action]
            denominator_estimator_gradient_term = np.power((1 - lambda_)*curr_score + lambda_*prob_context[:, curr_action], 2)
            estimator_gradient_term = numerator_estimator_gradient_term/denominator_estimator_gradient_term

            estimator_gradient_term2 = np.sum(gradient_over_actions * estimated_rewards_by_reg_model[i, :, 0][None, :], axis=1)

            estimator_gradient_terms[:, i] = estimator_gradient_term2 + estimator_gradient_term * diff[i] - eta * renyi_div_gradient_term

        optimization_function_gradient = np.average(estimator_gradient_terms, axis=1)

        return optimization_function_gradient

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        d2_Renyi: float,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value by the DR estimator.

        """
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            d2_Renyi=d2_Renyi,
        ).mean()

    def _estimate_round_rewards(
            self,
            reward: np.ndarray,
            action: np.ndarray,
            position: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            estimated_rewards_by_reg_model: np.ndarray,
            d2_Renyi: float,
            **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DRos estimator for each round.

        """
        n_rounds = action.shape[0]
        evaluation_dist = action_dist[np.arange(n_rounds), action, position]
        weights = evaluation_dist / pscore
        lambda_ = self._compute_lambda_value(
            action_dist=action_dist,
            d2_Renyi=d2_Renyi,
            weights=weights,
        )

        # n_rounds = action.shape[0]
        # evaluation_dist = action_dist[np.arange(n_rounds), action, position]
        transf_iw = evaluation_dist / ((1 - lambda_)*pscore + lambda_*evaluation_dist)
        q_hat_at_position = estimated_rewards_by_reg_model[
                            np.arange(n_rounds), :, position
                            ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += transf_iw * (reward - q_hat_factual)
        return estimated_rewards
        
@dataclass
class DoublyRobustWithShrinkageOptimal(DoublyRobust):
    """Estimate the policy value by Doubly Robust with optimistic shrinkage (DRos).

    Note
    ------
    DR with (optimistic) shrinkage replaces the importance weight in the original DR estimator with a new weight mapping
    found by directly optimizing sharp bounds on the resulting MSE.

    .. math::

        \\hat{V}_{\\mathrm{DRos}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w_o(x_t,a_t;\\lambda) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    :math:`w_{o} (x_t,a_t;\\lambda)` is a new weight by the shrinkage technique which is defined as

    .. math::

        w_{o} (x_t,a_t;\\lambda) := \\frac{\\lambda}{w^2(x_t,a_t) + \\lambda} w(x_t,a_t).

    When :math:`\\lambda=0`, we have :math:`w_{o} (x,a;\\lambda)=0` corresponding to the DM estimator.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, :math:`w_{o} (x,a;\\lambda)` increases and in the limit becomes equal to
    the original importance weight, corresponding to the standard DR estimator.


    Parameters
    ----------
    lambda_: float
        Shrinkage hyperparameter.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='dr-os'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    estimator_name: str = "dr-os-optimal"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DRos estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore


        q_hat_at_position = estimated_rewards_by_reg_model[
                            np.arange(n_rounds), :, position
                            ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]

        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )

        def obj(lambda_):
            shrinkage_weight = (lambda_ * iw) / (iw ** 2 + lambda_)
            z = iw ** 2
            estimated_rewards_ = estimated_rewards + shrinkage_weight * (reward - q_hat_factual)
            variance = np.var(estimated_rewards_)
            bias = np.sqrt(np.mean(z * (reward - q_hat_factual) ** 2)) * np.sqrt(np.mean(1 / z * (iw - shrinkage_weight) ** 2))

            return np.asscalar(bias ** 2 + variance)


        res = minimize(obj, x0=np.array([1]), bounds=(0, np.inf), method='Powell')
        lambda_opt = res['x']
        print(lambda_opt)

        shrinkage_weight = (lambda_opt * iw) / (iw ** 2 + lambda_opt)
        estimated_rewards += shrinkage_weight * (reward - q_hat_factual)

        return estimated_rewards

@dataclass
class TruncatedDoublyRobust(DoublyRobust):
    """Estimate the policy value by Doubly Robust with optimistic shrinkage (DRos).

    Note
    ------
    DR with (optimistic) shrinkage replaces the importance weight in the original DR estimator with a new weight mapping
    found by directly optimizing sharp bounds on the resulting MSE.

    .. math::

        \\hat{V}_{\\mathrm{DRos}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w_o(x_t,a_t;\\lambda) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    :math:`w_{o} (x_t,a_t;\\lambda)` is a new weight by the shrinkage technique which is defined as

    .. math::

        w_{o} (x_t,a_t;\\lambda) := \\frac{\\lambda}{w^2(x_t,a_t) + \\lambda} w(x_t,a_t).

    When :math:`\\lambda=0`, we have :math:`w_{o} (x,a;\\lambda)=0` corresponding to the DM estimator.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, :math:`w_{o} (x,a;\\lambda)` increases and in the limit becomes equal to
    the original importance weight, corresponding to the standard DR estimator.


    Parameters
    ----------
    lambda_: float
        Shrinkage hyperparameter.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='dr-os'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    threshold_ : float = 1
    significance: float = 0.1
    estimator_name: str = "dr-trunc"

    def _compute_threshold_value(
            self,
            action_dist: np.ndarray,
            d2_Renyi: float,
    ) -> float:
        if self.threshold_ == 'optimal':
            threshold_ = np.sqrt((3 * d2_Renyi*action_dist.shape[0])/(2 * np.log(1 / self.significance)))
            return threshold_
        else:
            return self.threshold_

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        d2_Renyi: float,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value by the DR estimator.

        """
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            d2_Renyi=d2_Renyi,
        ).mean()

    def _estimate_round_rewards(
            self,
            reward: np.ndarray,
            action: np.ndarray,
            position: np.ndarray,
            pscore: np.ndarray,
            action_dist: np.ndarray,
            estimated_rewards_by_reg_model: np.ndarray,
            d2_Renyi: float,
            **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DRos estimator for each round.

        """

        threshold_ = self._compute_threshold_value(
            action_dist=action_dist,
            d2_Renyi=d2_Renyi,
        )
        #print("threshold value is " + str(threshold_))
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        transf_iw = np.minimum(threshold_, iw)
        
        q_hat_at_position = estimated_rewards_by_reg_model[
                            np.arange(n_rounds), :, position
                            ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += transf_iw * (reward - q_hat_factual)
        return estimated_rewards
        
@dataclass
class SwitchDoublyRobustOptimal(DoublyRobust):
    """Estimate the policy value by Switch Doubly Robust (Switch-DR).

    Note
    -------
    Switch-DR aims to reduce the variance of the DR estimator by using direct method
    when the importance weight is large. This estimator estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SwitchDR}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\tau)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t)) \\mathbb{I} \\{ w(x_t,a_t) \\le \\tau \\}],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\tau (\\ge 0)` is a switching hyperparameter, which decides the threshold for the importance weight.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    Parameters
    ----------
    tau: float, default=1
        Switching hyperparameter. When importance weight is larger than this parameter, the DM estimator is applied, otherwise the DR estimator is applied.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='switch-dr'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    """

    tau: float = 1
    estimator_name: str = "switch-dr-optimal"


    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the Switch-DR estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore

        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]

        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        
        
        def obj(tau):
            switch_indicator = np.array(iw <= tau, dtype=int)
            y = switch_indicator * iw * reward + (1 - switch_indicator) * q_hat_factual
            variance = np.var(y) / n_rounds
            bias2 = np.mean(1 - switch_indicator) ** 2
            return np.asscalar(bias2 + variance)


        res = minimize(obj, x0=np.array([1]), bounds=(0, np.inf), method='Powell')
        tau_opt = res['x']
        print(tau_opt)
        
        
        switch_indicator = np.array(iw <= tau_opt, dtype=int)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += switch_indicator * iw * (reward - q_hat_factual)
        return estimated_rewards
        
        
        
@dataclass
class DoublyRobustWithShrinkageOptimal(DoublyRobust):
    """Estimate the policy value by Doubly Robust with optimistic shrinkage (DRos).

    Note
    ------
    DR with (optimistic) shrinkage replaces the importance weight in the original DR estimator with a new weight mapping
    found by directly optimizing sharp bounds on the resulting MSE.

    .. math::

        \\hat{V}_{\\mathrm{DRos}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w_o(x_t,a_t;\\lambda) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    :math:`w_{o} (x_t,a_t;\\lambda)` is a new weight by the shrinkage technique which is defined as

    .. math::

        w_{o} (x_t,a_t;\\lambda) := \\frac{\\lambda}{w^2(x_t,a_t) + \\lambda} w(x_t,a_t).

    When :math:`\\lambda=0`, we have :math:`w_{o} (x,a;\\lambda)=0` corresponding to the DM estimator.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, :math:`w_{o} (x,a;\\lambda)` increases and in the limit becomes equal to
    the original importance weight, corresponding to the standard DR estimator.


    Parameters
    ----------
    lambda_: float
        Shrinkage hyperparameter.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='dr-os'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    estimator_name: str = "dr-os-optimal"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DRos estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore


        q_hat_at_position = estimated_rewards_by_reg_model[
                            np.arange(n_rounds), :, position
                            ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]

        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )

        def obj(lambda_):
            shrinkage_weight = (lambda_ * iw) / (iw ** 2 + lambda_)
            z = iw ** 2
            estimated_rewards_ = estimated_rewards + shrinkage_weight * (reward - q_hat_factual)
            variance = np.var(estimated_rewards_)
            bias = np.sqrt(np.mean(z * (reward - q_hat_factual) ** 2)) * np.sqrt(np.mean(1 / z * (iw - shrinkage_weight) ** 2))



            return bias ** 2 + variance


        lambda_opt = minimize(obj, x0=np.array([1]), bounds=(0, np.inf), method='Powell').x

        shrinkage_weight = (lambda_opt * iw) / (iw ** 2 + lambda_opt)
        estimated_rewards += shrinkage_weight * (reward - q_hat_factual)

        return estimated_rewards

    def estimate_optimization_function_gradient_(
                self,
                context: np.ndarray,
                reward: np.ndarray,
                action: np.ndarray,
                pscore: np.ndarray,
                behav_action_dist: np.ndarray,
                #lambda_: float,
                eta: float,
                theta: np.ndarray,  #size of theta is 16 * n_actions
                d2Renyi: float,
                estimated_rewards_by_reg_model: np.ndarray,
                **kwargs,
            ) -> np.ndarray:


        num_samples = context.shape[0]
        context_size = context.shape[1]
        n_actions = behav_action_dist.shape[1]
        estimator_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)
        renyi_div_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)
        estimated_rewards = np.zeros((num_samples))

        iw = np.zeros((num_samples, ))

        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(num_samples), action, 0
        ]
        
        diff = reward - q_hat_factual
        
        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            curr_behav_dist = behav_action_dist[i]
            curr_reward = reward[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            estimated_rewards[i] = prob_context @ estimated_rewards_by_reg_model[i, :, 0]
            iw[i] = prob_context[:, curr_action] / curr_score
        
        def obj(lambda_):
            shrinkage_weight = (lambda_ * iw) / (iw ** 2 + lambda_)
            z = iw ** 2
            estimated_rewards_ = estimated_rewards + shrinkage_weight * (reward - q_hat_factual)
            variance = np.var(estimated_rewards_)
            bias = np.sqrt(np.mean(z * (reward - q_hat_factual) ** 2)) * np.sqrt(np.mean(1 / z * (iw - shrinkage_weight) ** 2))

            #print(bias, variance)
            return bias ** 2 + variance


        lambda_opt = minimize(obj, x0=np.array([1]), bounds=(0, np.inf), method='Powell').x

        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            curr_behav_dist = behav_action_dist[i]
            curr_reward = reward[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            renyi_div_gradient_term = np.sum(2*prob_context*gradient_over_actions/curr_behav_dist.T, axis=1)
            renyi_div_gradient_terms[:, i] = renyi_div_gradient_term
            iw[i] = prob_context[:, curr_action] / curr_score

            w = prob_context[:, curr_action] / curr_score
            deriv_w = lambda_opt * (lambda_opt - w ** 2) / (lambda_opt + w ** 2) ** 2

            estimator_gradient_term = deriv_w / curr_score * gradient_over_actions[:, curr_action]
            estimator_gradient_term2 = np.sum(gradient_over_actions * estimated_rewards_by_reg_model[i, :, 0][None, :], axis=1)

            estimator_gradient_terms[:, i] = estimator_gradient_term2 + estimator_gradient_term * diff[i] - eta * renyi_div_gradient_term

        # optimization_function_gradient = np.average(estimator_gradient_terms - eta * renyi_div_gradient_terms, axis=1)
        optimization_function_gradient = np.average(estimator_gradient_terms, axis=1)

        return optimization_function_gradient

@dataclass
class InverseProbabilityWeightingShrinkageOptimal(BaseOffPolicyEstimator):
    """Estimate the policy value by Doubly Robust with optimistic shrinkage (DRos).

    Note
    ------
    DR with (optimistic) shrinkage replaces the importance weight in the original DR estimator with a new weight mapping
    found by directly optimizing sharp bounds on the resulting MSE.

    .. math::

        \\hat{V}_{\\mathrm{DRos}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w_o(x_t,a_t;\\lambda) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`.

    :math:`w_{o} (x_t,a_t;\\lambda)` is a new weight by the shrinkage technique which is defined as

    .. math::

        w_{o} (x_t,a_t;\\lambda) := \\frac{\\lambda}{w^2(x_t,a_t) + \\lambda} w(x_t,a_t).

    When :math:`\\lambda=0`, we have :math:`w_{o} (x,a;\\lambda)=0` corresponding to the DM estimator.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, :math:`w_{o} (x,a;\\lambda)` increases and in the limit becomes equal to
    the original importance weight, corresponding to the standard DR estimator.


    Parameters
    ----------
    lambda_: float
        Shrinkage hyperparameter.
        This hyperparameter should be larger than or equal to 0., otherwise it is meaningless.

    estimator_name: str, default='dr-os'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    estimator_name: str = "is-os-optimal"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DRos estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore

        def obj(lambda_):
            shrinkage_weight = (lambda_ * iw) / (iw ** 2 + lambda_)
            estimated_rewards_ = shrinkage_weight * reward
            variance = np.var(estimated_rewards_)
            bias = np.sqrt(np.mean((iw - shrinkage_weight) ** 2)) * max(reward)
            return bias ** 2 + variance


        lambda_opt = minimize(obj, x0=np.array([1]), bounds=(0, np.inf), method='Powell').x

        shrinkage_weight = (lambda_opt * iw) / (iw ** 2 + lambda_opt)
        estimated_rewards = shrinkage_weight * reward

        return estimated_rewards

    def estimate_optimization_function_gradient_(
                self,
                context: np.ndarray,
                reward: np.ndarray,
                action: np.ndarray,
                pscore: np.ndarray,
                behav_action_dist: np.ndarray,
                #lambda_: float,
                eta: float,
                theta: np.ndarray,  #size of theta is 16 * n_actions
                d2Renyi: float,
                **kwargs,
            ) -> np.ndarray:


        num_samples = context.shape[0]
        context_size = context.shape[1]
        n_actions = behav_action_dist.shape[1]
        estimator_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)
        renyi_div_gradient_terms = np.zeros((n_actions*context_size, num_samples), dtype=float)

        iw = np.zeros((num_samples, ))
        
        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            iw[i] = prob_context[:, curr_action] / curr_score
        
        def obj(lambda_):
            shrinkage_weight = (lambda_ * iw) / (iw ** 2 + lambda_)
            estimated_rewards_ = shrinkage_weight * reward
            variance = np.var(estimated_rewards_)
            bias = np.sqrt(np.mean((iw - shrinkage_weight) ** 2)) * max(reward)
            return bias ** 2 + variance

        lambda_opt = minimize(obj, x0=np.array([1]), bounds=(0, np.inf), method='Powell').x

        for i in range(0, num_samples):
            curr_context = context[i]
            curr_action = action[i]
            curr_score = pscore[i]
            curr_behav_dist = behav_action_dist[i]
            gradient_over_actions, prob_context = compute_estimator_gradient_over_actions(curr_context, n_actions, theta)
            renyi_div_gradient_term = np.sum(2*prob_context*gradient_over_actions/curr_behav_dist.T, axis=1)
            renyi_div_gradient_terms[:, i] = renyi_div_gradient_term
            iw[i] = prob_context[:, curr_action] / curr_score

            w = prob_context[:, curr_action] / curr_score
            deriv_w = lambda_opt * (lambda_opt - w ** 2) / (lambda_opt + w ** 2) ** 2

            estimator_gradient_term = deriv_w / curr_score * gradient_over_actions[:, curr_action]

            estimator_gradient_terms[:, i] = estimator_gradient_term * reward[i] - eta * renyi_div_gradient_term

        # optimization_function_gradient = np.average(estimator_gradient_terms - eta * renyi_div_gradient_terms, axis=1)
        optimization_function_gradient = np.average(estimator_gradient_terms, axis=1)

        return optimization_function_gradient

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities
            by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
