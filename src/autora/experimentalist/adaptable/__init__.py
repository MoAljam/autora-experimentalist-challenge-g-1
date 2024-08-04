"""
adaptable Experimentalist
"""

import logging.handlers
from autora.utils.deprecation import deprecated_alias

import torch
from torch.nn.functional import softmax

# from sklearn.metrics.pairwise import euclidean_distances as euclidean
from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import euclidean
from scipy.stats import gaussian_kde

from scipy.stats import entropy
from scipy.special import kl_div
from scipy.special import rel_entr
from scipy.stats import norm, uniform


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from time import sleep

from typing import Union, List
import logging
import os

_logger = logging.getLogger(__name__)
# set log file to write logs into

# Create a logger for this module
os.makedirs("logs", exist_ok=True)
_module_logger = logging.getLogger("adaptable_experimentalist")
_module_logger.setLevel(logging.DEBUG)

# Create a rotating file handler with 5mb max for logging
file_handler = logging.handlers.RotatingFileHandler(
    "./logs/adaptable_experimentalist.log", maxBytes=5 * 1024 * 1024, backupCount=5
)
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)

# Add the file handler to the logger
_module_logger.addHandler(file_handler)


# jensen_shannon_divergence as the square of jensen shannon distance
def jensen_shannon_divergence(p, q):
    return jensenshannon(p, q) ** 2


class uniform_on_range:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def pdf(self, x):
        return uniform.pdf(x, loc=self.min_val, scale=self.max_val - self.min_val)

    def __call__(self, x):
        return self.pdf(x)


def surprisal_score(
    conditions: np.ndarray,
    models: list,
    # prior_model # compare to a prior model (example: uniform distribution)
    plot: bool = False,
) -> float:

    current_model = models[-1]
    current_observations = current_model.predict(conditions)
    current_observations = np.asarray(current_observations).flatten()

    # TODO chech for data dimensionality for compatibility with gaussian_kde
    _logger.debug(f"## adaptable.surprisal_score: conditions shape: {conditions.shape}")
    _logger.debug(f"## adaptable.surprisal_score: current_observations shape: {current_observations.shape}")

    try:
        current_distribution_estimator = gaussian_kde(current_observations)
    except Exception as e:
        _logger.error(f"Error in creating a gaussian kde: {e}")
        current_distribution_estimator = None  # will define a uniform distribution later

    if len(models) < 2:
        # if there is only one model, assume the the prior model is a uniform distribution
        min_observation, max_observation = np.min(current_observations), np.max(current_observations)
        prior_disribution_estimatior = uniform_on_range(min_observation, max_observation)
        _module_logger.debug(f"## adaptable.surprisal_score: prior distribution is a uniform distribution")
        _module_logger.debug(
            f"## adaptable.surprisal_score: sample from the uniform distribution: {prior_disribution_estimatior(current_observations)[:5]}"
        )

    else:
        previos_model = models[-2]
        prior_observations = previos_model.predict(conditions)
        prior_observations = np.asarray(prior_observations).flatten()

        min_observation = np.min([np.min(current_observations), np.min(prior_observations)])
        max_observation = np.max([np.max(current_observations), np.max(prior_observations)])
        try:
            prior_disribution_estimatior = gaussian_kde(prior_observations)
        except Exception as e:
            _logger.error(f"Error in creating a gaussian kde: {e}")
            _module_logger.error(f"Error in creating a gaussian kde: {e}")
            prior_disribution_estimatior = uniform_on_range(min_observation, max_observation)

    if current_distribution_estimator is None:
        current_distribution_estimator = uniform_on_range(min_observation, max_observation)

    shared_x = np.linspace(min_observation, max_observation, conditions.shape[0])

    current_distribution = current_distribution_estimator(shared_x)
    prior_disribution = prior_disribution_estimatior(shared_x)

    normalized_current_distribution = current_distribution / np.sum(current_distribution)
    normalized_prior_disribution = prior_disribution / np.sum(prior_disribution)

    # _module_logger.debug(f"## adaptable.surprisal_score: current_distribution shape: {current_distribution.shape}")
    # _module_logger.debug(
    #     f"## adaptable.surprisal_score: normalized_current_distribution shape: {normalized_current_distribution.shape}"
    # )
    score_kld = np.sum(kl_div(normalized_current_distribution, normalized_prior_disribution))
    _module_logger.debug(f"## adaptable.surprisal_score: kl_div score: {np.round(score_kld, 6)}")

    score_jsd = jensen_shannon_divergence(normalized_current_distribution, normalized_prior_disribution)
    _module_logger.debug(f"## adaptable.surprisal_score: js_div score: {np.round(score_jsd, 6)}")

    # NOTE you could stream this as video from here on some operating systems !!! ( TODO should be removed later )
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(shared_x, normalized_current_distribution, label="normalized current distribution")
        plt.plot(shared_x, normalized_prior_disribution, label="normalized prior distribution")
        plt.suptitle("Normalized distributions")
        plt.title(f"KL divergence: {np.round(score_kld, 6)} | JS divergence: {np.round(score_jsd, 6)}")
        plt.xlabel("observations")
        plt.ylabel("normalized density")
        plt.legend()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("./plots/adaptable_normalized_distributions.png")
        plt.clf()
        # sleep(5)

    return score_jsd


def map_meta_score_to_temperature(meta_score: float, **kwargs) -> float:
    pass


def exponential_bias(x, rate=4):
    return 1 - np.exp(-rate * x)


def logistic_bias(x, growth_rate=20, midpoint=0.5):
    return 1 / (1 + np.exp(-growth_rate * (x - midpoint)))


def polynomial_bias(x, power=0.5):
    return x**power


def get_sampler_index_gaussian(
    samplers_coords: np.ndarray, mu: Union[float, np.ndarray], std: Union[float, np.ndarray] = 1
):
    min_position = np.min(samplers_coords)
    max_position = np.max(samplers_coords)

    current_progress = mu * (max_position - min_position) + min_position
    # sample a point around the current progress with a gaussian distribution
    # check if mu is a scalar or an array
    if np.isscalar(mu):
        sample_normal = np.random.normal(current_progress, std, size=(1,))
    else:
        if np.isscalar(std):
            covariane = np.eye(len(mu)) * std
        else:  # std should be an array
            covariane = std
        try:
            sample_normal = np.random.multivariate_normal(current_progress, cov=covariane)
        except Exception as e:
            _logger.error(f"Error in sampling from a multivariate normal distribution: {e}")
            raise e

    # clip the sample to the range of the samplers_coords
    sample_normal = np.clip(sample_normal, min_position, max_position)
    # get the index of the nearest coordinate in the samplers_coords to the sample_normal
    if len(samplers_coords.shape) == 1:
        # add a dimension to the sample_normal
        samplers_coords = samplers_coords[:, np.newaxis]
    distance_to_samplers = np.apply_along_axis(lambda x: euclidean(x, sample_normal), 1, samplers_coords)
    index_closest_sampler = np.argmin(distance_to_samplers)
    return index_closest_sampler


# def meta_score(conditions, models, **kwargs):
#     pass


def score_sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    models: list,
    samplers: list,
    samplers_coords: list = None,
    num_samples: int = 1,
    meta_score_func: callable = None,
    sensitivity: Union[float, int] = 8,
    plot: bool = False,
) -> pd.DataFrame:
    """
    adaptive experimentalist that samples from a set of samplers based on its history of surprisal (as jensen shannon divergence)

    Args:
    conditions: conditions to sample from
    models: list of models
    samplers:   list of dicts
                each sampler is a dictionary with the following keys:
                - func: callable sampler
                - params: dict
                - name: str
    sampler_coords: list of the locations of the samplers on arbitrary 1d-scale
    num_samples: number of samples to generate
    meta_score_func: callable, function to calculate the meta score
    sensitivity:    float, int, scaler for the exponential bias
                    higher number maps to more sensitivity of changes in the meta score (surprisal , jsd)
    plot: bool, plot the distributions
    Returns:
    pd.DataFrame: new conditions with a score column
    """
    # if sampler_coords is not provided, assume 1d meta space
    if samplers_coords is None or len(samplers_coords) != len(samplers):
        _logger.warning("samplers_coords is not provided or has a different length than samplers")
        _logger.warning("generating uniform samplers_coords based on the length of samplers")
        samplers_coords = np.arange(len(samplers))

    assert np.isscalar(samplers_coords[0]), "samplers_coords should contain scalars"

    # make sure that samplers_coords is a numpy array
    samplers_coords = np.asarray(samplers_coords)

    condition_pool_copy = pd.DataFrame(conditions)

    # calculate the current meta score (surprisal score)
    if meta_score_func is None:
        meta_score_func = surprisal_score
    # current_meta_score = meta_score_func(conditions, models)
    current_meta_score = surprisal_score(conditions, models, plot=plot)

    # map the meta score [ jsd (0-1) ] to a temperature (0-1)
    current_temperature = exponential_bias(current_meta_score, rate=sensitivity)

    _module_logger.info(f"## adaptable.sample: current_meta_score: {np.round(current_meta_score, 6)}")
    _module_logger.debug(f"## adaptable.sample: current temprature: {np.round(current_temperature, 6)}")

    # make sure current_temperature is a scalar
    if not np.isscalar(current_temperature) or np.isnan(current_temperature):
        current_temperature = 0
    if np.isinf(current_temperature):
        current_temperature = 1
    if current_temperature < 0:
        raise ValueError("current_temperature should be a positive scalar")

    _module_logger.info(f"## adaptable.sample: adjusted current temprature: {np.round(current_temperature, 6)}")

    # calculate std based on the range of samplers coordinates and temprature
    # the std should be lower in the beginning and higher at the end
    # to counter act the effect the assumptions of the model in later stages
    abs_scale_diff = np.abs(np.max(samplers_coords) - np.min(samplers_coords))
    std = abs_scale_diff * 0.1  # NOTE arbitrary value

    # index of the nearest sampler to a gaussian around the current progress
    index_closest_sampler = get_sampler_index_gaussian(samplers_coords, current_temperature, std)

    # NOTE you could stream this as video from here on some operating systems !!! ( TODO should be removed later )
    if plot:
        # plot the gaussian distribution around the current progress
        _current_progress = current_temperature * (np.max(samplers_coords) - np.min(samplers_coords)) + np.min(
            samplers_coords
        )
        _x = np.linspace(np.min(samplers_coords), np.max(samplers_coords), 100)
        _y = norm.pdf(_x, loc=_current_progress, scale=std)
        plt.figure(figsize=(10, 5))
        plt.plot(_x, _y, label="gaussian distribution around the current progress")
        plt.axvline(x=_current_progress, color="r", linestyle="--", label="current progress")
        plt.axvline(x=samplers_coords[index_closest_sampler], color="g", linestyle="--", label="closest sampler")
        plt.xticks(samplers_coords, [sampler["name"] for sampler in samplers], rotation=45)
        plt.legend()
        plt.suptitle("Gaussian distribution around the current progress")
        plt.title(
            f"current tempreture: {np.round(_current_progress, 6)}, current meta score: {np.round(current_meta_score, 6)} \nstd: {std}"
        )
        plt.xlabel("samplers-space")
        plt.ylabel("density")
        os.makedirs("plots", exist_ok=True)
        plt.tight_layout()
        plt.savefig("./plots/adaptable_samplers_space.png")
        plt.clf()

    _logger.debug(f"## adaptable.sample: index closest sampler: {index_closest_sampler}")

    # get the sampler and its parameters
    sampler_dict = samplers[index_closest_sampler]
    sampler = sampler_dict["func"]
    sampler_params = sampler_dict["params"]
    sampler_name = sampler_dict.get("name", "unknown")

    _logger.debug(f"## adaptable.sample: sampler name: {sampler_name}")

    _module_logger.debug(f"## adaptable.sample: sampler name: {sampler_name}")
    if num_samples is None:
        num_samples = condition_pool_copy.shape[0]

    new_conditions = sampler(conditions=conditions, num_samples=num_samples, **sampler_params)

    # check if neu_conditions is a DataFrame
    # if it is a data fram but doen't have a score column, add a score column and fill it with zeros
    if isinstance(new_conditions, pd.DataFrame):
        if "score" not in new_conditions.columns:
            new_conditions["score"] = 0
    else:
        new_conditions = pd.DataFrame(new_conditions)
        # check if neu_conditions has the same number of columns or one more column than the original conditions
        if new_conditions.shape[1] == len(condition_pool_copy.columns):
            new_conditions["score"] = 0
        elif new_conditions.shape[1] == len(condition_pool_copy.columns) + 1:
            new_conditions.columns = list(condition_pool_copy.columns) + ["score"]
        else:
            raise ValueError("new conditions")

    return new_conditions


def sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    models: list,
    samplers: list,
    samplers_coords: list = None,
    num_samples: int = 1,
    meta_score_func: callable = None,
    sensitivity: Union[float, int] = 8,
    plot: bool = False,
) -> pd.DataFrame:
    """
    adaptive experimentalist that samples from a set of samplers based on its history of surprisal (as jensen shannon divergence)

    Args:
    conditions: conditions to sample from
    models: list of models
    samplers:   list of dicts
                each sampler is a dictionary with the following keys:
                - func: callable sampler
                - params: dict
                - name: str
    sampler_coords: list of the locations of the samplers on arbitrary 1d-scale
    num_samples: number of samples to generate
    meta_score_func: callable, function to calculate the meta score
    sensitivity:    float, int, scaler for the exponential bias
                    higher number maps to more sensitivity of changes in the meta score (surprisal , jsd)
    plot: bool, plot the distributions
    Returns:
    pd.DataFrame: new conditions with a score column
    """

    condition_pool_copy = conditions.copy()

    new_conditions = adaptable_score_sample(
        conditions=conditions,
        models=models,
        samplers=samplers,
        samplers_coords=samplers_coords,
        num_samples=num_samples,
        meta_score_func=meta_score_func,
        sensitivity=sensitivity,
        plot=plot,
    )
    new_conditions.drop("score", axis=1, inplace=True)

    if isinstance(condition_pool_copy, pd.DataFrame):
        new_conditions = pd.DataFrame(new_conditions, columns=condition_pool_copy.columns)

    return new_conditions


adaptable_sample = sample
adaptable_sample.__doc__ = """Alias for sample"""
adaptable_sampler = deprecated_alias(adaptable_sample, "adaptable_sampler")

adaptable_score_sample = score_sample
adaptable_score_sample.__doc__ = """Alias for score_sample"""
adaptable_score_sampler = deprecated_alias(adaptable_score_sample, "adaptive_score_sampler")
