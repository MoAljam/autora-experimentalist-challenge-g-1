"""
adaptable Experimentalist
"""

from .utils import exponential_bias, logistic_bias, polynomial_bias, jensen_shannon_divergence, UniformOnRange

import logging.handlers
from autora.utils.deprecation import deprecated_alias

from scipy.spatial.distance import euclidean
from scipy.stats import gaussian_kde

from scipy.special import kl_div
from scipy.stats import norm, uniform, multivariate_normal

import torch
import numpy as np
import pandas as pd

from typing import Union, List
import logging
import os

_logger = logging.getLogger(__name__)
# set log file to write logs into

# Create a logger for this module
# os.makedirs("logs", exist_ok=True)
# _module_logger = logging.getLogger("adaptable_experimentalist")
# _module_logger.setLevel(logging.DEBUG)

# # Create a rotating file handler with 5mb max for logging
# file_handler = logging.handlers.RotatingFileHandler(
#     "./logs/adaptable_experimentalist.log", maxBytes=5 * 1024 * 1024, backupCount=5
# )
# file_handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
# file_handler.setFormatter(formatter)
# _module_logger.addHandler(file_handler)


INSPECTION_PLOTS_DIR = "inspection_plots"


def surprisal_score(
    conditions: np.ndarray,
    models: list,
    # prior_model # compare to a prior model (example: uniform distribution)
    plot_info: bool = False,
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
        prior_disribution_estimatior = UniformOnRange(min_observation, max_observation)
        # _module_logger.debug(f"## adaptable.surprisal_score: prior distribution is a uniform distribution")
        # _module_logger.debug(
        #     f"## adaptable.surprisal_score: sample from the uniform distribution: {prior_disribution_estimatior(current_observations)[:5]}"
        # )

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
            # _module_logger.error(f"Error in creating a gaussian kde: {e}")
            prior_disribution_estimatior = UniformOnRange(min_observation, max_observation)

    if current_distribution_estimator is None:
        current_distribution_estimator = UniformOnRange(min_observation, max_observation)

    shared_x = np.linspace(min_observation, max_observation, conditions.shape[0])

    current_distribution = current_distribution_estimator(shared_x)
    prior_disribution = prior_disribution_estimatior(shared_x)

    # normalized_current_distribution = current_distribution / np.sum(current_distribution)
    # normalized_prior_disribution = prior_disribution / np.sum(prior_disribution)
    normalized_current_distribution = torch.nn.functional.softmax(torch.tensor(current_distribution), dim=0)
    normalized_current_distribution = normalized_current_distribution.numpy()
    normalized_prior_disribution = torch.nn.functional.softmax(torch.tensor(prior_disribution), dim=0)
    normalized_prior_disribution = normalized_prior_disribution.numpy()

    score_kld = np.sum(kl_div(normalized_current_distribution, normalized_prior_disribution))
    # _module_logger.debug(f"## adaptable.surprisal_score: kl_div score: {np.round(score_kld, 6)}")

    score_jsd = jensen_shannon_divergence(normalized_current_distribution, normalized_prior_disribution)
    # _module_logger.debug(f"## adaptable.surprisal_score: js_div score: {np.round(score_jsd, 6)}")

    # get the information about the current distributions and scores for further inspection/analysis if needed
    if plot_info:
        import json

        os.makedirs(os.path.join(INSPECTION_PLOTS_DIR, "surprisal_score_dists"), exist_ok=True)
        # create a unique filename with a timestamp inside the plots directory
        filename = f"surprisal_score_dists_{pd.Timestamp.now().strftime('m%d%H%M%S')}.json"
        filename = os.path.join(INSPECTION_PLOTS_DIR, "surprisal_score_dists", filename)
        with open(filename, "w") as f:
            json.dump(
                {
                    "shared_x": shared_x.tolist(),
                    "normalized_current_distribution": normalized_current_distribution.tolist(),
                    "normalized_prior_disribution": normalized_prior_disribution.tolist(),
                    "score_kld": float(score_kld),
                    "score_jsd": float(score_jsd),
                    "min_observation": float(min_observation),
                    "max_observation": float(max_observation),
                },
                f,
            )

    return score_jsd


def meta_score_to_sampler_space():
    pass


def get_sampler_index_gaussian(
    samplers_coords: np.ndarray, mu: Union[float, np.ndarray], std: Union[float, np.ndarray] = 1
):
    min_position = np.min(samplers_coords)
    max_position = np.max(samplers_coords)

    current_mu = mu * (max_position - min_position) + min_position
    # sample a point around the current progress with a gaussian distribution
    # check if mu is a scalar or an array
    if np.isscalar(mu):
        sample_normal = np.random.normal(current_mu, std, size=(1,))
    else:
        if np.isscalar(std):
            covariane = np.eye(len(mu)) * std
        else:  # std should be an array
            covariane = std
        try:
            sample_normal = np.random.multivariate_normal(current_mu, cov=covariane)
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


def get_sampler_weights_gaussian(
    samplers_coords: np.ndarray, mu: Union[float, np.ndarray], std: Union[float, np.ndarray] = 1
) -> np.ndarray:
    """
    Calculates the weights of samplers based on a Gaussian PDF of their coordinates.

    Args:
        samplers_coords (np.ndarray): Coordinates of the samplers.
        mu (Union[float, np.ndarray]): Mean value(s) for the Gaussian distribution.
        std (Union[float, np.ndarray], optional): Standard deviation(s) for the Gaussian distribution. Defaults to 1.

    Returns:
        np.ndarray: Weights for each sampler, normalized to sum to 1.
    """
    min_position = np.min(samplers_coords)
    max_position = np.max(samplers_coords)
    samplers_coords = np.asarray(samplers_coords)

    current_mu = mu * (max_position - min_position) + min_position

    if np.isscalar(mu):
        weights = norm.pdf(samplers_coords, loc=current_mu, scale=std)
    else:
        if np.isscalar(std):
            covariance = np.eye(len(mu)) * (std**2)
        else:
            covariance = std
        # ensure samplers_coords is multidimensional
        samplers_coords = np.atleast_2d(samplers_coords)
        current_mu = np.asarray(current_mu)
        weights = multivariate_normal.pdf(samplers_coords, mean=current_mu, cov=covariance)

    # normalize the weights to sum to 1
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights /= total_weight
    else:
        # avoid division by zero if total_weight is 0
        weights = np.zeros_like(weights)

    return weights


def score_sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    reference_conditions: Union[pd.DataFrame, np.ndarray],
    models: list,
    samplers: list,
    samplers_coords: list = None,
    num_samples: int = 1,
    meta_score_func: callable = None,
    sensitivity: Union[float, int] = 8,
    temperature: float = 0.1,
    plot_info: bool = False,
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
                    sensitivity to the changes in the meta score (surprisal , jsd) for tha mapping into the sampling space

    temperature:   float, int, scaler for the std of the gaussian distribution around the mapping int the sampling space
    plot_info: bool, plot the distributions
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
    reference_conditions = pd.DataFrame(reference_conditions)

    # combine the conditions and reference_conditions such that the raws don't repeat
    # condition_pool_copy = pd.concat([condition_pool_copy, reference_conditions], ignore_index=True)
    # condition_pool_copy = condition_pool_copy.drop_duplicates(keep="first")

    # calculate the current meta score (surprisal score)
    if meta_score_func is None:
        meta_score_func = surprisal_score
    # current_meta_score = meta_score_func(conditions, models)
    current_meta_score = surprisal_score(conditions, models, plot_info=plot_info)

    # map the meta score [ jsd (0-1) ] to linear space of sampling methods (0-1)
    current_projection = exponential_bias(current_meta_score, rate=sensitivity)
    # _module_logger.info(f"## adaptable.sample: current_meta_score: {np.round(current_meta_score, 6)}")
    # _module_logger.debug(f"## adaptable.sample: current projection: {np.round(current_projection, 6)}")

    # make sure current_temperature is a scalar
    if not np.isscalar(current_projection) or np.isnan(current_projection):
        current_projection = 0
    elif np.isinf(current_projection):
        current_projection = 1
    elif current_projection < 0:
        raise ValueError("current_projection should be a positive scalar")

    # _module_logger.info(f"## adaptable.sample: adjusted current projection: {np.round(current_projection, 6)}")

    # calculate std based on the range of samplers coordinates and temprature
    # the std should be lower in the beginning and higher at the end
    # to counter act the effect the assumptions of the model in later stages
    abs_scale_diff = np.abs(np.max(samplers_coords) - np.min(samplers_coords))
    std = abs_scale_diff * temperature

    # index of the nearest sampler to a gaussian around the current progress
    # index_closest_sampler = get_sampler_index_gaussian(samplers_coords, current_projection, std)
    # _logger.debug(f"## adaptable.sample: index closest sampler: {index_closest_sampler}")
    samplers_weights = get_sampler_weights_gaussian(samplers_coords, current_projection, std)

    if num_samples is None:
        num_samples = condition_pool_copy.shape[0]

    # get the scores of the conditions from all samplers
    # weighted by the gaussian distribution around the current progress
    weighted_final_scores = np.zeros(condition_pool_copy.shape[0])
    for sampler_dict, sampler_weight in zip(samplers, samplers_weights):
        # sampler_dict = samplers[index_closest_sampler]
        sampler = sampler_dict["func"]
        sampler_params = sampler_dict["params"]
        sampler_name = sampler_dict.get("name", "unknown")

        # new_conditions = sampler(conditions=conditions, num_samples=num_samples, **sampler_params)
        new_conditions = sampler(conditions=conditions, **sampler_params)
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

        new_conditions = new_conditions.sort_index()
        # weight the scores by the gaussian distribution around the current progress
        weighted_final_scores += new_conditions["score"].values * sampler_weight

    condition_pool_copy["score"] = weighted_final_scores
    new_conditions = condition_pool_copy.sort_values(by="score", ascending=False)
    # new_conditions = new_conditions.sort_values(by="score", ascending=False)
    # new_conditions = new_conditions[:num_samples]

    # pick n samples based on the scores as probabilities
    try:
        # normalize to 0-1
        weighted_final_scores_normalized = torch.nn.functional.softmax(torch.tensor(weighted_final_scores), dim=0)
        weighted_final_scores_normalized = weighted_final_scores_normalized.numpy()

    except ZeroDivisionError:
        # uniform distribution
        weighted_final_scores_normalized = np.ones_like(weighted_final_scores) / len(weighted_final_scores)

    new_conditions_indecies = np.random.choice(
        new_conditions.index,
        size=num_samples,
        p=weighted_final_scores_normalized,
        replace=False,
    )
    new_conditions = new_conditions.loc[new_conditions_indecies]

    # get the information about the sampling space
    if plot_info:
        # plot the gaussian distribution around the current progress
        _current_meta_score_projection = current_projection * (
            np.max(samplers_coords) - np.min(samplers_coords)
        ) + np.min(samplers_coords)
        _x = np.linspace(np.min(samplers_coords), np.max(samplers_coords), 100)
        _y = norm.pdf(_x, loc=_current_meta_score_projection, scale=std)

        # write the plot data into a json file
        import json

        os.makedirs(os.path.join(INSPECTION_PLOTS_DIR, "samplers_space"), exist_ok=True)

        filename = f"samplers_space_{pd.Timestamp.now().strftime('m%d%H%M%S')}.json"
        filename = os.path.join(INSPECTION_PLOTS_DIR, "samplers_space", filename)
        with open(filename, "w") as f:
            json.dump(
                {
                    "x": _x.tolist(),
                    "y": _y.tolist(),
                    "samplers_coords": samplers_coords.tolist(),
                    "samplers_names": [sampler["name"] for sampler in samplers],
                    # "index_closest_sampler": float(index_closest_sampler),
                    "samplers_weights": samplers_weights.tolist(),
                    "current_meta_score": float(current_meta_score),
                    "current_projection": float(current_projection),
                    "mu": float(_current_meta_score_projection),
                    "std": float(std),
                    "temperature": float(temperature),
                    "samplers_weights": samplers_weights.tolist(),
                },
                f,
            )

    return new_conditions


def sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    reference_conditions: Union[pd.DataFrame, np.ndarray],
    models: list,
    samplers: list,
    samplers_coords: list = None,
    num_samples: int = 1,
    meta_score_func: callable = None,
    sensitivity: Union[float, int] = 8,
    temperature: float = 0.1,
    plot_info: bool = False,
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
    temperature:   float, int, scaler for the std of the gaussian distribution around the mapping int the sampling space
    plot_info: bool, plot the distributions
    Returns:
    pd.DataFrame: new conditions with a score column
    """

    condition_pool_copy = conditions.copy()
    reference_conditions = reference_conditions.copy()

    new_conditions = adaptable_score_sample(
        conditions=conditions,
        reference_conditions=reference_conditions,
        models=models,
        samplers=samplers,
        samplers_coords=samplers_coords,
        num_samples=num_samples,
        meta_score_func=meta_score_func,
        sensitivity=sensitivity,
        temperature=temperature,
        plot_info=plot_info,
    )
    new_conditions.drop("score", axis=1, inplace=True)

    if isinstance(condition_pool_copy, pd.DataFrame):
        new_conditions = pd.DataFrame(new_conditions, columns=condition_pool_copy.columns)

    return new_conditions[:num_samples]


adaptable_sample = sample
adaptable_sample.__doc__ = """Alias for sample"""
adaptable_sampler = deprecated_alias(adaptable_sample, "adaptable_sampler")

adaptable_score_sample = score_sample
adaptable_score_sample.__doc__ = """Alias for score_sample"""
adaptable_score_sampler = deprecated_alias(adaptable_score_sample, "adaptive_score_sampler")
