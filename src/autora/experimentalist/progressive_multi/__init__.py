"""
progressive_multi Experimentalist
"""

# TODO:
# complete the arbitary dimensionality of the meta space
# - map 1d temprature to to the multi-dimensional space
#   - provide a costume mapping ?
# - already provide a multi-dimensional temprature matching the meta space ?? (worse case)

from autora.utils.deprecation import deprecated_alias

import numpy as np
import pandas as pd

# from sklearn.metrics.pairwise import euclidean_distances as euclidean
from scipy.spatial.distance import euclidean


from typing import Union, List
from logging import getLogger

_logger = getLogger(__name__)


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


def score_sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    samplers: list,
    samplers_coords: list = None,
    num_samples: int = None,
    temprature: float = 0.5,
) -> pd.DataFrame:
    assert isinstance(temprature, (int, float)), "temprature should be a scalar"
    assert 0 <= temprature <= 1, "temprature should be between 0 and 1"

    # if sampler_coords is not provided, assume 1d meta space
    if samplers_coords is None or len(samplers_coords) != len(samplers):
        _logger.warning("samplers_coords is not provided or has a different length than samplers")
        _logger.warning("generating uniform samplers_coords based on the length of samplers")
        samplers_coords = np.arange(len(samplers))

    assert np.isscalar(samplers_coords[0]), "samplers_coords should contain scalars"

    # make sure that samplers_coords is a numpy array
    samplers_coords = np.asarray(samplers_coords)

    condition_pool_copy = pd.DataFrame(conditions)
    # calculate std based on the range of samplers coordinates and temprature
    # the std should be lower in the beginning and higher at the end
    # to counter act the effect the assumptions of the model in later stages
    abs_scale_diff = np.abs(np.max(samplers_coords) - np.min(samplers_coords))
    std = abs_scale_diff * 0.1

    _logger.debug(f"## progressive_multi.sample: samplers_coords: {samplers_coords}")
    _logger.debug(f"## progressive_multi.sample: absulute difference: {abs_scale_diff}")
    _logger.debug(f"## progressive_multi.sample: temprature: {temprature}")
    _logger.debug(f"## progressive_multi.sample: std: {std}")

    # index of the nearest sampler to a gaussian around the current progress
    index_closest_sampler = get_sampler_index_gaussian(samplers_coords, temprature, std)

    _logger.debug(f"## progressive_multi.sample: index closest sampler: {index_closest_sampler}")

    # get the sampler and its parameters
    sampler_dict = samplers[index_closest_sampler]
    sampler = sampler_dict["func"]
    sampler_params = sampler_dict["params"]
    sampler_name = sampler_dict.get("name", "unknown")

    _logger.debug(f"## progressive_multi.sample: sampler name: {sampler_name}")

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
    samplers: list,
    samplers_coords: list = None,
    num_samples: int = 1,
    temprature: float = 0.5,
) -> pd.DataFrame:
    """
    Sample from a list of samplers based on the progress / assumed confidence of model in the sampling process.
    conditions: conditions to sample from
    samplers:   list of dicts
                each sampler is a dictionary with the following keys:
                - func: callable sampler
                - params: dict
                - name: str
    sampler_coords: list of the locations of the samplers on arbitrary 1d-scale
    num_samples: number of samples to generate
    temprature: float, 0 <= temprature <= 1
    """

    condition_pool_copy = conditions.copy()

    new_conditions = progressive_multi_score_sample(
        conditions=conditions,
        samplers=samplers,
        samplers_coords=samplers_coords,
        num_samples=num_samples,
        temprature=temprature,
    )
    new_conditions.drop("score", axis=1, inplace=True)

    if isinstance(condition_pool_copy, pd.DataFrame):
        new_conditions = pd.DataFrame(new_conditions, columns=condition_pool_copy.columns)

    return new_conditions


progressive_multi_sample = sample
progressive_multi_sample.__doc__ = """Alias for sample"""
progressive_multi_sampler = deprecated_alias(progressive_multi_sample, "progressive_multi_sampler")

progressive_multi_score_sample = score_sample
progressive_multi_score_sample.__doc__ = """Alias for score_sample"""
progressive_multi_score_sampler = deprecated_alias(progressive_multi_score_sample, "progressive_multi_score_sampler")
