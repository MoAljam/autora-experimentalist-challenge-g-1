from autora.experimentalist.adaptable import get_sampler_weights_gaussian
from scipy.stats import norm
import numpy as np


def test_sampler_weights_gaussian_univariate_case():
    """
    Test the function with scalar mu and std in the univariate case.
    """
    samplers_coords = np.array([0, 1, 2, 3, 4])
    mu = 0.5  # Progress as a fraction between 0 and 1
    std = 1

    weights = get_sampler_weights_gaussian(samplers_coords, mu, std)

    # Expected weights calculated using norm.pdf
    min_position = np.min(samplers_coords)
    max_position = np.max(samplers_coords)
    current_mu = mu * (max_position - min_position) + min_position
    expected_weights = norm.pdf(samplers_coords, loc=current_mu, scale=std)
    expected_weights /= np.sum(expected_weights)

    # np.testing.assert_almost_equal(weights, expected_weights, decimal=6)
    assert np.allclose(weights, expected_weights, atol=1e-6)
