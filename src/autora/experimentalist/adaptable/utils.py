import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm, uniform


# jensen_shannon_divergence as the square of jensen shannon distance
def jensen_shannon_divergence(p, q):
    return jensenshannon(p, q) ** 2


# helper class to define a uniform distribution on a range
# enables modularity for calculating pdf of multiple distributions
class UniformOnRange:
    """
    (helper class) uniform distribution on a specified range with a pdf method
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def pdf(self, x):
        return uniform.pdf(x, loc=self.min_val, scale=self.max_val - self.min_val)

    def __call__(self, x):
        return self.pdf(x)


def exponential_bias(x, rate=4):
    return 1 - np.exp(-rate * x)


def logistic_bias(x, growth_rate=20, midpoint=0.5):
    return 1 / (1 + np.exp(-growth_rate * (x - midpoint)))


def polynomial_bias(x, power=0.5):
    return x**power
