import os

# from autora.experimentalist.progressive_simple import progressive_simple_sample
# # from confirmation import confirmation_sample
# from autora.experimentalist.confirmation import confirmation_sample

# autora state
from autora.state import State, StandardState, on_state, estimator_on_state, Delta, VariableCollection

# experimentalist
from autora.experimentalist.grid import grid_pool
from autora.experimentalist.random import random_pool, random_sample

# theorist
from autora.theorist.bms import BMSRegressor

# sklearn
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# general
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# SET UP STATE
# Here, we use a non-standard State to be able to use a multiple models
@dataclass(frozen=True)
class CustomState(State):
    variables: Optional[VariableCollection] = field(default=None, metadata={"delta": "replace"})
    conditions: Optional[pd.DataFrame] = field(default=None, metadata={"delta": "replace", "converter": pd.DataFrame})
    experiment_data: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "extend", "converter": pd.DataFrame}
    )
    models_bms: List[BaseEstimator] = field(
        default_factory=list,
        metadata={"delta": "extend"},
    )
    models_lr: List[BaseEstimator] = field(
        default_factory=list,
        metadata={"delta": "extend"},
    )
    models_polyr: List[BaseEstimator] = field(
        default_factory=list,
        metadata={"delta": "extend"},
    )


# state wrapper for all theorists
@on_state()
def theorists_on_state(experiment_data, variables, bms_epochs):

    # extract conditions X and observations y from experiment data
    ivs = [iv.name for iv in variables.independent_variables]
    dvs = [dv.name for dv in variables.dependent_variables]
    X = experiment_data[ivs]
    y = experiment_data[dvs]

    # initialize and fit theorists
    theorist_bms = BMSRegressor(epochs=bms_epochs)
    theorist_polyr = PolynomialRegressor()
    theorist_lr = linear_model.LinearRegression()

    return Delta(
        models_bms=[theorist_bms.fit(X, y)], models_lr=[theorist_lr.fit(X, y)], models_polyr=[theorist_polyr.fit(X, y)]
    )


# state wrapper for grid pooler experimentalist (generates a grid of experiment conditions)
@on_state()
def grid_pool_on_state(variables):
    return Delta(conditions=grid_pool(variables))


# state wrapper for random pooler experimentalist (generates a pool of experiment conditions)
@on_state()
def random_pool_on_state(variables, num_samples, random_state=None):
    return Delta(conditions=random_pool(variables, num_samples, random_state))


# state wrapper for random experimentalist (samples experiment conditions from a set of conditions)
@on_state()
def random_sample_on_state(conditions, all_conditions, num_samples, random_state=None):
    return Delta(conditions=random_sample(all_conditions, num_samples, random_state))


# state wrapper for synthetic experiment runner
@on_state()
def run_experiment_on_state(conditions, experiment_runner):
    data = experiment_runner.run(conditions=conditions, added_noise=0.0)
    return Delta(experiment_data=data)


# the following function is used to compute the model performance
# on the validation set in terms of mean squared error
def get_validation_MSE(validation_experiment_data, working_state):
    ivs = [iv.name for iv in validation_experiment_data.variables.independent_variables]
    dvs = [dv.name for dv in validation_experiment_data.variables.dependent_variables]
    X = validation_experiment_data.experiment_data[ivs]
    y = validation_experiment_data.experiment_data[dvs]

    y_pred_bms = working_state.models_bms[-1].predict(X)
    y_pred_lr = working_state.models_lr[-1].predict(X)
    y_pred_polyr = working_state.models_polyr[-1].predict(X)

    MSE_bms = ((y - y_pred_bms) ** 2).mean()[0]
    MSE_lr = ((y - y_pred_lr) ** 2).mean()[0]
    MSE_polyr = ((y - y_pred_polyr) ** 2).mean()[0]

    min_MSE = min(MSE_bms, MSE_lr, MSE_polyr)

    return min_MSE


def plot_MSE(
    benchmark_MSE_log,
    working_MSE_log,
    title="Single Discovery Simulation",
    filename="plots/single_discovery_simulation.png",
):
    fig, ax = plt.subplots(1, 1)
    ax.plot(benchmark_MSE_log, label="benchmark_MSE_log")
    ax.plot(working_MSE_log, label="working_MSE_log")
    ax.set_xlabel("Sampled Data Points")
    ax.set_ylabel("MSE on Validation Set")
    ax.set_title(title)
    ax.legend()
    # plt.show()
    fig.savefig(filename)


def plot_mean_MSE_with_errorbar(
    benchmark_MSE_plot_data,
    working_MSE_plot_data,
    num_cycles,
    title="Averaged Discovery Simulations",
    filename="plots/averaged_discovery_simulations.png",
):
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(
        np.arange(num_cycles),
        np.mean(benchmark_MSE_plot_data, axis=0),
        yerr=np.std(benchmark_MSE_plot_data, axis=0),
        label="benchmark_MSE_log",
    )
    ax.errorbar(
        np.arange(num_cycles),
        np.mean(working_MSE_plot_data, axis=0),
        yerr=np.std(working_MSE_plot_data, axis=0),
        label="working_MSE_log",
    )
    ax.set_xlabel("Sampled Data Points")
    ax.set_ylabel("MSE on Validation Setc")
    ax.set_title(title)
    ax.legend()
    # plt.show()
    fig.savefig(filename)


class PolynomialRegressor:
    """
    This theorist fits a polynomial function to the data.
    """

    def __init__(self, degree: int = 3):
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()

    def fit(self, x, y):
        features = self.poly.fit_transform(x, y)
        self.model.fit(features, y)
        return self

    def predict(self, x):
        features = self.poly.fit_transform(x)
        return self.model.predict(features)

    def print_eqn(self):
        # Extract the coefficients and intercept
        coeffs = self.model.coef_
        intercept = self.model.intercept_

        # Handle multi-output case by iterating over each output's coefficients and intercept
        if coeffs.ndim > 1:
            for idx in range(coeffs.shape[0]):
                equation = f"y{idx+1} = {intercept[idx]:.3f}"
                feature_names = self.poly.get_feature_names_out()
                for coef, feature in zip(coeffs[idx], feature_names):
                    equation += f" + ({coef:.3f}) * {feature}"
                print(equation)
        else:
            equation = f"y = {intercept:.3f}"
            feature_names = self.poly.get_feature_names_out()
            for coef, feature in zip(coeffs, feature_names):
                equation += f" + ({coef:.3f}) * {feature}"
            print(equation)
