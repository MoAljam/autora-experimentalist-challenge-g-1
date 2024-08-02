# autora state
from autora.state import State, StandardState, on_state, estimator_on_state, Delta, VariableCollection

# experiment_runner
from autora.experiment_runner.synthetic.psychology.luce_choice_ratio import luce_choice_ratio
from autora.experiment_runner.synthetic.psychology.exp_learning import exp_learning
from autora.experiment_runner.synthetic.economics.expected_value_theory import expected_value_theory

# experimentalist
from autora.experimentalist.grid import grid_pool
from autora.experimentalist.random import random_pool, random_sample
from autora.experimentalist.falsification import falsification_sample
from autora.experimentalist.model_disagreement import model_disagreement_sample
from autora.experimentalist.uncertainty import uncertainty_sample
from autora.experimentalist.mixture import mixture_sample
from autora.experimentalist.novelty import novelty_sample
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

from autora.experimentalist.autora_experimentalist_example import our_sample
from confirmation import confirmation_sample

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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


# SET UP STATE
# Here, we use a non-standard State to be able to use a multiple models
@dataclass(frozen=True)
class CustomState(State):
    variables: Optional[VariableCollection] = field(
        default=None, metadata={"delta": "replace"}
    )
    conditions: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "replace", "converter": pd.DataFrame}
    )
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
              #  models_bms = [theorist_bms.fit(X, y)],
               models_lr=[theorist_lr.fit(X, y)],
               models_polyr=[theorist_polyr.fit(X, y)])



# def sample_method(index, methods_dict):
#     # sample a number from a normal distribution around the index
#     # then round it to the nearest integer
#     sample_normal = np.random.normal(index, 1)
#     sample_rounded = round(sample_normal)
#     # cap the values to the range of the keys
#     sample_capped = np.clip(sample_rounded, 0, len(methods_dict) - 1)
#     return methods_dict[sample_capped]



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

    # y_pred_bms = working_state.models_bms[-1].predict(X)
    y_pred_lr = working_state.models_lr[-1].predict(X)
    y_pred_polyr = working_state.models_polyr[-1].predict(X)

    # MSE_bms = ((y - y_pred_bms)**2).mean()[0]
    MSE_lr = ((y - y_pred_lr)**2).mean()[0]
    MSE_polyr = ((y - y_pred_polyr)**2).mean()[0]

    # min_MSE = min(MSE_bms, MSE_lr, MSE_polyr)
    min_MSE = min(MSE_lr, MSE_polyr)


    return min_MSE






# **** STATE WRAPPER FOR YOUR EXPERIMENTALIST ***
@on_state()
def custom_sample_on_state(experiment_data,
                           models_bms,
                           models_lr,
                           models_polyr,
                           all_conditions,
                           methods_dict=None,
                           cycle=None,
                           max_cycle=None,
                           num_samples=1,
                           random_state=None):

  # this is just an example where we integrate the model diagreement sampler
  # into the wrapper
  
  # conditions = model_disagreement_sample(
  #         all_conditions,
  #         models = [models_bms[-1], models_lr[-1], models_polyr[-1]],
  #         num_samples = num_samples
  #     )
  
  # map the cycle number to index in the range of the methods
  # def sample(
  #       conditions: Union[pd.DataFrame, np.ndarray],
  #       models: List,
  #       temprature: float,
  #       methods_space: dict,
  #       # reference_conditions: Union[pd.DataFrame, np.ndarray],
  #       num_samples: int = 1) -> pd.DataFrame:

  #novelity
  #model disagreement

  # print("## HERE", 0)

  tempreture = cycle/max_cycle
  tempreture = np.clip(tempreture, 0, 1)

    #   conditions: Union[pd.DataFrame, np.ndarray],
    # model,
    # reference_conditions: Union[pd.DataFrame, np.ndarray],
    # reference_observations: Union[pd.DataFrame, np.ndarray],
    # metadata: VariableCollection,
    # num_samples: Optional[int] = None,
    # training_epochs: int = 1000,
    # training_lr: float = 1e-3,
    # plot: bool = False,

  ivs = [iv.name for iv in experiment_data.variables.independent_variables]
  dvs = [dv.name for dv in experiment_data.variables.dependent_variables]
  X = experiment_data.experiment_data[ivs]
  Y = experiment_data.experiment_data[dvs]
  meta_data = VariableCollection(
    independent_variables = ivs,
    dependent_variables = dvs
  )

  ivs = [iv.name for iv in experiment_data.variables.independent_variables]
  dvs = [dv.name for dv in experiment_data.variables.dependent_variables]
  X = experiment_data.experiment_data[ivs]
  Y = experiment_data.experiment_data[dvs]


  methods_space = {
  0:{
    "func": novelty_sample,
    "name": "novelty",
    "params": {
      "conditions": all_conditions,
      "num_samples": num_samples,
      "reference_conditions": X
      }
  },
  # 1:{
  #   "func": falsification_sample,
  #   "name": "falsification",
  #   "params": {
  #     "conditions": all_conditions,
  #     "num_samples": num_samples,
  #     "reference_conditions": X,
  #     "reference_observations": Y, # TODO
  #     "metadata": meta_data,
  #     "model" : models_polyr[-1]
  #     }
  # },
  # 2:{
  #   "func": model_disagreement_sample,
  #   "name": "model_disagreement",
  #   "params": {
  #     "conditions": all_conditions,
  #     "num_samples": num_samples,
  #     # "models": [models_bms[-1], models_lr[-1], models_polyr[-1]]}
  #     "models": [models_lr[-1], models_polyr[-1]]}

  # },
  1:{
    "func": confirmation_sample,
    "name": "confirmation",
    "params": {
      "conditions": all_conditions,
      "num_samples": num_samples,
      "reference_conditions": X,
      "reference_observations": Y, # TODO
      "metadata": meta_data,
      "model" : models_polyr[-1]
      }
  }
  }
  
  
  conditions = our_sample(
          all_conditions,
          models = [models_lr[-1], models_polyr[-1]],
          temprature = tempreture,
          methods_space = methods_space,
          num_samples = num_samples
      )
  
  # print("## HERE", 3)

  return Delta(conditions=conditions)



def run_simulation(num_cycles, num_conditions_per_cycle, num_initial_conditions, bms_epochs, experiment_runner, sim=0):

  methods = [random_sample, falsification_sample]
  methods_key = range(len(methods))
  methods_dict = dict(zip(methods_key, methods))
  # VALIDATION STATE
  # at every step of our discovery process, we will evaluate the performance
  # of the theorist against the ground truth. Here, we will define the ground
  # truth as a grid of data points sampled across the domain of the experimental
  # design space. We will store this validation set in a separate validation states

  # create AutoRA state for validation purposes
  validation_conditions = CustomState(variables=experiment_runner.variables)
  validation_experiment_data = CustomState(variables=experiment_runner.variables)

  # our validation set will be consist of a grid of experiment conditons
  # across the entire experimental design domain
  validation_conditions = grid_pool_on_state(validation_conditions)
  validation_experiment_data = grid_pool_on_state(validation_experiment_data)
  validation_experiment_data = run_experiment_on_state(validation_experiment_data, experiment_runner=experiment_runner)


  benchmark_MSE_log = list()
  working_MSE_log = list()

  # INITIAL STATE
  # We begin our discovery experiment with randomly sampled data set for 10
  # conditions. We will use the same state for each experimentalist method.

  # create initial AutoRA state which we will use for our discovery expeirments
  initial_state = CustomState(variables=experiment_runner.variables)

  # we will initiate our discovery process with 10 randomly sampled experiment conditions
  initial_state = random_pool_on_state(initial_state,
                                      num_samples=num_initial_conditions,
                                      random_state = sim)

  # we obtain the corresponding experiment data
  initial_state = run_experiment_on_state(initial_state, experiment_runner=experiment_runner)

  # initialize benchmark state for random experimentalist
  benchmark_state = CustomState(**initial_state.__dict__)

  # initialize working state for your custom experimentalist
  working_state = CustomState(**initial_state.__dict__)

  # for each discovery cycle
  for cycle in range(num_cycles):

    print("SIMULATION " + str(sim)  + " / DISCOVERY CYCLE " + str(cycle))

    # first, we fit a model to the data
    print("Fitting models on benchmark state...")
    benchmark_state = theorists_on_state(benchmark_state, bms_epochs=bms_epochs)
    print("Fitting models on working state...")
    working_state = theorists_on_state(working_state, bms_epochs=bms_epochs)

    # now we can determine how well the models do on the validation set
    benchmark_MSE = get_validation_MSE(validation_experiment_data, benchmark_state)
    benchmark_MSE_log.append(benchmark_MSE)

    working_MSE = get_validation_MSE(validation_experiment_data, working_state)
    working_MSE_log.append(working_MSE)

    # then we determine the next experiment condition
    print("Sampling new experiment conditions...")
    benchmark_state = random_sample_on_state(benchmark_state,
                                              all_conditions=validation_conditions.conditions,
                                              num_samples=num_conditions_per_cycle)
    working_state = custom_sample_on_state(working_state,
                                           experiment_data=working_state,
                                            all_conditions=validation_conditions.conditions,
                                          num_samples=num_conditions_per_cycle,
                                          methods_dict=methods_dict,
                                          cycle=cycle,
                                          max_cycle=num_cycles)

    # print("## conditions: ", benchmark_state.conditions)

    print("Obtaining observations...")
    # we obtain the corresponding experiment data
    benchmark_state = run_experiment_on_state(benchmark_state, experiment_runner=experiment_runner)
    working_state = run_experiment_on_state(working_state, experiment_runner=experiment_runner)



  return benchmark_MSE_log, working_MSE_log, benchmark_state, working_state






if __name__ == "__main__":
  # meta parameters

  print("hhhh"*20)
  fig , ax = plt.subplots(1, 1)
  ax.plot([1, 2, 3, 4])
  plt.show()
  fig.show()
  print(type(fig))
  # DO NOT CHANGE THESE PARAMETERS
  num_cycles = 2
  num_conditions_per_cycle = 1
  num_initial_conditions = 1

  # YOU MAY CHANGE THESE PARAMETERS
  num_discovery_simulations = 10
  bms_epochs = 10 # Note, to speed things up, you can set bms_epochs = 10 or even bms_epochs = 1 (this will lead to poor performance of the BMS regressor but the other two theorists will still fit)

  # setting experiment runner and theorist
  experiment_runner = exp_learning()

  # run simulation
  benchmark_MSE_log, working_MSE_log, benchmark_state, working_state = run_simulation(num_cycles, num_conditions_per_cycle, num_initial_conditions, bms_epochs, experiment_runner)

  
  # lets plot the benchmark_MSE_log and the workign_MSE_log
  ax.plot(benchmark_MSE_log, label='benchmark_MSE_log')
  ax.plot(working_MSE_log, label='working_MSE_log')
  ax.set_xlabel('Sampled Data Points')
  ax.set_ylabel('MSE on Validation Set')
  ax.set_title('Single Discovery Simulation')
  ax.legend()
  plt.show()


  # we can also investigate the final state more closely
  # for example, these are all the experimental data collected
  # under random sampling:
  print(benchmark_state.experiment_data)
  # and for your custom experimentalist
  print(working_state.experiment_data)


  benchmark_MSE_plot_data = np.zeros([num_discovery_simulations, num_cycles])
  working_MSE_plot_data = np.zeros([num_discovery_simulations, num_cycles])

  for sim in range(num_discovery_simulations):
      benchmark_MSE_log, working_MSE_log, benchmark_state, working_state = run_simulation(num_cycles, num_conditions_per_cycle, num_initial_conditions, bms_epochs, experiment_runner, sim)

      benchmark_MSE_plot_data[sim, :] = benchmark_MSE_log
      working_MSE_plot_data[sim, :] = working_MSE_log

  # plot the data with standard error
  ax.errorbar(np.arange(num_cycles), np.mean(benchmark_MSE_plot_data, axis=0), yerr=np.std(benchmark_MSE_plot_data, axis=0), label='benchmark_MSE_log')
  ax.errorbar(np.arange(num_cycles), np.mean(working_MSE_plot_data, axis=0), yerr=np.std(working_MSE_plot_data, axis=0), label='working_MSE_log')
  ax.set_xlabel('Sampled Data Points')
  ax.set_ylabel('MSE on Validation Setc')
  ax.set_title('Averaged Discovery Simulations')
  ax.legend()
  plt.show()
  # plot without error bars
  ax.plot(np.mean(benchmark_MSE_plot_data, axis=0)[5:], label='benchmark_MSE_log')
  ax.plot(np.mean(working_MSE_plot_data, axis=0)[5:], label='working_MSE_log')
  ax.set_xlabel('Sampled Data Points')
  ax.set_ylabel('MSE on Validation Set')
  ax.set_title('Averaged Discovery Simulations')
  ax.legend()
  plt.show()