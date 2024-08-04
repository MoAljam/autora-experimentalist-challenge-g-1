from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.ERROR)

from autora.experimentalist.progressive import progressive_sample
from autora.experimentalist.progressive_multi import progressive_multi_sample
from autora.experimentalist.adaptable import adaptable_sample
from autora.experimentalist.confirmation import confirmation_sample, confirmation_score_sample
from tqdm import tqdm


from utils import (
    CustomState,
    grid_pool_on_state,
    random_pool_on_state,
    random_sample_on_state,
    run_experiment_on_state,
    get_validation_MSE,
    theorists_on_state,
    plot_MSE,
    plot_mean_MSE_with_errorbar,
)

# autora state
from autora.state import State, StandardState, on_state, estimator_on_state, Delta, VariableCollection

# experiment_runner
from autora.experiment_runner.synthetic.psychology.luce_choice_ratio import luce_choice_ratio
from autora.experiment_runner.synthetic.psychology.exp_learning import exp_learning
from autora.experiment_runner.synthetic.psychology.q_learning import q_learning
from autora.experiment_runner.synthetic.economics.expected_value_theory import expected_value_theory
from autora.experiment_runner.synthetic.economics.prospect_theory import prospect_theory
from autora.experiment_runner.synthetic.neuroscience.task_switching import task_switching
from autora.experiment_runner.synthetic.psychophysics.stevens_power_law import stevens_power_law
from autora.experiment_runner.synthetic.psychophysics.weber_fechner_law import weber_fechner_law

# experimentalist
from autora.experimentalist.grid import grid_pool
from autora.experimentalist.random import random_pool, random_sample
from autora.experimentalist.falsification import falsification_sample, falsification_score_sample
from autora.experimentalist.model_disagreement import model_disagreement_sample, model_disagreement_score_sample
from autora.experimentalist.uncertainty import uncertainty_sample
from autora.experimentalist.novelty import novelty_sample, novelty_score_sample
from autora.experimentalist.mixture import mixture_sample

# sklearn
from sklearn.model_selection import train_test_split

# general
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List
import os


# region custom_sample_on_state
# **** STATE WRAPPER FOR YOUR EXPERIMENTALIST ***
@on_state()
def custom_sample_on_state(
    experiment_data,
    variables,
    models_bms,
    models_lr,
    models_polyr,
    all_conditions,
    cycle=None,
    max_cycle=None,
    num_samples=1,
    random_state=None,
):
    # temperature(0-1) to determine the progress of the discovery process
    temperature = cycle / max_cycle
    temperature = np.clip(temperature, 0, 1)

    # get the input relevant to some of the samplers
    # independent and dependent variables for the metadata
    iv = variables.independent_variables
    dv = variables.dependent_variables
    meta_data = VariableCollection(independent_variables=iv, dependent_variables=dv)

    # reference conditions and observations
    ivs = [iv.name for iv in variables.independent_variables]
    dvs = [dv.name for dv in variables.dependent_variables]
    reference_conditions = experiment_data[ivs]
    reference_observations = experiment_data[dvs]

    # NOTE: the sampler is performeing a bit worse when including falsification and confirmation
    #       possiblly due to passing only one model to theses samplers
    #       while the performance is based on 3 models
    samplers = [
        {
            "func": novelty_score_sample,
            "name": "novelty",
            "params": {"reference_conditions": reference_conditions},
        },
        {
            "func": random_sample,
            "name": "random",
            "params": {},
        },
        {
            "func": falsification_score_sample,
            "name": "falsification",
            "params": {
                "reference_conditions": reference_conditions,
                "reference_observations": reference_observations,
                "metadata": meta_data,
                "model": models_polyr[-1],
            },
        },
        {
            "func": model_disagreement_score_sample,
            "name": "model_disagreement",
            "params": {
                "models": [models_bms[-1], models_lr[-1], models_polyr[-1]],
            },
        },
        {
            "func": confirmation_score_sample,
            "name": "confirmation",
            "params": {
                "reference_conditions": reference_conditions,
                "reference_observations": reference_observations,
                "metadata": meta_data,
                "model": models_polyr[-1],
            },
        },
    ]
    samplers_coords = [0, 1, 3, 4, 6]  # optional

    new_conditions = adaptable_sample(
        conditions=all_conditions,
        models=models_polyr,
        samplers=samplers,
        num_samples=num_samples,
        samplers_coords=samplers_coords,
        sensitivity=10,
        plot=True,
    )

    # new_conditions = progressive_sample(
    #     conditions=all_conditions,
    #     num_samples=num_samples,
    #     # models=[models_lr[-1], models_polyr[-1]],
    #     temprature=temperature,
    #     samplers=samplers,
    #     # samplers_coords=samplers_coords,
    # )

    return Delta(conditions=new_conditions)


# region run simulation
def run_simulation(num_cycles, num_conditions_per_cycle, num_initial_conditions, bms_epochs, experiment_runner, sim=0):
    # VALIDATION STATE
    # at every step of our discovery process, we will evaluate the performance
    # of the theorist against the ground truth. Here, we will define the ground
    # truth as a grid of data points sampled across the domain of the experimental
    # design space. We will store this validation set in a separate validation states

    # create AutoRA state for validation purposes
    # validation_conditions = CustomState(variables=experiment_runner.variables)
    # validation_experiment_data = CustomState(variables=experiment_runner.variables)

    # # our validation set will be consist of a grid of experiment conditons
    # # across the entire experimental design domain
    # validation_conditions = grid_pool_on_state(validation_conditions)
    # validation_experiment_data = grid_pool_on_state(validation_experiment_data)
    # validation_experiment_data = run_experiment_on_state(
    #     validation_experiment_data, experiment_runner=experiment_runner
    # )

    with ThreadPoolExecutor(max_workers=2) as executor:
        validation_conditions_future = executor.submit(
            grid_pool_on_state, CustomState(variables=experiment_runner.variables)
        )
        validation_experiment_data_future = executor.submit(
            grid_pool_on_state, CustomState(variables=experiment_runner.variables)
        )

    validation_conditions = validation_conditions_future.result()
    validation_experiment_data = run_experiment_on_state(
        validation_experiment_data_future.result(), experiment_runner=experiment_runner
    )

    benchmark_MSE_log = list()
    working_MSE_log = list()

    # INITIAL STATE
    # We begin our discovery experiment with randomly sampled data set for 10
    # conditions. We will use the same state for each experimentalist method.

    # create initial AutoRA state which we will use for our discovery expeirments
    initial_state = CustomState(variables=experiment_runner.variables)

    # we will initiate our discovery process with 10 randomly sampled experiment conditions
    initial_state = random_pool_on_state(initial_state, num_samples=num_initial_conditions, random_state=sim)

    # we obtain the corresponding experiment data
    initial_state = run_experiment_on_state(initial_state, experiment_runner=experiment_runner)

    # initialize benchmark state for random experimentalist
    benchmark_state = CustomState(**initial_state.__dict__)

    # initialize working state for your custom experimentalist
    working_state = CustomState(**initial_state.__dict__)

    # for each discovery cycle
    for cycle in tqdm(range(num_cycles), leave=False, desc="discovery cycles"):

        # print("SIMULATION " + str(sim) + " / DISCOVERY CYCLE " + str(cycle))

        # first, we fit a model to the data
        # print("Fitting models on benchmark state...")
        # benchmark_state = theorists_on_state(benchmark_state, bms_epochs=bms_epochs)
        # print("Fitting models on working state...")
        # working_state = theorists_on_state(working_state, bms_epochs=bms_epochs)

        with ThreadPoolExecutor(max_workers=2) as executor:
            benchmark_future = executor.submit(theorists_on_state, benchmark_state, bms_epochs=bms_epochs)
            working_future = executor.submit(theorists_on_state, working_state, bms_epochs=bms_epochs)

        benchmark_state = benchmark_future.result()
        working_state = working_future.result()

        # now we can determine how well the models do on the validation set
        # benchmark_MSE = get_validation_MSE(validation_experiment_data, benchmark_state)
        # benchmark_MSE_log.append(benchmark_MSE)

        # working_MSE = get_validation_MSE(validation_experiment_data, working_state)
        # working_MSE_log.append(working_MSE)

        # MSE calculation in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            benchmark_MSE_future = executor.submit(get_validation_MSE, validation_experiment_data, benchmark_state)
            working_MSE_future = executor.submit(get_validation_MSE, validation_experiment_data, working_state)

        benchmark_MSE_log.append(benchmark_MSE_future.result())
        working_MSE_log.append(working_MSE_future.result())

        # then we determine the next experiment condition
        # print("Sampling new experiment conditions...")
        # benchmark_state = random_sample_on_state(
        #     benchmark_state, all_conditions=validation_conditions.conditions, num_samples=num_conditions_per_cycle
        # )
        # working_state = custom_sample_on_state(
        #     working_state,
        #     all_conditions=validation_conditions.conditions,
        #     num_samples=num_conditions_per_cycle,
        #     cycle=cycle,
        #     max_cycle=num_cycles,
        # )

        # print("Obtaining observations...")
        # # we obtain the corresponding experiment data
        # benchmark_state = run_experiment_on_state(benchmark_state, experiment_runner=experiment_runner)
        # working_state = run_experiment_on_state(working_state, experiment_runner=experiment_runner)

        with ThreadPoolExecutor(max_workers=2) as executor:
            benchmark_sample_future = executor.submit(
                random_sample_on_state,
                benchmark_state,
                all_conditions=validation_conditions.conditions,
                num_samples=num_conditions_per_cycle,
            )
            working_sample_future = executor.submit(
                custom_sample_on_state,
                working_state,
                all_conditions=validation_conditions.conditions,
                num_samples=num_conditions_per_cycle,
                cycle=cycle,
                max_cycle=num_cycles,
            )

        benchmark_state = benchmark_sample_future.result()
        working_state = working_sample_future.result()

        with ThreadPoolExecutor(max_workers=2) as executor:
            benchmark_experiment_future = executor.submit(
                run_experiment_on_state, benchmark_state, experiment_runner=experiment_runner
            )
            working_experiment_future = executor.submit(
                run_experiment_on_state, working_state, experiment_runner=experiment_runner
            )

        benchmark_state = benchmark_experiment_future.result()
        working_state = working_experiment_future.result()

    return benchmark_MSE_log, working_MSE_log, benchmark_state, working_state


# region main
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    # DO NOT CHANGE THESE PARAMETERS
    # meta parameters
    num_cycles = 20
    num_cycles_weber = num_cycles * 2  # weber fechner law is slower to converge in both cases
    num_conditions_per_cycle = 1
    num_initial_conditions = 1

    # YOU MAY CHANGE THESE PARAMETERS
    num_discovery_simulations = 10
    bms_epochs = 1  # Note, to speed things up, you can set bms_epochs = 10 or even bms_epochs = 1 (this will lead to poor performance of the BMS regressor but the other two theorists will still fit)

    experiment_runners = [
        # psychology
        # luce_choice_ratio(),
        # exp_learning(),
        # economics
        # expected_value_theory(),
        prospect_theory(),
        # neuroscience
        # task_switching(),
        # psychophysics
        weber_fechner_law(),
    ]

    for experiment_runner in tqdm(experiment_runners, leave=True, desc="experiment runners"):
        print("## Running simulation for " + experiment_runner.name)
        num_cycle_to_run = num_cycles
        if experiment_runner.name == weber_fechner_law().name:
            num_cycle_to_run = num_cycles_weber

        benchmark_MSE_log, working_MSE_log, benchmark_state, working_state = run_simulation(
            num_cycle_to_run, num_conditions_per_cycle, num_initial_conditions, bms_epochs, experiment_runner
        )
        plot_MSE(
            benchmark_MSE_log,
            working_MSE_log,
            title="Single Discovery Simulation for " + experiment_runner.name,
            filename="plots/single_sim_" + experiment_runner.name + ".png",
        )

        print("## Finished simulation for " + experiment_runner.name)
        print("## -----------------------------------------")

    # exit()
    for experiment_runner in tqdm(experiment_runners, leave=True, desc="experiment runners"):
        print("## Running simulation for " + experiment_runner.name)
        num_cycle_to_run = num_cycles
        if experiment_runner.name == weber_fechner_law().name:
            num_cycle_to_run = num_cycles_weber

        benchmark_MSE_plot_data = np.zeros([num_discovery_simulations, num_cycle_to_run])
        working_MSE_plot_data = np.zeros([num_discovery_simulations, num_cycle_to_run])

        for sim in tqdm(range(num_discovery_simulations), leave=False, desc="discovery simulations"):
            benchmark_MSE_log, working_MSE_log, benchmark_state, working_state = run_simulation(
                num_cycle_to_run, num_conditions_per_cycle, num_initial_conditions, bms_epochs, experiment_runner, sim
            )

            benchmark_MSE_plot_data[sim, :] = benchmark_MSE_log
            working_MSE_plot_data[sim, :] = working_MSE_log

        plot_mean_MSE_with_errorbar(
            benchmark_MSE_plot_data,
            working_MSE_plot_data,
            num_cycle_to_run,
            title="Averaged Discovery Simulations for " + experiment_runner.name,
            filename="plots/avg_sims_" + experiment_runner.name + ".png",
        )

        print("## Finished simulation for " + experiment_runner.name)
        print("## -----------------------------------------")
