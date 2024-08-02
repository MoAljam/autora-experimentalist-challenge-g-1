"""
Example Experimentalist
"""
import numpy as np
import pandas as pd

from typing import Union, List

def meta_controll(conditions: Union[pd.DataFrame, np.ndarray], models: List) -> pd.DataFrame:
    return np.random.rand()

def sample_method(index, methods_space):
    # sample a number from a normal distribution around the index
    # then round it to the nearest integer
    # print("###### sample_method")
    methods_indecies = np.array(list(methods_space.keys()))
    sample_normal = np.random.normal(index, 0.2)
    sample_rounded = round(sample_normal)
    # cap the values to the range of the keys
    sample_capped = np.clip(sample_rounded, np.min(methods_indecies), np.max(methods_indecies))

    # pick the method with the index nearest to the sample_clipped
    method_index = methods_indecies[np.argmin(np.abs(methods_indecies - sample_capped))]
    # print("## method_index:", method_index)
    
    return methods_space[method_index]['func'], method_index

def our_sample(
        conditions: Union[pd.DataFrame, np.ndarray],
        models: List,
        temprature: float,
        methods_space: dict,
        # reference_conditions: Union[pd.DataFrame, np.ndarray],
        num_samples: int = 1) -> pd.DataFrame:
    # x = meta_controll(conditions, models)
    # high score 
    # -
    # low score
    # -

    ## 
    # samplers: [(fun, name, [weights]), ... ]
    # params: {name: {}, ...}
    # samplers = []
    # params = {}
    # temprature = 1
    ##
    # print("## temprature:", temprature)
    # print("## methods_space:", methods_space)
    # print("models:", models)
    # print("conditions:", conditions)

    methods_keys = np.array(list(methods_space.keys()))
    index_range = np.arange(methods_keys.min(), methods_keys.max())
    index = index_range[int(len(index_range)*temprature)]


    method, method_index = sample_method(index, methods_space)

    # print("## index:", index)
    # print("## method_index:", method_index)
    # print("## method:", method)
    # print("## methods_space[index]:", methods_space[index])
    print("## method name:", methods_space[method_index]["name"])
    print("## method:", method)

    new_conditions = method(**(methods_space[method_index]["params"]))
    return new_conditions
    # return new_conditions[:num_samples]
