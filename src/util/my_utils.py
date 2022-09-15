from typing import Union

import numpy as np
import scipy.stats.qmc
import torch


def to_float_tensor(arr: np.ndarray) -> torch.Tensor:
 
    return torch.tensor(arr.copy(), dtype=torch.float64)


def copy_num(x, length: int) -> np.ndarray:
   
    if np.array(x).size == 1:
        return np.full(length, x)
    else:
        return x


def make_slices(each_step: list[int]) -> list[slice]:
    

    i = 0
    res = []
    for step in each_step:
        res.append(slice(i, i + step))
        i += step
    return res


class LHS:
   

    def __init__(self, seed: Union[int, np.random.Generator] = None):
       
        if seed is None:
            generator = np.random.default_rng()
        elif isinstance(seed, np.random.Generator):
            generator = seed
        else:
            generator = np.random.default_rng(seed)
        self.generator: np.random.Generator = generator
        return

    def generate(self, bounds: list[tuple], n_sample: int) -> np.ndarray:
        
        dim = len(bounds)
        sampler = scipy.stats.qmc.LatinHypercube(d=dim, seed=self.generator)
        sample = sampler.random(n_sample)
        bound_arr = np.array(bounds)
        scaled_sample = scipy.stats.qmc.scale(sample, l_bounds=bound_arr[:, 0], u_bounds=bound_arr[:, 1])
        return scaled_sample


class FunctionValue:
  
    def __init__(self, value, x, require_update: bool = False):
       
        self.value = value
        self.x = x
        self.need_update = require_update
        return
