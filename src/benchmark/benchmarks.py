import functools as ft
import math
import os
import pathlib
import subprocess
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, Any, Literal

import gpytorch.likelihoods
import joblib
import numpy as np
import pandas as pd
import pickle
import scipy.interpolate
import scipy.optimize
import scipy.special
import sklearn
import sklearn.kernel_approximation
import torch

from optimize import MultistartLBFGS
from util import LHS

torch.set_default_dtype(torch.float64)


if os.environ['HOME'] + '/local/bin' not in os.environ['PATH']:
    os.environ['PATH'] = os.environ['HOME'] + '/local/bin:' + os.environ['PATH']



__n_cpu: int = 1


def set_n_cpu(n: int):
   
    global __n_cpu
    __n_cpu = n
    return


def get_n_cpu() -> int:
   
    global __n_cpu
    return __n_cpu


class SynthFunction(ABC):

    def __init__(self, bounds: list[tuple[Any, Any]], minmax_range: Optional[tuple[Any, Any]] = None):
        self.__bounds = bounds
        self.__scale_min = None  
        self.__scale_max = None  
        self.__scaled: bool = False  
        if minmax_range is not None:
            self.__scaled = True
            self.__scale_min = float(minmax_range[0])
            self.__scale_max = float(minmax_range[1])
        return

    @property
    def bounds(self) -> list[tuple[Any, Any]]:
        return self.__bounds

    @property
    def scaled(self) -> bool:
        return self.__scaled

    @property
    def scale_min(self) -> float:
        return self.__scale_min

    @property
    def scale_max(self) -> float:
        return self.__scale_max

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def default_bounds(self) -> list[tuple[Any, Any]]:
        pass

    @property
    @abstractmethod
    def raw_min(self) -> float:
        pass

    @property
    @abstractmethod
    def raw_max(self) -> float:
        pass

    @property
    def min(self) -> float:
        if self.scaled:
            return self.scale_min
        return self.raw_min

    @property
    def max(self) -> float:
        if self.scaled:
            return self.scale_max
        return self.raw_max

    def _min_max_scale(self, f_value: torch.Tensor):
     
        if not self.scaled:
            return f_value
        scale01 = (f_value - self.raw_min) / (self.raw_max - self.raw_min)
        return scale01 * (self.scale_max - self.scale_min) + self.scale_min

    @abstractmethod
    def eval_f(self, x: np.ndarray, with_grad: bool = False):
     
        pass


class Constant(SynthFunction):
    """定数関数."""

    def __init__(self, dim: int, bounds: list[tuple[Any, Any]] = None, value: float = 0.):
      
        self.__default_bounds = [(-10., 10.) for _ in range(dim)]
        if bounds is None:
            bounds = self.__default_bounds
        super().__init__(bounds=bounds)
        self.__dim = dim
        self.__value = float(value)
        self.__raw_min = self.__value
        self.__raw_max = self.__value
        return

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    @property
    def raw_min(self) -> float:
        return self.__raw_min

    @property
    def raw_max(self) -> float:
        return self.__raw_max

    def eval_f(self, x: np.ndarray, with_grad: bool = False):
        assert x.ndim == 2 and x.shape[1] == self.dim
        if with_grad:
            return np.array([self.__value]).flatten(), np.zeros(x.size).flatten()
        else:
            return np.full((x.shape[0], 1), self.__value)


class GPTest(SynthFunction):
    

    def __init__(self, dim: int, seed: Optional[int] = None,
                 ls: float = 4., var: float = (10 / 1.96) ** 2, n_component: int = 1000,
                 bounds: list[tuple[Any, Any]] = None, minmax_range: tuple[Any, Any] = None):
        self.__default_bounds = [(-10., 10.) for _ in range(dim)]
        if bounds is None:
            bounds = self.__default_bounds
        super().__init__(bounds=bounds, minmax_range=minmax_range)
        self.__dim = dim
        self.__seed = seed
        self.__ls = ls
        self.__var = var
        self.__n_component = n_component
        self.__rng = np.random.default_rng(seed)

        
        rbf_sampler = sklearn.kernel_approximation.RBFSampler(gamma=1 / (2 * (self.__ls ** 2)),
                                                              random_state=np.random.RandomState(self.__seed),
                                                              n_components=self.__n_component)
        rbf_sampler.fit(np.zeros((1, self.__dim)))
        self.__feature_omega = torch.from_numpy(rbf_sampler.random_weights_.copy())
        self.__feature_offset = torch.from_numpy(rbf_sampler.random_offset_.copy())
        np_w = self.__rng.normal(loc=0, scale=np.sqrt(self.__var), size=(self.__n_component, 1))
        self.__w = torch.from_numpy(np_w)  
        self.__raw_min = None
        self.__raw_max = None
        self.__post_init()
        return

    def __post_init(self):
        optimizer = MultistartLBFGS(print_detail='error', n_parallel=get_n_cpu(), seed=0)
        result = optimizer.run(func=ft.partial(self.__opt_helper, with_grad=True),
                               bounds=self.bounds, jac=True, minimize=True,
                               f_to_init_eval=ft.partial(self.__opt_helper, with_grad=False))
        self.__raw_min = float(result.fun)
        result = optimizer.run(func=ft.partial(self.__opt_helper, with_grad=True),
                               bounds=self.bounds, jac=True, minimize=False,
                               f_to_init_eval=ft.partial(self.__opt_helper, with_grad=False))
        self.__raw_max = float(result.fun)
        return

    def __opt_helper(self, x_, with_grad: bool, negate=False):
        
        x_ = torch.tensor(x_, dtype=torch.float64, requires_grad=with_grad)
        if x_.ndim == 1:
            x_ = x_.reshape(1, self.__dim)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x_)
            if negate:
                res *= -1
        if with_grad:
            grad = torch.autograd.grad(res, x_)[0]
            return res.detach().clone().numpy().flatten(), grad.detach().clone().numpy().flatten()
        else:
            return res.detach().clone().numpy().flatten()

    def __to_rff(self, x: torch.Tensor) -> torch.Tensor:
       
        assert x.ndim == 2
        projection = x @ self.__feature_omega + self.__feature_offset
        projection = torch.cos(projection)
        projection *= (2. ** 0.5) / (self.__n_component ** 0.5)
        return projection  

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    @property
    def raw_min(self) -> float:
        return self.__raw_min

    @property
    def raw_max(self) -> float:
        return self.__raw_max

    def __eval_raw_f(self, x: torch.Tensor) -> torch.Tensor:
     
        assert x.ndim == 2, 'x.ndim={}'.format(x.ndim)
        res = self.__to_rff(x) @ self.__w
        return res

    def eval_f(self, x: np.ndarray, with_grad: bool = False):
        x = torch.tensor(x.copy(), dtype=torch.float64, requires_grad=with_grad)
        assert x.ndim == 2
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x)
            res = self._min_max_scale(res)
        if with_grad:
            grad = torch.autograd.grad(res, x)[0]
            return res.detach().clone().numpy(), grad.detach().clone().numpy()
        else:
            return res.clone().numpy().reshape(-1, 1)


class NegAckley(SynthFunction):
    
    def __init__(self, dim: int, bounds: list[tuple[Any, Any]] = None, minmax_range=None,
                 a: float = 20., b: float = 0.2, c: float = 2 * math.pi):
      
        self.__default_bounds = [(-2., 2.) for _ in range(dim)]
        if bounds is None:
            bounds = self.__default_bounds
        super().__init__(bounds=bounds, minmax_range=minmax_range)
        self.__dim = dim
        self.__raw_min = None
        self.__raw_max = 0
        self.a = torch.tensor(a, dtype=torch.float64)
        self.b = torch.tensor(b, dtype=torch.float64)
        self.c = torch.tensor(c, dtype=torch.float64)
        self.__post_init()
        return

    def __post_init(self):
        optimizer = MultistartLBFGS(print_detail='error', n_parallel=get_n_cpu(), seed=0)
        result = optimizer.run(func=ft.partial(self.__opt_helper, with_grad=True), bounds=self.bounds, jac=True,
                               minimize=True,
                               f_to_init_eval=ft.partial(self.__opt_helper, with_grad=False))
        self.__raw_min = float(result.fun)
        return

    def __opt_helper(self, x_, with_grad: bool, negate=False):
       
        x_ = torch.tensor(x_, dtype=torch.float64, requires_grad=with_grad)
        if x_.ndim == 1:
            x_ = x_.reshape(1, self.__dim)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x_)
            if negate:
                res *= -1
        if with_grad:
            grad = torch.autograd.grad(res, x_)[0]
            return res.detach().clone().numpy().flatten(), grad.detach().clone().numpy().flatten()
        else:
            return res.detach().clone().numpy().flatten()

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    @property
    def raw_min(self) -> float:
        return self.__raw_min

    @property
    def raw_max(self) -> float:
        return self.__raw_max

    def __eval_raw_f(self, x: torch.Tensor) -> torch.Tensor:
        
        assert x.ndim == 2, 'x.ndim={}'.format(x.ndim)
        part1 = -self.a * torch.exp(-self.b / math.sqrt(self.dim) * torch.norm(x, dim=1))
        part2 = -(torch.exp(torch.mean(torch.cos(self.c * x), dim=1)))
        res = part1 + part2 + self.a + torch.tensor(math.e)
        return -res

    def eval_f(self, x: np.ndarray, with_grad: bool = False):
        assert x.ndim == 2
        x = torch.tensor(x.copy(), dtype=torch.float64, requires_grad=with_grad)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x)
            res = self._min_max_scale(res)
        if with_grad:
            grad = torch.autograd.grad(res, x)[0]
            return res.detach().clone().numpy(), grad.detach().clone().numpy()
        else:
            return res.clone().numpy().reshape(-1, 1)


class NegRosenbrock(SynthFunction):
   
    def __init__(self, dim: int, bounds: list[tuple[Any, Any]] = None, minmax_range=None):
        
        self.__default_bounds = [(-2., 2.) for _ in range(dim)]
        if bounds is None:
            bounds = self.__default_bounds
        super().__init__(bounds=bounds, minmax_range=minmax_range)
        self.__dim = dim
        self.__raw_min = None
        self.__raw_max = 0
        self.__post_init()
        return

    def __post_init(self):
        optimizer = MultistartLBFGS(print_detail='error', n_parallel=get_n_cpu(), seed=0)
        result = optimizer.run(func=ft.partial(self.__opt_helper, with_grad=True), bounds=self.bounds, jac=True,
                               minimize=True,
                               f_to_init_eval=ft.partial(self.__opt_helper, with_grad=False))
        self.__raw_min = float(result.fun)
        return

    def __opt_helper(self, x_, with_grad: bool, negate=False):
        
        x_ = torch.tensor(x_, dtype=torch.float64, requires_grad=with_grad)
        if x_.ndim == 1:
            x_ = x_.reshape(1, self.__dim)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x_)
            if negate:
                res *= -1
        if with_grad:
            grad = torch.autograd.grad(res, x_)[0]
            return res.detach().clone().numpy().flatten(), grad.detach().clone().numpy().flatten()
        else:
            return res.detach().clone().numpy().flatten()

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    @property
    def raw_min(self) -> float:
        return self.__raw_min

    @property
    def raw_max(self) -> float:
        return self.__raw_max

    @staticmethod
    def __eval_raw_f(x: torch.Tensor) -> torch.Tensor:
        
        assert x.ndim == 2, 'x.ndim={}'.format(x.ndim)
        res = -torch.sum(
            100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2,
            dim=1,
        )
        return res

    def eval_f(self, x: np.ndarray, with_grad: bool = False):
        assert x.ndim == 2
        x = torch.tensor(x.copy(), dtype=torch.float64, requires_grad=with_grad)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x)
            res = self._min_max_scale(res)
        if with_grad:
            grad = torch.autograd.grad(res, x)[0]
            return res.detach().clone().numpy(), grad.detach().clone().numpy()
        else:
            return res.clone().numpy().reshape(-1, 1)


class NegLevy(SynthFunction):
   

    def __init__(self, dim: int, bounds: list[tuple[Any, Any]] = None, minmax_range=None):
       
        self.__default_bounds = [(-10., 10.) for _ in range(dim)]
        if bounds is None:
            bounds = self.__default_bounds
        super().__init__(bounds=bounds, minmax_range=minmax_range)
        self.__dim = dim
        self.__raw_min = None
        self.__raw_max = 0
        self.__post_init()
        return

    def __post_init(self):
        optimizer = MultistartLBFGS(print_detail='error', n_parallel=get_n_cpu(), seed=0)
        result = optimizer.run(func=ft.partial(self.__opt_helper, with_grad=True), bounds=self.bounds, jac=True,
                               minimize=True,
                               f_to_init_eval=ft.partial(self.__opt_helper, with_grad=False))
        self.__raw_min = float(result.fun)
        return

    def __opt_helper(self, x_, with_grad: bool, negate=False):
      
        x_ = torch.tensor(x_, dtype=torch.float64, requires_grad=with_grad)
        if x_.ndim == 1:
            x_ = x_.reshape(1, self.__dim)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x_)
            if negate:
                res *= -1
        if with_grad:
            grad = torch.autograd.grad(res, x_)[0]
            return res.detach().clone().numpy().flatten(), grad.detach().clone().numpy().flatten()
        else:
            return res.detach().clone().numpy().flatten()

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    @property
    def raw_min(self) -> float:
        return self.__raw_min

    @property
    def raw_max(self) -> float:
        return self.__raw_max

    def __eval_raw_f(self, x: torch.Tensor) -> torch.Tensor:
       
        assert x.ndim == 2, 'x.ndim={}'.format(x.ndim)

        w = 1. + (x - 1.) / 4.
        pi = torch.tensor(np.pi)
        res = torch.sin( pi * w[:,0] ) ** 2
        + torch.sum(
            (w[:, :-1] - 1.)**2 * (1. + 10. * torch.sin( pi * w[:, :-1] + 1.) **2),
            dim=1,
        )
        + (w[:, -1] - 1.) **2 * (1 + torch.sin(2 * pi * w[:, -1])**2)

        return -res

    def eval_f(self, x: np.ndarray, with_grad: bool = False):
        assert x.ndim == 2
        x = torch.tensor(x.copy(), dtype=torch.float64, requires_grad=with_grad)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x)
            res = self._min_max_scale(res)
        if with_grad:
            grad = torch.autograd.grad(res, x)[0]
            return res.detach().clone().numpy(), grad.detach().clone().numpy()
        else:
            return res.clone().numpy().reshape(-1, 1)


class NegSchwefel(SynthFunction):
   

    def __init__(self, dim: int, bounds: list[tuple[Any, Any]] = None, minmax_range=None):
      
        self.__default_bounds = [(-500., 500.) for _ in range(dim)]
        if bounds is None:
            bounds = self.__default_bounds
        super().__init__(bounds=bounds, minmax_range=minmax_range)
        self.__dim = dim
        self.__raw_min = None
        self.__raw_max = 0
        self.constant = torch.tensor(418.9829 * self.__dim, dtype=torch.float64)

        self.__post_init()
        return

    def __post_init(self):
        optimizer = MultistartLBFGS(print_detail='error', n_parallel=get_n_cpu(), seed=0)
        result = optimizer.run(func=ft.partial(self.__opt_helper, with_grad=True), bounds=self.bounds, jac=True,
                               minimize=True,
                               f_to_init_eval=ft.partial(self.__opt_helper, with_grad=False))
        self.__raw_min = float(result.fun)
        return

    def __opt_helper(self, x_, with_grad: bool, negate=False):
       
        x_ = torch.tensor(x_, dtype=torch.float64, requires_grad=with_grad)
        if x_.ndim == 1:
            x_ = x_.reshape(1, self.__dim)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x_)
            if negate:
                res *= -1
        if with_grad:
            grad = torch.autograd.grad(res, x_)[0]
            return res.detach().clone().numpy().flatten(), grad.detach().clone().numpy().flatten()
        else:
            return res.detach().clone().numpy().flatten()

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    @property
    def raw_min(self) -> float:
        return self.__raw_min

    @property
    def raw_max(self) -> float:
        return self.__raw_max

    def __eval_raw_f(self, x: torch.Tensor) -> torch.Tensor:
        
        assert x.ndim == 2, 'x.ndim={}'.format(x.ndim)

        res = self.constant - torch.sum( x * torch.sin(torch.sqrt(torch.abs(x))), dim=1,)

        return -res

    def eval_f(self, x: np.ndarray, with_grad: bool = False):
        assert x.ndim == 2
        x = torch.tensor(x.copy(), dtype=torch.float64, requires_grad=with_grad)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x)
            res = self._min_max_scale(res)
        if with_grad:
            grad = torch.autograd.grad(res, x)[0]
            return res.detach().clone().numpy(), grad.detach().clone().numpy()
        else:
            return res.clone().numpy().reshape(-1, 1)


class NegRastrigin(SynthFunction):
   

    def __init__(self, dim: int, bounds: list[tuple[Any, Any]] = None, minmax_range=None):
     
        self.__default_bounds = [(-5.12, 5.12) for _ in range(dim)]
        if bounds is None:
            bounds = self.__default_bounds
        super().__init__(bounds=bounds, minmax_range=minmax_range)
        self.__dim = dim
        self.__raw_min = None
        self.__raw_max = 0
        self.constant = torch.tensor(10 * self.__dim, dtype=torch.float64)

        self.__post_init()
        return

    def __post_init(self):
        optimizer = MultistartLBFGS(print_detail='error', n_parallel=get_n_cpu(), seed=0)
        result = optimizer.run(func=ft.partial(self.__opt_helper, with_grad=True), bounds=self.bounds, jac=True,
                               minimize=True,
                               f_to_init_eval=ft.partial(self.__opt_helper, with_grad=False))
        self.__raw_min = float(result.fun)
        return

    def __opt_helper(self, x_, with_grad: bool, negate=False):
        
        x_ = torch.tensor(x_, dtype=torch.float64, requires_grad=with_grad)
        if x_.ndim == 1:
            x_ = x_.reshape(1, self.__dim)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x_)
            if negate:
                res *= -1
        if with_grad:
            grad = torch.autograd.grad(res, x_)[0]
            return res.detach().clone().numpy().flatten(), grad.detach().clone().numpy().flatten()
        else:
            return res.detach().clone().numpy().flatten()

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    @property
    def raw_min(self) -> float:
        return self.__raw_min

    @property
    def raw_max(self) -> float:
        return self.__raw_max

    def __eval_raw_f(self, x: torch.Tensor) -> torch.Tensor:
       
        assert x.ndim == 2, 'x.ndim={}'.format(x.ndim)

        pi = torch.tensor(np.pi)
        res = self.constant + torch.sum( x**2 - 10. * torch.cos(2*pi*x), dim=1,)

        return - res

    def eval_f(self, x: np.ndarray, with_grad: bool = False):
        assert x.ndim == 2
        x = torch.tensor(x.copy(), dtype=torch.float64, requires_grad=with_grad)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x)
            res = self._min_max_scale(res)
        if with_grad:
            grad = torch.autograd.grad(res, x)[0]
            return res.detach().clone().numpy(), grad.detach().clone().numpy()
        else:
            return res.clone().numpy().reshape(-1, 1)



class NegSphere(SynthFunction):
   
    def __init__(self, dim: int, bounds: list[tuple[Any, Any]] = None, minmax_range=None):
     
        self.__default_bounds = [(-5.12, 5.12) for _ in range(dim)]
        if bounds is None:
            bounds = self.__default_bounds
        super().__init__(bounds=bounds, minmax_range=minmax_range)
        self.__dim = dim
        self.__raw_min = None
        self.__raw_max = 0

        self.__post_init()
        return

    def __post_init(self):
        optimizer = MultistartLBFGS(print_detail='error', n_parallel=get_n_cpu(), seed=0)
        result = optimizer.run(func=ft.partial(self.__opt_helper, with_grad=True), bounds=self.bounds, jac=True,
                               minimize=True,
                               f_to_init_eval=ft.partial(self.__opt_helper, with_grad=False))
        self.__raw_min = float(result.fun)
        return

    def __opt_helper(self, x_, with_grad: bool, negate=False):
       
        x_ = torch.tensor(x_, dtype=torch.float64, requires_grad=with_grad)
        if x_.ndim == 1:
            x_ = x_.reshape(1, self.__dim)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x_)
            if negate:
                res *= -1
        if with_grad:
            grad = torch.autograd.grad(res, x_)[0]
            return res.detach().clone().numpy().flatten(), grad.detach().clone().numpy().flatten()
        else:
            return res.detach().clone().numpy().flatten()

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    @property
    def raw_min(self) -> float:
        return self.__raw_min

    @property
    def raw_max(self) -> float:
        return self.__raw_max

    def __eval_raw_f(self, x: torch.Tensor) -> torch.Tensor:
        
        assert x.ndim == 2, 'x.ndim={}'.format(x.ndim)

        res = torch.sum( x**2, dim=1,)

        return - res

    def eval_f(self, x: np.ndarray, with_grad: bool = False):
        assert x.ndim == 2
        x = torch.tensor(x.copy(), dtype=torch.float64, requires_grad=with_grad)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x)
            res = self._min_max_scale(res)
        if with_grad:
            grad = torch.autograd.grad(res, x)[0]
            return res.detach().clone().numpy(), grad.detach().clone().numpy()
        else:
            return res.clone().numpy().reshape(-1, 1)





class NegBeale(SynthFunction):
   

    def __init__(self, dim: int, bounds: list[tuple[Any, Any]] = None, minmax_range=None):
      
        self.__default_bounds = [(-4.5, 4.5) for _ in range(dim)]
        if bounds is None:
            bounds = self.__default_bounds
        super().__init__(bounds=bounds, minmax_range=minmax_range)
        self.__dim = dim
        self.__raw_min = None
        self.__raw_max = 0
        self.__post_init()
        return

    def __post_init(self):
        optimizer = MultistartLBFGS(print_detail='error', n_parallel=get_n_cpu(), seed=0)
        result = optimizer.run(func=ft.partial(self.__opt_helper, with_grad=True), bounds=self.bounds, jac=True,
                               minimize=True,
                               f_to_init_eval=ft.partial(self.__opt_helper, with_grad=False))
        self.__raw_min = float(result.fun)
        return

    def __opt_helper(self, x_, with_grad: bool, negate=False):
       
        x_ = torch.tensor(x_, dtype=torch.float64, requires_grad=with_grad)
        if x_.ndim == 1:
            x_ = x_.reshape(1, self.__dim)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x_)
            if negate:
                res *= -1
        if with_grad:
            grad = torch.autograd.grad(res, x_)[0]
            return res.detach().clone().numpy().flatten(), grad.detach().clone().numpy().flatten()
        else:
            return res.detach().clone().numpy().flatten()

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    @property
    def raw_min(self) -> float:
        return self.__raw_min

    @property
    def raw_max(self) -> float:
        return self.__raw_max

    def __eval_raw_f(self, x: torch.Tensor) -> torch.Tensor:
        
        assert x.ndim == 2, 'x.ndim={}'.format(x.ndim)

        res = (1.5 - x[:,0] + x[:,0] * x[:,1])**2 + (2.25 - x[:,0] + x[:,0] * x[:,1]**2)**2 + (2.625 - x[:,0] + x[:,0] * x[:,1]**3)**2

        return -res

    def eval_f(self, x: np.ndarray, with_grad: bool = False):
        assert x.ndim == 2
        x = torch.tensor(x.copy(), dtype=torch.float64, requires_grad=with_grad)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x)
            res = self._min_max_scale(res)
        if with_grad:
            grad = torch.autograd.grad(res, x)[0]
            return res.detach().clone().numpy(), grad.detach().clone().numpy()
        else:
            return res.clone().numpy().reshape(-1, 1)



class NegMatyas(SynthFunction):
    

    def __init__(self, dim: int, bounds: list[tuple[Any, Any]] = None, minmax_range=None):
        
        self.__default_bounds = [(-10., 10.) for _ in range(dim)]
        if bounds is None:
            bounds = self.__default_bounds
        super().__init__(bounds=bounds, minmax_range=minmax_range)
        self.__dim = dim
        self.__raw_min = None
        self.__raw_max = 0
        self.__post_init()
        return

    def __post_init(self):
        optimizer = MultistartLBFGS(print_detail='error', n_parallel=get_n_cpu(), seed=0)
        result = optimizer.run(func=ft.partial(self.__opt_helper, with_grad=True), bounds=self.bounds, jac=True,
                               minimize=True,
                               f_to_init_eval=ft.partial(self.__opt_helper, with_grad=False))
        self.__raw_min = float(result.fun)
        return

    def __opt_helper(self, x_, with_grad: bool, negate=False):
        
        x_ = torch.tensor(x_, dtype=torch.float64, requires_grad=with_grad)
        if x_.ndim == 1:
            x_ = x_.reshape(1, self.__dim)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x_)
            if negate:
                res *= -1
        if with_grad:
            grad = torch.autograd.grad(res, x_)[0]
            return res.detach().clone().numpy().flatten(), grad.detach().clone().numpy().flatten()
        else:
            return res.detach().clone().numpy().flatten()

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    @property
    def raw_min(self) -> float:
        return self.__raw_min

    @property
    def raw_max(self) -> float:
        return self.__raw_max

    def __eval_raw_f(self, x: torch.Tensor) -> torch.Tensor:
       
        assert x.ndim == 2, 'x.ndim={}'.format(x.ndim)

        res = 0.26*(x[:,0]**2 + x[:,1]**2) - 0.48 * x[:,0] * x[:,1]
        return -res

    def eval_f(self, x: np.ndarray, with_grad: bool = False):
        assert x.ndim == 2
        x = torch.tensor(x.copy(), dtype=torch.float64, requires_grad=with_grad)
        with torch.set_grad_enabled(with_grad):
            res = self.__eval_raw_f(x)
            res = self._min_max_scale(res)
        if with_grad:
            grad = torch.autograd.grad(res, x)[0]
            return res.detach().clone().numpy(), grad.detach().clone().numpy()
        else:
            return res.clone().numpy().reshape(-1, 1)





class RealFunction(ABC):
   
    def __init__(self, bounds: list[tuple[Any, Any]]):
        
        self.__bounds = bounds
        return

    @property
    def bounds(self) -> list[tuple[Any, Any]]:
        return self.__bounds

    @property
    @abstractmethod
    def in_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def out_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def default_bounds(self) -> list[tuple[Any, Any]]:
        pass

    @abstractmethod
    def eval_f(self, x):
        
        pass


class PDiffusion1st(RealFunction):
   

    def __init__(self, bounds: list[tuple[Any, Any]] = None, seed: int = None,
                 log_fit: bool = True, shape='gaussian', parallel: int = 1):
       
        self.__default_bounds = [(700., 1050.),
                                 (100., 5000.),
                                 (19., np.log10(1.5e21))]
        if bounds is None:
            bounds = self.__default_bounds
        assert shape in {'gaussian', 'erfc'}
        super().__init__(bounds=bounds)
        self.shape = shape
        self.rng = np.random.default_rng(seed)
        self.__in_dim = 3
        self.__out_dim = 4
        self.lhs = LHS(self.rng)
        self.log_fit = log_fit
        self.parallel = parallel

        
        CPP = 10 ** np.linspace(13, 20.35, 200)
        CP = CPP + (2 * (1.6 * 10 ** -41) * CPP ** 3) / (1 - (1.6 * 10 ** -41) * CPP ** 2)
        self.f = scipy.interpolate.interp1d(CP, CPP, kind="quadratic")
        self.dt = 1e-1  
        self.dx = 1e-6  
        return

    @property
    def in_dim(self) -> int:
        return self.__in_dim

    @property
    def out_dim(self) -> int:
        return self.__out_dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    
    def __d(self, t, c):
        kB = 8.6171E-5  
        Di_P = 5.8 * 10 ** -5 * np.exp(-2.1 / (kB * t))
        Di_I = 2.3 * 10 ** -1 * np.exp(-2.6 / (kB * t))
        Di_PV = 7.6 * 10 ** 4 * np.exp(-5.2 / (kB * t))

        k = 0.5
        m = 2
        l = -1.8

        ni = 1.8 * 10 ** 21 * np.exp(-0.66 / (kB * t))
        n = (1 / 2) * (c + np.sqrt(c ** 2 + 4 * ni ** 2))

        h = 1.5

        Deff_Pi = Di_P * (n / ni) ** k * h
        Deff_I = Di_I * (n / ni) ** l * h
        Deff_PV = Di_PV * (self.f(c) / ni) ** m * h

        Deff_P = (Deff_Pi * Deff_I) / (Deff_Pi + Deff_I) + Deff_PV

        Csat = 4.1 * 10 ** 22 * np.exp(-0.44 / (kB * t)) * np.ones_like(c)
        Cmod = np.min(np.hstack((c.reshape(-1, 1), Csat.reshape(-1, 1))), axis=1)

        Deff = Deff_P * (Cmod / c)

        return Deff

    
    @staticmethod
    def __gauss(x, a, b, c, d, c_init):
        para = np.array([a, b, c, d])
        C_c = para[0] * np.exp(-np.square(x) / np.square(para[1]))
        C_c += para[2] * np.exp(-np.square(x) / np.square(para[3]))
        return np.max(np.hstack((C_c.reshape(-1, 1), c_init.reshape(-1, 1))), axis=1)

    @staticmethod
    def __erfc(x, a, b, c, d, c_init):
        para = np.array([a, b, c, d])
        C_c = para[0] * scipy.special.erfc(x / para[1]) + para[2] * scipy.special.erfc(x / para[3])
        return np.max(np.hstack((C_c.reshape(-1, 1), c_init.reshape(-1, 1))), axis=1)

    def eval_f(self, x):
        
        
        
        x = x.copy().reshape(3, )
        Temp1 = x[0] + 273  
        Time1 = x[1]
        C0 = np.power(10., x[2])  

        
        total_time1 = int(Time1 / self.dt)  

        total_x = 1.0e-3  
        total_xnum = int(total_x / self.dx)  
        C_init = np.ones(total_xnum) * 1E14  

        
        
        C = np.copy(C_init)
        C_ = np.copy(C_init)

        
        for i in range(total_time1):
            C[0] = C0
            C_[1:-1] = C[1:-1] + self.__d(Temp1, C[1:-1]) * (C[2:] - 2 * C[1:-1] + C[:-2]) / self.dx ** 2 * self.dt
            C_ = np.max(np.hstack((C_.reshape(-1, 1), C_init.reshape(-1, 1))), axis=1)
            C_[0] = C0
            C = np.copy(C_)

        x_plt = np.linspace(0, total_x, total_xnum) * 1E4  
        
        bounds = [[0.9 * C0, 0., 1e14, 0.001], [1.1 * C0, 5., C0 / 10, 5.]]  

        func_dict = {'gaussian': self.__gauss, 'erfc': self.__erfc}
        func = func_dict[self.shape]

        def shape_wrapper(_x, a, b, c, d):
            return func(_x, a=a, b=b, c=c, d=d, c_init=C_init)

        def log_shape_wrapper(_x, a, b, c, d):
            return np.log10(func(_x, a=a, b=b, c=c, d=d, c_init=C_init))

        
        lhs_bounds = [(bounds[0][i], bounds[1][i]) for i in range(len(bounds[0]))]
        para_set = self.lhs.generate(bounds=lhs_bounds, n_sample=100)
        para_set = np.vstack([para_set, np.array([C0, 0.05, 1E18, 0.2])])

        def fit(_para, _log_fit: bool = True):
            
            try:
                if _log_fit:
                    _popt, _ = scipy.optimize.curve_fit(log_shape_wrapper, x_plt, np.log10(C), bounds=bounds,
                                                        p0=_para)  
                    _loss = np.sum(np.square(np.log10(C) - log_shape_wrapper(x_plt, *_popt)))
                else:
                    _popt, _ = scipy.optimize.curve_fit(shape_wrapper, x_plt, C, bounds=bounds, p0=_para)  
                    _loss = np.sum(np.square(C - shape_wrapper(x_plt, *_popt)))
                return _loss, _popt
            except RuntimeError:
                return None

        p = joblib.Parallel(n_jobs=self.parallel)
        with joblib.parallel_backend('loky', inner_max_num_threads=1):
            results = p(joblib.delayed(fit)(para) for para in para_set)

        best_para = None
        min_loss = None

        
        for result in results:
            if result is None:
                continue
            loss = result[0]
            popt = result[1]
            if best_para is None or min_loss > loss:
                best_para = popt
                min_loss = loss

        if best_para is None:
            raise RuntimeError('curve_fit is failed for all initial points.')
        popt = best_para
        
        
        
        
        
        fit_param = popt.copy()
        
        
        fit_param = np.log10(fit_param)
        result = fit_param.reshape(1, 4)
        return result, {'profile': C, 'logC0': x[2]}


class PDiffusion2nd(RealFunction):
    

    def __init__(self, bounds: list[tuple[Any, Any]] = None, seed: int = None,
                 log_fit: bool = True, shape: str = 'gaussian', parallel: int = 1):
      
        
        
        self.__default_bounds = [(np.log10(0.9 * 1e19), np.log10(1.1 * 1.5e21)),
                                 (np.log10(0.003), np.log10(2.5)),
                                 (17.96198364, 20.17609126),
                                 (np.log10(0.003), np.log10(2.5)),
                                 (700., 1050.),
                                 (100., 5000.)]
        if bounds is None:
            bounds = self.__default_bounds
        assert shape in {'gaussian', 'erfc'}
        super().__init__(bounds=bounds)
        self.shape = shape
        self.rng = np.random.default_rng(seed)
        self.__in_dim = 2 + 4
        self.__out_dim = 4
        self.lhs = LHS(self.rng)
        self.log_fit = log_fit
        self.parallel = parallel

        
        CPP = 10 ** np.linspace(13, 20.35, 200)
        CP = CPP + (2 * (1.6 * 10 ** -41) * CPP ** 3) / (1 - (1.6 * 10 ** -41) * CPP ** 2)
        self.f = scipy.interpolate.interp1d(CP, CPP, kind="quadratic")
        self.dt = 1e-1  
        self.dx = 1e-6  
        return

    @property
    def in_dim(self) -> int:
        return self.__in_dim

    @property
    def out_dim(self) -> int:
        return self.__out_dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    def __d(self, t, c):
        kB = 8.6171E-5  
        Di_P = 5.8 * 10 ** -5 * np.exp(-2.1 / (kB * t))
        Di_I = 2.3 * 10 ** -1 * np.exp(-2.6 / (kB * t))
        Di_PV = 7.6 * 10 ** 4 * np.exp(-5.2 / (kB * t))

        k = 0.5
        m = 2
        l = -1.8

        ni = 1.8 * 10 ** 21 * np.exp(-0.66 / (kB * t))
        n = (1 / 2) * (c + np.sqrt(c ** 2 + 4 * ni ** 2))

        h = 1.5

        Deff_Pi = Di_P * (n / ni) ** k * h
        Deff_I = Di_I * (n / ni) ** l * h
        Deff_PV = Di_PV * (self.f(c) / ni) ** m * h

        Deff_P = (Deff_Pi * Deff_I) / (Deff_Pi + Deff_I) + Deff_PV

        Csat = 4.1 * 10 ** 22 * np.exp(-0.44 / (kB * t)) * np.ones_like(c)
        Cmod = np.min(np.hstack((c.reshape(-1, 1), Csat.reshape(-1, 1))), axis=1)

        Deff = Deff_P * (Cmod / c)

        return Deff

    
    @staticmethod
    def __gauss(x, a, b, c, d, c_init):
        para = np.array([a, b, c, d])
        C_c = para[0] * np.exp(-np.square(x) / np.square(para[1]))
        C_c += para[2] * np.exp(-np.square(x) / np.square(para[3]))
        return np.max(np.hstack((C_c.reshape(-1, 1), c_init.reshape(-1, 1))), axis=1)

    @staticmethod
    def __erfc(x, a, b, c, d, c_init):
        para = np.array([a, b, c, d])
        C_c = para[0] * scipy.special.erfc(x / para[1]) + para[2] * scipy.special.erfc(x / para[3])
        return np.max(np.hstack((C_c.reshape(-1, 1), c_init.reshape(-1, 1))), axis=1)

    def eval_f(self, x, info: dict = None):
     
        
        x = x.copy().reshape(6, )
        x[0] = np.power(10., x[0])
        x[1] = np.power(10., x[1])
        x[2] = np.power(10., x[2])
        x[3] = np.power(10., x[3])

        Temp2 = x[4] + 273
        Time2 = x[5]
        C0 = np.power(10., info['logC0'])  

        
        total_time2 = int(Time2 / self.dt)  

        total_x = 1.0e-3  
        total_xnum = int(total_x / self.dx)  
        C_init = np.ones(total_xnum) * 1E14  

        
        
        C = np.copy(info['profile'])
        C_ = np.copy(info['profile'])

        
        for i in range(total_time2):
            C[0] = C0
            C_[1:-1] = C[1:-1] + self.__d(Temp2, C[1:-1]) * (C[2:] - 2 * C[1:-1] + C[:-2]) / self.dx ** 2 * self.dt
            C_ = np.max(np.hstack((C_.reshape(-1, 1), C_init.reshape(-1, 1))), axis=1)
            C_[0] = C0
            C = np.copy(C_)

        x_plt = np.linspace(0, total_x, total_xnum) * 1E4  
        
        bounds = [[0.9 * C0, 0., 1e14, 0.001], [1.1 * C0, 5., 1.1 * C0 / 10, 5.]]  

        func_dict = {'gaussian': self.__gauss, 'erfc': self.__erfc}
        func = func_dict[self.shape]

        def shape_wrapper(_x, a, b, c, d):
            return func(_x, a=a, b=b, c=c, d=d, c_init=C_init)

        def log_shape_wrapper(_x, a, b, c, d):
            return np.log10(func(_x, a=a, b=b, c=c, d=d, c_init=C_init))

        
        lhs_bounds = [(bounds[0][i], bounds[1][i]) for i in range(len(bounds[0]))]
        para_set = self.lhs.generate(bounds=lhs_bounds, n_sample=100)
        para_set = np.vstack([para_set, np.array([C0, 0.05, 1e18, 0.2])])
        para_set = np.vstack([para_set, x[:4]])

        def fit(_para, _log_fit: bool = True):
            
            try:
                if _log_fit:
                    _popt, _ = scipy.optimize.curve_fit(log_shape_wrapper, x_plt, np.log10(C), bounds=bounds,
                                                        p0=_para)  
                    _loss = np.sum(np.square(np.log10(C) - log_shape_wrapper(x_plt, *_popt)))
                else:
                    _popt, _ = scipy.optimize.curve_fit(shape_wrapper, x_plt, C, bounds=bounds, p0=_para)  
                    _loss = np.sum(np.square(C - shape_wrapper(x_plt, *_popt)))
                return _loss, _popt
            except RuntimeError:
                return None

        p = joblib.Parallel(n_jobs=self.parallel)
        with joblib.parallel_backend('loky', inner_max_num_threads=1):
            results = p(joblib.delayed(fit)(para) for para in para_set)

        best_para = None
        min_loss = None

        
        for result in results:
            if result is None:
                continue
            loss = result[0]
            popt = result[1]
            if best_para is None or min_loss > loss:
                best_para = popt
                min_loss = loss

        if best_para is None:
            raise RuntimeError('curve_fit is failed for all initial points.')
        popt = best_para
        
        
        
        
        
        fit_param = popt.copy()
        
        
        fit_param = np.log10(fit_param)
        result = fit_param.reshape(1, 4)
        return result


class PDiffusion(RealFunction):
    

    def __init__(self, bounds: list[tuple[Any, Any]] = None, seed: int = None,
                 log_fit: bool = True, shape: str = 'gaussian', parallel: int = 1):
        self.__default_bounds = [(700., 1050.),
                                 (100., 5000.),
                                 (700., 1050.),
                                 (100., 5000.),
                                 (19., np.log10(1.5e21))]
        if bounds is None:
            bounds = self.__default_bounds
        assert shape in {'gaussian', 'erfc'}
        super().__init__(bounds=bounds)
        self.shape = shape
        self.rng = np.random.default_rng(seed)
        self.__in_dim = 5
        self.__out_dim = 4
        self.lhs = LHS(self.rng)
        self.log_fit = log_fit
        self.parallel = parallel

        
        CPP = 10 ** np.linspace(13, 20.35, 200)
        CP = CPP + (2 * (1.6 * 10 ** -41) * CPP ** 3) / (1 - (1.6 * 10 ** -41) * CPP ** 2)
        self.f = scipy.interpolate.interp1d(CP, CPP, kind="quadratic")
        self.dt = 1e-1  
        self.dx = 1e-6  
        return

    @property
    def in_dim(self) -> int:
        return self.__in_dim

    @property
    def out_dim(self) -> int:
        return self.__out_dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    
    def __d(self, t, c):
        kB = 8.6171E-5  
        Di_P = 5.8 * 10 ** -5 * np.exp(-2.1 / (kB * t))
        Di_I = 2.3 * 10 ** -1 * np.exp(-2.6 / (kB * t))
        Di_PV = 7.6 * 10 ** 4 * np.exp(-5.2 / (kB * t))

        k = 0.5
        m = 2
        l = -1.8

        ni = 1.8 * 10 ** 21 * np.exp(-0.66 / (kB * t))
        n = (1 / 2) * (c + np.sqrt(c ** 2 + 4 * ni ** 2))

        h = 1.5

        Deff_Pi = Di_P * (n / ni) ** k * h
        Deff_I = Di_I * (n / ni) ** l * h
        Deff_PV = Di_PV * (self.f(c) / ni) ** m * h

        Deff_P = (Deff_Pi * Deff_I) / (Deff_Pi + Deff_I) + Deff_PV

        Csat = 4.1 * 10 ** 22 * np.exp(-0.44 / (kB * t)) * np.ones_like(c)
        Cmod = np.min(np.hstack((c.reshape(-1, 1), Csat.reshape(-1, 1))), axis=1)

        Deff = Deff_P * (Cmod / c)

        return Deff

    
    @staticmethod
    def __gauss(x, a, b, c, d, c_init):
        para = np.array([a, b, c, d])
        C_c = para[0] * np.exp(-np.square(x) / np.square(para[1]))
        C_c += para[2] * np.exp(-np.square(x) / np.square(para[3]))
        return np.max(np.hstack((C_c.reshape(-1, 1), c_init.reshape(-1, 1))), axis=1)

    @staticmethod
    def __erfc(x, a, b, c, d, c_init):
        para = np.array([a, b, c, d])
        C_c = para[0] * scipy.special.erfc(x / para[1]) + para[2] * scipy.special.erfc(x / para[3])
        return np.max(np.hstack((C_c.reshape(-1, 1), c_init.reshape(-1, 1))), axis=1)

    def eval_f(self, x):
        
        
        x = x.reshape(5, )
        Temp1 = x[0] + 273
        Time1 = x[1]
        Temp2 = x[2] + 273
        Time2 = x[3]
        C0 = np.power(10., x[4])  

        
        total_time1 = int(Time1 / self.dt)  
        total_time2 = int(Time2 / self.dt)  

        total_x = 1.0e-3  
        total_xnum = int(total_x / self.dx)  
        C_init = np.ones(total_xnum) * 1E14  

        
        
        C = np.copy(C_init)
        C_ = np.copy(C_init)

        
        for i in range(total_time1):
            C[0] = C0
            C_[1:-1] = C[1:-1] + self.__d(Temp1, C[1:-1]) * (C[2:] - 2 * C[1:-1] + C[:-2]) / self.dx ** 2 * self.dt
            C_ = np.max(np.hstack((C_.reshape(-1, 1), C_init.reshape(-1, 1))), axis=1)
            C_[0] = C0
            C = np.copy(C_)

        for i in range(total_time2):
            C[0] = C0
            C_[1:-1] = C[1:-1] + self.__d(Temp2, C[1:-1]) * (C[2:] - 2 * C[1:-1] + C[:-2]) / self.dx ** 2 * self.dt
            C_ = np.max(np.hstack((C_.reshape(-1, 1), C_init.reshape(-1, 1))), axis=1)
            C_[0] = C0
            C = np.copy(C_)

        x_plt = np.linspace(0, total_x, total_xnum) * 1E4  
        
        bounds = [[0.9 * C0, 0., 1e14, 0.001], [1.1 * C0, 5., C0 / 10, 5.]]  

        func_dict = {'gaussian': self.__gauss, 'erfc': self.__erfc}
        func = func_dict[self.shape]

        def shape_wrapper(_x, a, b, c, d):
            return func(_x, a=a, b=b, c=c, d=d, c_init=C_init)

        def log_shape_wrapper(_x, a, b, c, d):
            return np.log10(func(_x, a=a, b=b, c=c, d=d, c_init=C_init))

        
        lhs_bounds = [(bounds[0][i], bounds[1][i]) for i in range(len(bounds[0]))]
        para_set = self.lhs.generate(bounds=lhs_bounds, n_sample=100)
        para_set = np.vstack([para_set, np.array([C0, 0.05, 1E18, 0.2])])

        def fit(_para, _log_fit: bool = True):
            try:
                if _log_fit:
                    _popt, _ = scipy.optimize.curve_fit(log_shape_wrapper, x_plt, np.log10(C), bounds=bounds,
                                                        p0=_para)  
                    _loss = np.sum(np.square(np.log10(C) - log_shape_wrapper(x_plt, *_popt)))
                else:
                    _popt, _ = scipy.optimize.curve_fit(shape_wrapper, x_plt, C, bounds=bounds, p0=_para)  
                    _loss = np.sum(np.square(C - shape_wrapper(x_plt, *_popt)))
                return _loss, _popt
            except RuntimeError:
                return None

        p = joblib.Parallel(n_jobs=self.parallel)
        with joblib.parallel_backend('loky', inner_max_num_threads=1):
            results = p(joblib.delayed(fit)(para) for para in para_set)

        best_para = None
        min_loss = None

        for result in results:
            if result is None:
                continue
            loss = result[0]
            popt = result[1]
            if best_para is None or min_loss > loss:
                best_para = popt
                min_loss = loss

        if best_para is None:
            raise RuntimeError('curve_fit is failed for all initial points.')
        popt = np.append(C0, best_para)
        
        
        
        
        
        fit_param = popt.copy()
        fit_param[0] = np.log10(fit_param[0])
        fit_param[2] = np.log10(fit_param[2])
        result = fit_param.reshape(1, 4)
        return result, {'profile': C}


class PC1DEmu(RealFunction):
  

    def __init__(self, bounds: list[tuple[Any, Any]] = None):
        self.__default_bounds = [(np.log10(0.9 * 1e19), np.log10(1.1 * 1.5e21)),
                                 (np.log10(0.003), np.log10(2.5)),  
                                 (17.96198364, 20.17609126),
                                 (np.log10(0.003), np.log10(2.5)),  
                                 (50., 250.),
                                 (14., 17.)]
        if bounds is None:
            bounds = self.__default_bounds

        super().__init__(bounds=bounds)
        self.__in_dim = 4 + 2
        self.__out_dim = 1

        from benchmark.emulator import GPRegression
        self.__emulator_dir = pathlib.Path(__file__).parent.parent.parent / 'emulator/64bit'
        train_set = torch.load(self.__emulator_dir / 'train_set.pkl')
        train_x = train_set['x']
        train_y = train_set['y']
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = 1e-4
        likelihood.noise_covar.raw_noise.requires_grad = False
        self._emulator = GPRegression(train_x, train_y, likelihood)
        state_dict = torch.load(self.__emulator_dir / 'emulator.pth')
        self._emulator.load_state_dict(state_dict)
        self._emulator.eval()
        self._emulator.likelihood.eval()
        return

    @property
    def in_dim(self) -> int:
        return self.__in_dim

    @property
    def out_dim(self) -> int:
        return self.__out_dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    def eval_f(self, x: np.ndarray):
       
        assert x.ndim == 2
        x = x.copy().reshape(-1, 6)
        x[:, 1] = np.power(10., x[:, 1])
        x[:, 3] = np.power(10., x[:, 3])

        x = torch.tensor(x.copy(), dtype=torch.float64)

        with torch.no_grad():
            pred = self._emulator.predict(x)
            res = pred.mean.clone().numpy()
        return res.reshape(-1, 1)



class EC_1st(RealFunction):
  
    def __init__(self, bounds: list[tuple[Any, Any]] = None):
        self.__default_bounds = [(50., 300.), 
                                 (0.25, 4.), 
                                 (100., 1000.), 
                                 (10., 100.), 
                                 (270., 420.), 
                                 (10., 40.), 
                                 (15., 60.)] 
        if bounds is None:
            bounds = self.__default_bounds

        super().__init__(bounds=bounds)
        self.__in_dim = 7
        self.__out_dim = 2

        self.__emulator_dir = pathlib.Path(__file__).parent.parent.parent / 'emulator/ECdata'
        self.basis_dim = 1000
        
        with open(self.__emulator_dir / 'J0_RFF_weights.pickle', 'rb') as f:
            self.J0_RFF_weights = pickle.load(f)
        with open(self.__emulator_dir / 'R1_RFF_weights.pickle', 'rb') as f:
            self.R1_RFF_weights = pickle.load(f)
        
        with open(self.__emulator_dir / 'J0_RFF_offset.pickle', 'rb') as f:
            self.J0_RFF_offset = pickle.load(f)
        with open(self.__emulator_dir / 'R1_RFF_offset.pickle', 'rb') as f:
            self.R1_RFF_offset = pickle.load(f)
        
        with open(self.__emulator_dir / 'J0_BLM_weights.pickle', 'rb') as f:
            self.J0_BLM_weights = pickle.load(f)
        with open(self.__emulator_dir / 'R1_BLM_weights.pickle', 'rb') as f:
            self.R1_BLM_weights = pickle.load(f)
        
        with open(self.__emulator_dir / 'J0_BLM_offset.pickle', 'rb') as f:
            self.J0_BLM_offset = pickle.load(f)
        with open(self.__emulator_dir / 'R1_BLM_offset.pickle', 'rb') as f:
            self.R1_BLM_offset = pickle.load(f)
        
        with open(self.__emulator_dir / 'J0_BLM_std.pickle', 'rb') as f:
            self.J0_BLM_std = pickle.load(f)
        with open(self.__emulator_dir / 'R1_BLM_std.pickle', 'rb') as f:
            self.R1_BLM_std = pickle.load(f)
        return

    @property
    def in_dim(self) -> int:
        return self.__in_dim

    @property
    def out_dim(self) -> int:
        return self.__out_dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    def eval_f(self, x: np.ndarray):
      
        x = np.atleast_2d(x)

        J0_sample_path = self.J0_BLM_std * np.sqrt(2 / self.basis_dim) * np.cos(x.dot(self.J0_RFF_weights.T) + self.J0_RFF_offset).dot(self.J0_BLM_weights) + self.J0_BLM_offset

        R1_sample_path = self.R1_BLM_std * np.sqrt(2 / self.basis_dim) * np.cos(x.dot(self.R1_RFF_weights.T) + self.R1_RFF_offset).dot(self.R1_BLM_weights) + self.R1_BLM_offset

        res = np.c_[np.c_[J0_sample_path], np.c_[R1_sample_path]]
        return res


class EC_2nd(RealFunction):
   

    def __init__(self, bounds: list[tuple[Any, Any]] = None):
        self.__default_bounds = [(0.01, 1.), 
                                 (0.01, 0.1),
                                 (0.01, 0.1)] 
        if bounds is None:
            bounds = self.__default_bounds

        super().__init__(bounds=bounds)
        self.__in_dim = 3
        self.__out_dim = 1

        self.__emulator_dir = pathlib.Path(__file__).parent.parent.parent / 'emulator/ECdata'
        self.basis_dim = 1000
        
        with open(self.__emulator_dir / 'Eff_RFF_weights.pickle', 'rb') as f:
            self.Eff_RFF_weights = pickle.load(f)
        
        with open(self.__emulator_dir / 'Eff_RFF_offset.pickle', 'rb') as f:
            self.Eff_RFF_offset = pickle.load(f)
        
        with open(self.__emulator_dir / 'Eff_BLM_weights.pickle', 'rb') as f:
            self.Eff_BLM_weights = pickle.load(f)
        
        with open(self.__emulator_dir / 'Eff_BLM_offset.pickle', 'rb') as f:
            self.Eff_BLM_offset = pickle.load(f)
        
        with open(self.__emulator_dir / 'Eff_BLM_std.pickle', 'rb') as f:
            self.Eff_BLM_std = pickle.load(f)
        return

    @property
    def in_dim(self) -> int:
        return self.__in_dim

    @property
    def out_dim(self) -> int:
        return self.__out_dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    def eval_f(self, x: np.ndarray):
       
        x = np.atleast_2d(x)

        J_L = 38. * (1 - x[:,2])
        R_s = x[:,1] / x[:,2]
        x = np.c_[x[:,:2], np.c_[J_L], np.c_[R_s]]

        Eff_sample_path = self.Eff_BLM_std * np.sqrt(2 / self.basis_dim) * np.cos(x.dot(self.Eff_RFF_weights.T) + self.Eff_RFF_offset).dot(self.Eff_BLM_weights) + self.Eff_BLM_offset
        res = np.c_[Eff_sample_path]
        return res



@dataclass
class Bookmark:
    line: int  
    prefix: str  







class PC1D(RealFunction):
   

    def __init__(self, bounds: list[tuple[Any, Any]] = None, shape: str = 'gaussian'):
        self.__default_bounds = [(19., np.log10(1.5e21)),
                                 (0.00685163, 1.59361012),
                                 (17.96198364, 20.17609126),
                                 (0.01485839, 1.96602677),
                                 (50., 250.),
                                 (14., 17.)]
        if bounds is None:
            bounds = self.__default_bounds
        shape_dict = {'uniform': 0, 'exponential': 1, 'gaussian': 2, 'erfc': 3}
        assert shape in shape_dict.keys()
        super().__init__(bounds=bounds)
        self.shape = shape_dict[shape]
        self.__in_dim = 4 + 2
        self.__out_dim = 1
        self.__pc1d_dir = pathlib.Path(__file__).parent.parent.parent / 'pc1d'
        self.__pc1d_exe = 'cmdPC1D.exe'
        
        self.__txt2bin = 'ascii2prm.exe'
        base_setting = 'base.txt'
        self.__setting_txt = 'python_setting_{}.txt'
        self.__setting_bin = 'python_setting_{}.prm'
        self.__dop_file = 'python_dop_{}.dop'
        self.__out_file = 'python_out_{}.txt'

        self.__cmd_exec = ft.partial(subprocess.run, cwd=str(self.__pc1d_dir), encoding='utf-8',
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self.__b_shape1 = Bookmark(line=166 - 1, prefix='CDiffusion::m_Profile=')
        self.__b_peak1 = Bookmark(line=167 - 1, prefix='CDiffusion::m_Npeak=')
        self.__b_depth1 = Bookmark(line=168 - 1, prefix='CDiffusion::m_Depth=')

        
        self.__b_shape2 = Bookmark(line=174 - 1, prefix='CDiffusion::m_Profile=')
        self.__b_peak2 = Bookmark(line=175 - 1, prefix='CDiffusion::m_Npeak=')
        self.__b_depth2 = Bookmark(line=176 - 1, prefix='CDiffusion::m_Depth=')

        self.__b_thickness = Bookmark(line=194 - 1, prefix='CRegion::m_Thickness=')
        self.__b_doping = Bookmark(line=199 - 1, prefix='CRegion::m_BkgndDop=')
        self.__b_ext_file = Bookmark(line=195 - 1, prefix='CRegion::m_FrontFilename=')
        self.__b_ext_flag = Bookmark(line=196 - 1, prefix='CRegion::m_FrontExternal=')

        bookmarks = [self.__b_peak1, self.__b_depth1, self.__b_peak2, self.__b_depth2,
                     self.__b_thickness, self.__b_doping]

        with open(self.__pc1d_dir / base_setting, 'r') as reader:
            self.__default_setting = reader.readlines()
        for item in bookmarks:
            assert self.__default_setting[item.line].startswith(item.prefix)
        return

    @property
    def in_dim(self) -> int:
        return self.__in_dim

    @property
    def out_dim(self) -> int:
        return self.__out_dim

    @property
    def default_bounds(self) -> list[tuple[Any, Any]]:
        return self.__default_bounds

    def eval_f(self, x, use_profile: bool = False, return_step: bool = False):
        if not use_profile:
            assert x.ndim == 2

        x = x.reshape(6, )
        peak1 = np.power(10., x[0])  
        depth1 = x[1] / 10000.  
        peak2 = np.power(10., x[2])  
        depth2 = x[3] / 10000.  
        thickness = x[4] / 10000.  
        doping = np.power(10., x[5])  
        setting = self.__default_setting.copy()
        setting[self.__b_shape1.line] = self.__b_shape1.prefix + str(self.shape) + '\n'
        setting[self.__b_peak1.line] = self.__b_peak1.prefix + str(peak1) + '\n'
        setting[self.__b_depth1.line] = self.__b_depth1.prefix + str(depth1) + '\n'

        setting[self.__b_shape2.line] = self.__b_shape2.prefix + str(self.shape) + '\n'
        setting[self.__b_peak2.line] = self.__b_peak2.prefix + str(peak2) + '\n'
        setting[self.__b_depth2.line] = self.__b_depth2.prefix + str(depth2) + '\n'
        setting[self.__b_thickness.line] = self.__b_thickness.prefix + str(thickness) + '\n'
        setting[self.__b_doping.line] = self.__b_doping.prefix + str(doping) + '\n'

        def _create_files(func_id):
            _setting_txt = self.__setting_txt.format(func_id)
            _setting_bin = self.__setting_bin.format(func_id)
            _out_file = self.__out_file.format(func_id)

            with open(self.__pc1d_dir / _setting_txt, 'w') as writer:
                writer.writelines(setting)
            max_retries = 3
            for i in range(1, max_retries + 1):
                try:
                    self.__cmd_exec(['wine', self.__txt2bin, _setting_txt, _setting_bin], timeout=3)
                except subprocess.TimeoutExpired as error:
                    print(f'timeout:{i}')
                    if i != max_retries:
                        continue
                    print('stdout')
                    print(error.stdout)
                    print('stderr')
                    print(error.stderr)
                    raise error
                else:
                    break
            try:
                self.__cmd_exec(['wine', self.__pc1d_exe, '-i', _setting_bin, '-o', _out_file], timeout=70)
            except subprocess.TimeoutExpired as error:
                print('stdout')
                print(error.stdout)
                print('stderr')
                print(error.stderr)
                raise error

            _res = pd.read_csv(self.__pc1d_dir / _out_file, delimiter='\t', header=0)
            _p_max = -_res['Base Power'].to_numpy().min() * 1000

            if np.isnan(_p_max):
                print(_out_file)
                print(_res)
                import sys
                sys.exit(11)
            (self.__pc1d_dir / _out_file).unlink(missing_ok=True)
            (self.__pc1d_dir / _setting_txt).unlink(missing_ok=True)
            (self.__pc1d_dir / _setting_bin).unlink(missing_ok=True)
            return _res, _p_max

        function_id = id(_create_files)
        res, p_max = _create_files(function_id)

        if return_step:
            n_step = len(res)
            return p_max.reshape(1, 1), n_step
        return p_max.reshape(1, 1)

    @classmethod
    def to_cui_param(cls, x):
    
        if x.ndim == 1:
            x = x.reshape(6, 1)
        x[:, 0] = np.power(10., x[:, 0])
        x[:, 1] = x[:, 1] / 10000
        x[:, 2] = np.power(10., x[:, 2])
        x[:, 3] = x[:, 3] / 10000
        x[:, 4] = x[:, 4] / 10000
        x[:, 5] = np.power(10., x[:, 5])
        return x

    @classmethod
    def to_gui_param(cls, x):
       
        if x.ndim == 1:
            x = x.reshape(6, 1)
        x[:, 0] = np.power(10., x[:, 0])
        x[:, 2] = np.power(10., x[:, 2])
        x[:, 5] = np.power(10., x[:, 5])
        return x
