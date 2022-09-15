import functools as ft
import multiprocessing as mp
import time
import typing
from collections.abc import Callable

import numpy as np
import scipy.optimize as opt
import torch.multiprocessing as torch_mp
import tqdm
from joblib import parallel_backend, Parallel, delayed

from util import LHS


class _FuncWrapper:
    def __init__(self, func: Callable, args: tuple = (), negate: bool = False, with_jac: bool = True):
        self.func = func
        self.args = args
        self.negate = negate
        self.with_jac = with_jac

    def eval(self, x):
        val = self.func(x, *self.args)
        if not self.negate:
            return val

        if self.with_jac:
            
            return -val[0], -val[1]
        else:
            return -val


class LBFGS:
    def __init__(self, ftol: float = 2.220446049250313e-09, gtol: float = 1e-5, maxfun: int = 15000,
                 print_detail: str = None):
        
        self.ftol = ftol
        self.gtol = gtol
        self.max_fun = maxfun
        self.print_detail = print_detail
        return

    def run(self, x0, func: Callable, bounds: list[tuple], args: tuple = (),
            jac: typing.Union[bool, Callable] = True, minimize: bool = True, disp=-1):
        if callable(jac):
            wrap_f = _FuncWrapper(func, args=args, negate=not minimize)
            wrap_jac = _FuncWrapper(jac, args=args, negate=not minimize)
        else:
            wrap_f = _FuncWrapper(func, args=args, negate=not minimize, with_jac=jac)
            wrap_jac = jac

        options = {'maxfun': self.max_fun, 'ftol': self.ftol, 'gtol': self.gtol, 'disp': disp}
        result: opt.OptimizeResult = opt.minimize(wrap_f.eval, x0, method='L-BFGS-B',
                                                  jac=wrap_jac, bounds=bounds, options=options)

        if self.print_detail == 'always':
            print(f'success: {result.success}')
            print(f'message: {result.message}\n')
        elif self.print_detail == 'error' and not result.success:
            print(f'success: {result.success}')
            print(f'message: {result.message}\n')
        if not minimize:
            result['fun'] *= -1
            result['jac'] *= -1
        return result


class MultistartResults(typing.NamedTuple):
    x: np.ndarray  
    fun: np.ndarray  
    rough: list[opt.OptimizeResult]
    strict: list[opt.OptimizeResult]
    time_init_eval: float
    time_rough: float
    time_strict: float
    total_nfev: int
    n_failed: int


class MultistartLBFGS:
    def __init__(self, rough_optimizer: LBFGS = None, strict_optimizer: LBFGS = None, n_init_sample: int = 1000,
                 n_parallel: int = 1, seed: int = None,
                 print_time: bool = False, print_detail: typing.Optional[str] = None):

        if rough_optimizer is None:
            rough_optimizer = LBFGS(ftol=1e-3, gtol=1e-3, print_detail=print_detail)
        if strict_optimizer is None:
            strict_optimizer = LBFGS(print_detail=print_detail)

        self.rough_optimizer = rough_optimizer
        self.strict_optimizer = strict_optimizer

        self.n_parallel = n_parallel
        self.n_init_sample = n_init_sample
        
        self.sampler = LHS(seed)
        self.print_time = print_time
        self.print_detail = print_detail
        return

    @staticmethod
    def block_job(x0_set, optimizer: LBFGS, func, bounds, args: tuple = (),
                  jac: typing.Union[bool, Callable] = True,
                  minimize: bool = True, disp: int = -1):
        return [optimizer.run(func=func, bounds=bounds, args=args, x0=x0,
                              jac=jac, minimize=minimize, disp=disp) for x0 in x0_set]

    def run(self, func: Callable, bounds: list[tuple], f_to_init_eval: Callable,
            args: tuple = (),
            jac: typing.Union[bool, Callable] = True,
            additional_init_points: np.ndarray = None, top_n: int = 5, stop_val: float = None,
            skip_fzero_point: bool = False, minimize: bool = True, disp: int = -1, *, backend: str = 'joblib'):
        assert backend in {'joblib', 'torch', 'multiprocessing'}
        if stop_val is None:
            stop_val = -np.inf if minimize else np.infty

        wrap_init_f = _FuncWrapper(f_to_init_eval, args=args, negate=not minimize, with_jac=False)

        init_x = np.array(self.sampler.generate(bounds, self.n_init_sample))

        begin_init = time.time()
        init_x_split = np.array_split(init_x, self.n_parallel)

        if backend == 'joblib' or self.n_parallel == 1:
            pool = Parallel(n_jobs=self.n_parallel)
            with parallel_backend('loky', inner_max_num_threads=1):
                init_f = pool(delayed(wrap_init_f.eval)(x_sp) for x_sp in init_x_split)
        elif backend == 'torch':
            ctx = torch_mp.get_context('spawn')
            with ctx.Pool(self.n_parallel) as pool:
                init_f = pool.map(wrap_init_f.eval, init_x_split)
            pool.close()
            pool.terminate()
        else:
            ctx = mp.get_context('spawn')
            with ctx.Pool(self.n_parallel) as pool:
                init_f = pool.map(wrap_init_f.eval, init_x_split)
            pool.close()
            pool.terminate()

        init_f = np.concatenate(init_f)
        end_init = time.time()
        time_init_eval = end_init - begin_init

        if self.print_time:
            print(f'init_eval :  {end_init - begin_init} sec')
        best_index = np.argmin(init_f) if minimize else np.argmax(init_f)
        best_value = init_f[best_index]
        if (minimize and best_value < stop_val) or (not minimize and best_value > stop_val):
            result = MultistartResults(x=init_x[best_index], fun=best_value, rough=[], strict=[],
                                       time_init_eval=time_init_eval, time_rough=0., time_strict=0.,
                                       total_nfev=0, n_failed=0)
            return result

        if skip_fzero_point:
            nonzero_init_x = init_x[init_f != 0.]
            if self.print_detail == 'always':
                print(f'number of f(x)=0: {init_x.shape[0] - nonzero_init_x.shape[0]}')
            if nonzero_init_x.size == 0:
                nonzero_init_x = init_x
            init_x = nonzero_init_x

        begin_rough = time.time()
        if self.n_parallel == 1 or init_x.shape[0] < 20:
            if self.print_detail == 'always':
                init_x = tqdm.tqdm(init_x)
            rough_results = [self.rough_optimizer.run(func=func, bounds=bounds, args=args, x0=xi, jac=jac,
                                                      minimize=minimize, disp=disp) for xi in init_x]
        else:
            x_split = np.array_split(init_x, min(init_x.shape[0], self.n_parallel * 10))
            if backend == 'joblib':
                with parallel_backend('loky', inner_max_num_threads=1):
                    parallel_results = pool(delayed(self.block_job)(optimizer=self.rough_optimizer, x0_set=x_sp,
                                                                    func=func, bounds=bounds, args=args, jac=jac,
                                                                    minimize=minimize, disp=disp)
                                            for x_sp in x_split)
            elif backend == 'torch':
                ctx = torch_mp.get_context('spawn')
                with ctx.Pool(self.n_parallel) as pool:
                    parallel_results = pool.map(ft.partial(self.block_job, optimizer=self.rough_optimizer,
                                                           func=func, bounds=bounds, args=args, jac=jac,
                                                           minimize=minimize, disp=disp),
                                                x_split)
                    pool.close()
                    pool.terminate()
            else:
                ctx = mp.get_context('spawn')
                with ctx.Pool(self.n_parallel) as pool:
                    parallel_results = pool.map(ft.partial(self.block_job, optimizer=self.rough_optimizer,
                                                           func=func, bounds=bounds, args=args, jac=jac,
                                                           minimize=minimize, disp=disp),
                                                x_split)
                    pool.close()
                    pool.terminate()
            rough_results = []
            for item in parallel_results:
                rough_results.extend(item)
        end_rough = time.time()
        time_rough = end_rough - begin_rough
        total_nfev = init_x.shape[0]

        n_fail = 0
        for res in rough_results:
            total_nfev += res.nfev
            if not res.success:
                n_fail += 1
                if self.print_detail in {'always', 'error'}:
                    print(res)
        if self.print_time:
            print(f'(rough) L-BFGS-B : times={len(rough_results)} / total= {time_rough} sec, {n_fail=}')

        rough_f = np.array([item['fun'] for item in rough_results]).flatten()
        best_index = np.argmin(rough_f) if minimize else np.argmax(rough_f)
        best_value = rough_f[best_index]
        if (minimize and best_value < stop_val) or (not minimize and best_value > stop_val):
            result = MultistartResults(x=rough_results[best_index]['x'], fun=best_value, rough=rough_results, strict=[],
                                       time_init_eval=time_init_eval, time_rough=time_rough, time_strict=0.,
                                       total_nfev=total_nfev, n_failed=n_fail)
            return result

        if minimize:
            argsort_fval = np.argsort(rough_f)
        else:
            argsort_fval = np.argsort(-rough_f)

        strict_index = argsort_fval[:top_n]
        strict_x0 = np.array([rough_results[i]['x'] for i in strict_index])

        if additional_init_points is not None:
            strict_x0 = np.vstack([strict_x0, additional_init_points])

        begin_strict = time.time()
        if backend == 'joblib' or self.n_parallel == 1:
            with parallel_backend('loky', inner_max_num_threads=1):
                strict_results = pool(delayed(self.strict_optimizer.run)(func=func, bounds=bounds, args=args, x0=xi,
                                                                         jac=jac, minimize=minimize, disp=disp)
                                      for xi in strict_x0)
        elif backend == 'torch':
            ctx = torch_mp.get_context('spawn')
            with ctx.Pool(self.n_parallel) as pool:
                strict_results = pool.map(ft.partial(self.strict_optimizer.run, func=func, bounds=bounds, args=args,
                                                     jac=jac, minimize=minimize, disp=disp),
                                          strict_x0)
                pool.close()
                pool.terminate()
        else:
            ctx = mp.get_context('spawn')
            with ctx.Pool(self.n_parallel) as pool:
                strict_results = pool.map(ft.partial(self.strict_optimizer.run, func=func, bounds=bounds, args=args,
                                                     jac=jac, minimize=minimize, disp=disp),
                                          strict_x0)
                pool.close()
                pool.terminate()
        end_strict = time.time()
        time_strict = end_strict - begin_strict

        for res in strict_results:
            total_nfev += res.nfev
            if not res.success:
                n_fail += 1
                if self.print_detail in {'always', 'error'}:
                    print(res)

        if self.print_time:
            print(f'(strict) L-BFGS-B : {time_strict} sec')

        if minimize:
            best_index = np.argmin([item['fun'] for item in strict_results])
        else:
            best_index = np.argmax([item['fun'] for item in strict_results])

        best_x = strict_results[best_index]['x']
        best_fun = strict_results[best_index]['fun']
        result = MultistartResults(x=best_x, fun=best_fun, rough=rough_results, strict=strict_results,
                                   time_init_eval=time_init_eval, time_rough=time_rough, time_strict=time_strict,
                                   total_nfev=total_nfev, n_failed=n_fail)
        return result
