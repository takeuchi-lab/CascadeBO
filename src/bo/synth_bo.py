import bisect
import dataclasses as dc
import datetime
import functools as ft
import time
import typing
from collections.abc import Callable

import gpytorch.mlls
import memory_profiler
import numpy as np
import torch

import benchmark as bm
from acquisition import CascadeEI, CBO, EI, CascadeUCB, UCB, CascadeUncertainty, EIFunctionNetwork
from models import CascadeMOGP, SingleTaskGP, fit_param
from optimize import MultistartLBFGS, LBFGS
from util import make_slices, to_float_tensor, FunctionValue


@dc.dataclass
class OutputStock:
    out_stage: int  
    y_value: np.ndarray
    obtained_iteration: int
    af_value: np.ndarray = None
    opt_x: np.ndarray = None
    need_update: bool = False
    remain_times: int = 1
    suggested_ctrl_param: np.ndarray = None

    target_lower: FunctionValue = dc.field(default_factory=lambda: FunctionValue(None, None, require_update=True))
    target_upper: FunctionValue = dc.field(default_factory=lambda: FunctionValue(None, None, require_update=True))


@dc.dataclass
class DecisionResult:
    next_stg: int
    next_input: np.ndarray
    af: typing.Union[float, list[float], np.ndarray, None]
    stock_index: typing.Optional[int]
    remain_param: np.ndarray = None


@dc.dataclass
class SynthCascade:
    bench: list[bm.SynthFunction]
    ctrl_dim: list[int]
    each_cost: list[int]
    ctrl_domain: list[list[tuple[typing.Any, typing.Any]]]
    out_dim: typing.Union[int, list[int]] = 1

    def __post_init__(self):
        self.n_stage: int = len(self.ctrl_dim)
        self.all_d: int = sum(self.ctrl_dim)

        if isinstance(self.out_dim, int):
            self.out_dim = [self.out_dim for _ in range(self.n_stage)]

        self.input_dim: list[int] = self.ctrl_dim.copy()
        for i in range(1, self.n_stage):
            
            self.input_dim[i] += self.out_dim[i - 1]
        self.slices: list[slice] = make_slices(self.ctrl_dim)
        self.remain_cost: list[int] = np.cumsum(self.each_cost[::-1])[::-1].tolist()
        self.min: float = self.bench[-1].min
        self.max: float = self.bench[-1].max
        return

    def ctrl_domain_to_end(self, from_i: int):
        res = []
        for xd in self.ctrl_domain[from_i:]:
            res += xd
        return res

    def to_ctrl_split(self, x: np.ndarray) -> list[np.ndarray]:
        assert x.ndim == 2, f'x.shape={x.shape}'
        return [x[:, s] for s in self.slices]

    def observe_to_end(self, x, minus=False):
        if x.ndim == 1:
            x = x.reshape(1, self.all_d)
        x_sp = self.to_ctrl_split(x)
        z = self.bench[0].eval_f(x_sp[0])
        for i in range(1, self.n_stage):
            con_x = np.hstack([z, x_sp[i]])
            z = self.bench[i].eval_f(con_x)
        if minus:
            z *= -1
        return z


DecisionFunc = Callable[..., DecisionResult]
DiscardFunc = Callable[typing.Optional[OutputStock], int]
CBORange = typing.Literal['true', 'wide', 'narrow']


class MultioutCascadeBO:
    def __init__(self, cascade: SynthCascade, model: CascadeMOGP, method: str, seed: int, f_best: float,
                 n_thread: int = 1, mc_samples: int = 1000, stock_lifetime: int = 1,
                 noise_scale=1e-2, maximize: bool = True, eval_inference_r: bool = False,
                 discard_strategy: str = 'save_all', max_stocks: int = 10,
                 ucb_root_beta=2., lipschitz=1,
                 discard_stocks_interval: int = 1,
                 ):
        decision_funcs: dict[str, DecisionFunc] = {'ei': self._ei_decision,
                                                   'ei_seq': self._ei_seq_decision,
                                                   'ei_nonadaptive': ft.partial(self._ei_non_adaptive_decision,
                                                                                full_sample=False),
                                                   'eifn': ft.partial(self._ei_non_adaptive_decision, full_sample=True),
                                                   'ucb_seq': self._ucb_seq_decision,
                                                   'cb_upd': self._ucb_seq_upd_decision,
                                                   'cbo': ft.partial(self._cbo_decision, y_range='true'),
                                                   'cbo_wide': ft.partial(self._cbo_decision, y_range='wide'),
                                                   'cbo_narrow': ft.partial(self._cbo_decision, y_range='narrow'),
                                                   'random': self._random_decision,
                                                   'joint': ft.partial(self._joint_decision, af='EI'),
                                                   'jointUCB': ft.partial(self._joint_decision, af='UCB')}

        valid_discard_strategy: dict[str, DiscardFunc] = {'save_all': self._no_discard,
                                                          'discard_all': self._discard_all,
                                                          'fifo': self._discard_stocks_by_fifo,
                                                          'af': self._discard_stocks_by_af,
                                                          'ci': self._discard_stocks_by_ci}
        assert method in decision_funcs.keys()
        assert discard_strategy in valid_discard_strategy.keys()
        
        self.cascade = cascade
        self.model = model
        self.method = method
        self.seed = seed
        self.f_best: float = f_best
        self.mc_samples = mc_samples
        self.stock_lifetime = stock_lifetime
        self.noise_scale = noise_scale
        self.maximize = maximize
        self.n_thread = n_thread
        self.eval_inference_r = eval_inference_r
        self._discard_strategy: DiscardFunc = valid_discard_strategy[discard_strategy]
        self.max_stocks = max_stocks
        self.root_beta = ucb_root_beta
        self.lipschitz = lipschitz
        self.discard_stock_interval = discard_stocks_interval
        if discard_strategy != 'save_all':
            assert self.is_suspendable()

        
        self.f_opt = self.cascade.max if self.maximize else self.cascade.min
        self.af_optimizer = MultistartLBFGS(rough_optimizer=LBFGS(ftol=1e-3, gtol=1e-3), strict_optimizer=LBFGS(),
                                            n_parallel=n_thread, seed=self.seed)

        if maximize:
            y_best = self.model.all_train_targets[-1].max()
        else:
            y_best = self.model.all_train_targets[-1].min()
        self.y_best: float = float(y_best)
        self.x_domain = cascade.ctrl_domain
        self.rng = np.random.default_rng(seed)
        self.iteration: int = 0
        self.prev_stage = None
        self.sorted_stocks: list[OutputStock] = []
        self._get_next: DecisionFunc = decision_funcs[self.method]
        self.base_samples: list[torch.Tensor] = []

        for i in range(self.n_stage):
            bs = self.rng.standard_normal(size=(self.mc_samples, self.model.out_dims[i]))
            self.base_samples.append(to_float_tensor(bs))

        
        self.decided_param = None

        
        self.misc_rng = np.random.default_rng(seed)
        self.misc_optimizer = MultistartLBFGS(rough_optimizer=LBFGS(ftol=1e-3, gtol=1e-3), strict_optimizer=LBFGS(),
                                              n_parallel=n_thread, seed=self.seed)
        self.suggested_x = None  
        self.target_lower: typing.Optional[float] = None

        
        self.cost_hist = [0]
        self.obs_best_hist = [float(self.y_best)]
        self.infer_best_hist = []
        s_regret = float(self.f_opt - self.f_best)
        if not self.maximize:
            s_regret *= -1
        self.s_regret_hist = [s_regret]
        self.i_regret_hist = []

        self.elapsed_time: list[datetime.timedelta] = []
        self.n_stock_hist: list[int] = [0]
        self.n_discarded: list[int] = [0]
        self.discard_time: list[datetime.timedelta] = []

        
        self.suspend_hist: list[bool] = []
        self.obs_stage_hist: list[int] = []

        
        if self.eval_inference_r:
            suggest_x, target_lower = self.infer_best()
            f_inferred = self.cascade.observe_to_end(suggest_x)
            self.infer_best_hist.append(float(f_inferred))
            i_regret = float(self.f_opt - f_inferred)
            if not self.maximize:
                i_regret *= -1
            self.i_regret_hist.append(i_regret)
            self.suggested_x = suggest_x
            self.target_lower = float(target_lower)
        else:
            self.infer_best_hist.append(None)
            self.i_regret_hist.append(None)
        return

    @property
    def n_stage(self):
        return self.cascade.n_stage

    @property
    def current_cost(self):
        return self.cost_hist[-1]

    def is_suspendable(self):
        return self.method in {'ei'}

    def cascade_lower(self, entire_ctrl_param: np.ndarray, with_grad: bool = False):
        lower_bound = CascadeUCB(self.model, root_beta=self.root_beta, lipschitz=self.lipschitz, lcb=self.maximize)
        if with_grad:
            x = to_float_tensor(entire_ctrl_param).reshape(1, -1)
            x.requires_grad = True
            lower = lower_bound.evaluate(obs_stage=0, x=x)
            grad = torch.autograd.grad(lower, x)[0]
            return lower.detach().clone().numpy().astype(np.float64), grad.detach().clone().numpy().astype(np.float64)
        else:
            x = to_float_tensor(entire_ctrl_param)
            with torch.no_grad():
                lower = lower_bound.evaluate(obs_stage=0, x=x)
            return lower.clone().numpy().astype(np.float64)

    def infer_best(self, additional_point=None):
        result = self.misc_optimizer.run(ft.partial(self.cascade_lower, with_grad=True),
                                         f_to_init_eval=ft.partial(self.cascade_lower, with_grad=False),
                                         bounds=self.cascade.ctrl_domain_to_end(0),
                                         minimize=not self.maximize, additional_init_points=additional_point)
        lower_val = result.fun
        suggest_x = result.x
        return suggest_x, lower_val

    def update_target_lower(self):
        if not self.eval_inference_r:
            x, target_lower = self.infer_best(additional_point=self.suggested_x)
            self.target_lower = float(target_lower)
            self.suggested_x = x
        for stock in self.sorted_stocks:
            if self.maximize:
                lower = stock.target_lower
            else:
                lower = stock.target_upper
            if lower.need_update or lower.value is None:
                result = self._ucb_optimize(obs_stage=stock.out_stage + 1, prev_y=stock.y_value, lcb=self.maximize,
                                            additional_point=lower.x)
                lower.x = result.x
                lower.value = result.fun
                lower.need_update = False
            if self.target_lower < lower.value:
                self.target_lower = float(lower.value)
        return

    def _check_seq_state(self) -> bool:
        is_valid = True
        is_valid &= len(self.sorted_stocks) in {0, 1}
        return is_valid

    def cascade_ei(self, remain_ctrl_param: np.ndarray, obs_stage: int, prev_y=None, with_grad: bool = False,
                   full_sample: bool = False):
        if prev_y is not None:
            prev_y = prev_y.flatten()

        if full_sample:
            CEI = EIFunctionNetwork(self.model, self.y_best, base_samples=self.base_samples, maximize=self.maximize)
        else:
            CEI = CascadeEI(self.model, self.y_best, base_samples=self.base_samples, maximize=self.maximize)

        if with_grad:
            x = to_float_tensor(remain_ctrl_param)
            x.requires_grad = True
            if prev_y is None:
                x_cat = x
            else:
                x_cat = torch.cat([to_float_tensor(prev_y), x])
            x_cat = x_cat.reshape(1, -1)
            ei = CEI.evaluate(obs_stage=obs_stage, x=x_cat)
            grad = torch.autograd.grad(ei, x)[0]
            return ei.detach().clone().numpy().astype(np.float64), grad.detach().clone().numpy().astype(np.float64)
        else:
            nx = remain_ctrl_param.shape[0]
            x = to_float_tensor(remain_ctrl_param)
            if prev_y is None:
                x_cat = x
            else:
                x_cat = torch.cat([torch.tile(to_float_tensor(prev_y), (nx, 1)), x], dim=1)
            with torch.no_grad():
                ei = CEI.evaluate(obs_stage=obs_stage, x=x_cat)
            return ei.clone().numpy().astype(np.float64)

    def _ei_optimize(self, obs_stage: int, prev_y=None, additional_point=None, full_sample: bool = False):
        ei_func = ft.partial(self.cascade_ei, obs_stage=obs_stage, prev_y=prev_y, with_grad=True,
                             full_sample=full_sample)
        init_eval_f = ft.partial(self.cascade_ei, obs_stage=obs_stage, prev_y=prev_y, with_grad=False,
                                 full_sample=full_sample)
        result = self.af_optimizer.run(ei_func,
                                       bounds=self.cascade.ctrl_domain_to_end(obs_stage), minimize=not self.maximize,
                                       additional_init_points=additional_point, jac=True, skip_fzero_point=True,
                                       f_to_init_eval=init_eval_f
                                       )
        return result

    def _ei_decision(self) -> DecisionResult:
        next_stg = 0
        ei_max = -1
        best_input = None
        best_ctrl_batch = None

        
        stg = 0
        stock_i = None
        result = self._ei_optimize(stg, prev_y=None)
        ei_val = result.fun
        norm_ei = ei_val / self.cascade.remain_cost[stg]
        batch_x = result.x

        selected_point = batch_x[:self.cascade.ctrl_dim[next_stg]].reshape(1, -1)
        with torch.no_grad():
            prediction = self.model.predict(stg, torch.from_numpy(selected_point.copy()))
            pred_mean, pred_std = prediction.mean.clone().numpy(), prediction.stddev.clone().numpy()

        print(f'stg={stg}, EI={norm_ei} ({ei_val}), batch={result.x}, prev_y=None, nfev={result.total_nfev}')
        print(f'\tprediction: mean={pred_mean}, std={pred_std}')

        if ei_max < norm_ei:
            next_stg = stg
            ei_max = norm_ei
            best_input = selected_point
            best_ctrl_batch = batch_x
            stock_i = None

        
        for i, stock in enumerate(self.sorted_stocks):
            stg = stock.out_stage + 1
            nfev = 0
            if stock.need_update:
                result = self._ei_optimize(obs_stage=stg, prev_y=stock.y_value,
                                           additional_point=stock.suggested_ctrl_param)
                ei_val = result.fun
                norm_ei = ei_val / self.cascade.remain_cost[stg]
                nfev = result.total_nfev
                batch_x = result.x
                stock.opt_x = batch_x[:self.cascade.ctrl_dim[stg]].reshape(1, -1)
                stock.af_value = norm_ei
                stock.need_update = False
            else:
                batch_x = stock.suggested_ctrl_param
            norm_ei = stock.af_value
            ei_val = norm_ei * self.cascade.remain_cost[stg]

            selected_point = np.concatenate([stock.y_value.flatten(), stock.opt_x.flatten()]).reshape(1, -1)
            with torch.no_grad():
                prediction = self.model.predict(stg, torch.from_numpy(selected_point.copy()))
                pred_mean, pred_std = prediction.mean.clone().numpy(), prediction.stddev.clone().numpy()

            print(f'stg={stg}, EI={norm_ei} ({ei_val}), batch={batch_x}, prev_y={stock.y_value}, nfev={nfev}')
            print(f'\tprediction: mean={pred_mean}, std={pred_std}')

            if ei_max < norm_ei:
                next_stg = stg
                ei_max = norm_ei
                best_input = selected_point
                best_ctrl_batch = batch_x
                stock_i = i
                
        best_ctrl_batch = best_ctrl_batch.flatten()
        if next_stg == self.n_stage - 1:
            remain_param = None
        else:
            remain_param = best_ctrl_batch[self.cascade.ctrl_dim[next_stg]:].flatten()
        result = DecisionResult(next_stg=next_stg, next_input=best_input, af=ei_max, stock_index=stock_i,
                                remain_param=remain_param)
        return result

    def _ei_non_adaptive_decision(self, full_sample: bool = False) -> DecisionResult:
        assert self._check_seq_state()
        if len(self.sorted_stocks) != 0:
            stock = self.sorted_stocks[0]
            next_stg = stock.out_stage + 1
            stock_i = 0
            next_x = np.concatenate([stock.y_value.flatten(), self.decided_param[-1].flatten()]).reshape(1, -1)
            self.decided_param.pop(-1)
            result = DecisionResult(next_stg=next_stg, next_input=next_x, af=stock.af_value, stock_index=stock_i)
            return result

        next_stg = 0
        result = self._ei_optimize(obs_stage=next_stg, prev_y=None, full_sample=full_sample)
        ei_val = result.fun
        batch_x = result.x
        stock_i = None
        prev_y = None

        decided_param = []
        i = 0
        for param_d in self.cascade.ctrl_dim:
            decided_param.insert(0, result.x[i:i + param_d].flatten())
            i += param_d
        self.decided_param = decided_param
        next_x = decided_param.pop(-1).reshape(1, -1)

        print(f'stg={next_stg}, EI={ei_val}, batch={batch_x}, {prev_y=}, nfev={result.total_nfev}')
        result = DecisionResult(next_stg=next_stg, next_input=next_x, af=ei_val, stock_index=stock_i)
        return result

    def _ei_seq_decision(self) -> DecisionResult:
        assert self._check_seq_state()

        if len(self.sorted_stocks) == 0:
            next_stg = 0
            result = self._ei_optimize(obs_stage=next_stg, prev_y=None)
            ei_val = result.fun
            batch_x = result.x
            next_x = batch_x[:self.model.in_dims[next_stg]].reshape(1, -1)
            stock_i = None
            prev_y = None
        else:
            stock = self.sorted_stocks[0]
            next_stg = stock.out_stage + 1
            result = self._ei_optimize(obs_stage=next_stg, prev_y=stock.y_value,
                                       additional_point=stock.suggested_ctrl_param)
            ei_val = result.fun
            batch_x = result.x
            prev_y = stock.y_value.flatten()
            next_x = np.hstack([prev_y, batch_x[:self.cascade.ctrl_dim[next_stg]]]).reshape(1, -1)
            stock_i = 0
        print(f'stg={next_stg}, EI={ei_val}, batch={batch_x}, {prev_y=}, nfev={result.total_nfev}')
        if next_stg == self.n_stage - 1:
            remain_param = None
        else:
            remain_param = batch_x[self.cascade.ctrl_dim[next_stg]:].flatten()
        result = DecisionResult(next_stg=next_stg, next_input=next_x, af=ei_val, stock_index=stock_i,
                                remain_param=remain_param)
        return result

    def cascade_ucb(self, remain_ctrl_param: np.ndarray, obs_stage: int, prev_y=None, with_grad: bool = False,
                    lcb: bool = False):
        if prev_y is not None:
            prev_y = prev_y.flatten()
        cUCB = CascadeUCB(self.model, root_beta=self.root_beta, lipschitz=self.lipschitz, lcb=lcb)
        if with_grad:
            x = to_float_tensor(remain_ctrl_param)
            x.requires_grad = True
            if prev_y is None:
                x_cat = x
            else:
                x_cat = torch.cat([to_float_tensor(prev_y), x])
            x_cat = x_cat.reshape(1, -1)
            ucb = cUCB.evaluate(obs_stage=obs_stage, x=x_cat)
            grad = torch.autograd.grad(ucb, x)[0]
            return ucb.detach().clone().numpy().astype(np.float64), grad.detach().clone().numpy().astype(np.float64)
        else:
            nx = remain_ctrl_param.shape[0]
            x = to_float_tensor(remain_ctrl_param)
            if prev_y is None:
                x_cat = x
            else:
                x_cat = torch.cat([torch.tile(to_float_tensor(prev_y), (nx, 1)), x], dim=1)
            with torch.no_grad():
                ucb = cUCB.evaluate(obs_stage=obs_stage, x=x_cat)
            return ucb.clone().numpy().astype(np.float64)

    def _ucb_optimize(self, obs_stage: int, prev_y=None, additional_point=None, lcb: bool = False):
        f_g = ft.partial(self.cascade_ucb, obs_stage=obs_stage, prev_y=prev_y, with_grad=True, lcb=lcb)
        f = ft.partial(self.cascade_ucb, obs_stage=obs_stage, prev_y=prev_y, with_grad=False, lcb=lcb)
        result = self.af_optimizer.run(f_g,
                                       bounds=self.cascade.ctrl_domain_to_end(obs_stage), minimize=not self.maximize,
                                       additional_init_points=additional_point, jac=True, skip_fzero_point=False,
                                       f_to_init_eval=f)
        return result

    def cascade_uncertainty(self, remain_ctrl_param: np.ndarray, obs_stage: int, prev_y=None, with_grad: bool = False):
        if prev_y is not None:
            prev_y = prev_y.flatten()
        cVar = CascadeUncertainty(self.model, lipschitz=self.lipschitz)
        if with_grad:
            x = to_float_tensor(remain_ctrl_param)
            x.requires_grad = True
            if prev_y is None:
                x_cat = x
            else:
                x_cat = torch.cat([to_float_tensor(prev_y), x])
            x_cat = x_cat.reshape(1, -1)
            var = cVar.evaluate(obs_stage=obs_stage, x=x_cat)
            grad = torch.autograd.grad(var, x)[0]
            return var.detach().clone().numpy().astype(np.float64), grad.detach().clone().numpy().astype(np.float64)
        else:
            nx = remain_ctrl_param.shape[0]
            x = to_float_tensor(remain_ctrl_param)
            if prev_y is None:
                x_cat = x
            else:
                x_cat = torch.cat([torch.tile(to_float_tensor(prev_y), (nx, 1)), x], dim=1)
            with torch.no_grad():
                var = cVar.evaluate(obs_stage=obs_stage, x=x_cat)
            return var.clone().numpy().astype(np.float64)

    def _var_optimize(self, obs_stage: int, prev_y=None, additional_point=None):
        f_g = ft.partial(self.cascade_uncertainty, obs_stage=obs_stage, prev_y=prev_y, with_grad=True)
        f = ft.partial(self.cascade_uncertainty, obs_stage=obs_stage, prev_y=prev_y, with_grad=False)
        result = self.af_optimizer.run(f_g,
                                       bounds=self.cascade.ctrl_domain_to_end(obs_stage), minimize=False,
                                       additional_init_points=additional_point, jac=True, skip_fzero_point=False,
                                       f_to_init_eval=f)
        return result

    def _ucb_seq_upd_old_decision(self) -> DecisionResult:
        assert self._check_seq_state()

        if len(self.sorted_stocks) == 0:
            next_stg = 0
            result = self._ucb_optimize(obs_stage=next_stg, prev_y=None, lcb=not self.maximize)
            ucb_val = result.fun
            batch_x = result.x
            next_x = batch_x[:self.model.in_dims[next_stg]].reshape(1, -1)
            stock_i = None
            prev_y = None
        else:
            stock = self.sorted_stocks[0]
            next_stg = stock.out_stage + 1
            result = self._ucb_optimize(obs_stage=next_stg, prev_y=stock.y_value, lcb=not self.maximize,
                                        additional_point=stock.suggested_ctrl_param)
            ucb_val = result.fun
            batch_x = result.x
            prev_y = stock.y_value.flatten()
            next_x = np.hstack([prev_y, batch_x[:self.cascade.ctrl_dim[next_stg]]]).reshape(1, -1)
            stock_i = 0
        print(f'stg={next_stg}, UCB={ucb_val}, batch={batch_x}, {prev_y=}, nfev={result.total_nfev}')
        if next_stg == self.n_stage - 1:
            remain_param = None
        else:
            remain_param = batch_x[self.cascade.ctrl_dim[next_stg]:].flatten()
        result = DecisionResult(next_stg=next_stg, next_input=next_x, af=ucb_val, stock_index=stock_i,
                                remain_param=remain_param)
        return result

    def _ucb_seq_upd_decision(self) -> DecisionResult:
        assert self._check_seq_state()

        self.update_target_lower()
        eta = 1e-4 / (1 + np.log(self.iteration + 1))

        if len(self.sorted_stocks) == 0:
            next_stg = 0
            result_ucb = self._ucb_optimize(obs_stage=next_stg, prev_y=None, lcb=not self.maximize)
            result_var = self._var_optimize(obs_stage=next_stg, prev_y=None)
            

            ucb_improve = result_ucb.fun - self.target_lower
            if not self.maximize:
                ucb_improve *= -1
            scaled_var = result_var.fun * eta
            if ucb_improve >= scaled_var:
                af = ucb_improve
                batch_x = result_ucb.x
            else:
                print(f'ucb_improve {float(ucb_improve)}  < scaled_var {float(scaled_var)}')
                af = scaled_var
                batch_x = result_var.x
            next_x = batch_x[:self.model.in_dims[next_stg]].reshape(1, -1)
            stock_i = None
            prev_y = None
        else:
            stock = self.sorted_stocks[0]
            next_stg = stock.out_stage + 1
            result_ucb = self._ucb_optimize(obs_stage=next_stg, prev_y=stock.y_value, lcb=not self.maximize,
                                            additional_point=stock.suggested_ctrl_param)
            result_var = self._var_optimize(obs_stage=next_stg, prev_y=stock.y_value)
            ucb_improve = result_ucb.fun - self.target_lower
            scaled_var = result_var.fun * eta
            if ucb_improve >= scaled_var:
                af = ucb_improve
                batch_x = result_ucb.x
            else:
                af = scaled_var
                batch_x = result_var.x
            prev_y = stock.y_value.flatten()
            next_x = np.hstack([prev_y, batch_x[:self.cascade.ctrl_dim[next_stg]]]).reshape(1, -1)
            stock_i = 0
        nfev = result_ucb.total_nfev + result_var.total_nfev
        print(f'stg={next_stg}, AF={af}, batch={batch_x}, {prev_y=}, nfev={nfev}')
        if next_stg == self.n_stage - 1:
            remain_param = None
        else:
            remain_param = batch_x[self.cascade.ctrl_dim[next_stg]:].flatten()
        result = DecisionResult(next_stg=next_stg, next_input=next_x, af=af, stock_index=stock_i,
                                remain_param=remain_param)
        return result

    def _ucb_seq_decision(self) -> DecisionResult:
        assert self._check_seq_state()

        if len(self.sorted_stocks) != 0:
            stock = self.sorted_stocks[0]
            next_stg = stock.out_stage + 1
            stock_i = 0
            next_x = np.concatenate([stock.y_value.flatten(), self.decided_param[-1].flatten()]).reshape(1, -1)
            self.decided_param.pop(-1)
            result = DecisionResult(next_stg=next_stg, next_input=next_x, af=stock.af_value, stock_index=stock_i)
            return result

        
        next_stg = 0
        result = self._ucb_optimize(obs_stage=next_stg, prev_y=None, lcb=False)
        ucb_val = result.fun
        batch_x = result.x

        decided_param = []
        i = 0
        for param_d in self.cascade.ctrl_dim:
            decided_param.insert(0, result.x[i:i + param_d].flatten())
            i += param_d
        next_x = decided_param.pop(-1).reshape(1, -1)
        stock_i = None
        prev_y = None

        self.decided_param = decided_param

        print(f'stg={next_stg}, UCB={ucb_val}, batch={batch_x}, {prev_y=}, nfev={result.total_nfev}')
        result = DecisionResult(next_stg=next_stg, next_input=next_x, af=ucb_val, stock_index=stock_i)
        return result

    def _random_decision(self) -> DecisionResult:
        assert self._check_seq_state()
        if len(self.sorted_stocks) == 0:
            next_stg = 0
            bounds = np.array(self.x_domain[next_stg])
            next_x = self.rng.uniform(bounds[:, 0], bounds[:, 1]).reshape(1, -1)
            stock_i = None
        else:
            stock = self.sorted_stocks[0]
            next_stg = stock.out_stage + 1
            bounds = np.array(self.x_domain[next_stg])
            next_param = self.rng.uniform(bounds[:, 0], bounds[:, 1]).flatten()
            next_x = np.concatenate([stock.y_value.flatten(), next_param]).reshape(1, -1)
            stock_i = 0
        result = DecisionResult(next_stg=next_stg, next_input=next_x, af=None, stock_index=stock_i)
        return result

    def cbo(self, ctrl_param_i: np.ndarray, obs_stage: int, desired_y: np.ndarray = None, with_grad=False):
        AF = CBO(self.model, self.y_best)
        if desired_y is not None:
            desired_y = to_float_tensor(desired_y)
        if with_grad:
            x = to_float_tensor(ctrl_param_i).reshape(1, -1)
            x.requires_grad = True

            af_value = AF.evaluate(obs_stage=obs_stage, x=x, desired_prev_y=desired_y)
            grad = torch.autograd.grad(af_value, x)[0].detach().clone().numpy().astype(np.float64)
            af_value = af_value.detach().clone().numpy().astype(np.float64)
            return af_value, grad
        else:
            x = to_float_tensor(ctrl_param_i)
            with torch.no_grad():
                af_value = AF.evaluate(obs_stage=obs_stage, x=x, desired_prev_y=desired_y)
            return af_value.clone().numpy().astype(np.float64)

    def _cbo_optimize(self, obs_stage: int, desired_y=None, y_range: CBORange = 'true'):
        minimize = obs_stage != self.n_stage - 1
        skip_fzero = obs_stage == self.n_stage - 1
        bounds_arr = np.array(self.cascade.bench[obs_stage].bounds)
        if y_range == 'true' or obs_stage == 0:
            pass
        else:
            prev_out_dim = self.cascade.out_dim[obs_stage - 1]
            if y_range == 'wide':
                factor = 2
            elif y_range == 'narrow':
                factor = 0.5
            else:
                raise ValueError(f'Unknown y_range: {y_range}')
            length = (bounds_arr[:prev_out_dim, 1] - bounds_arr[:prev_out_dim, 0]) * factor
            center = (bounds_arr[:prev_out_dim, 1] + bounds_arr[:prev_out_dim, 0]) / 2
            bounds_arr[:prev_out_dim:, 1] = center + length / 2
            bounds_arr[:prev_out_dim:, 0] = center - length / 2

        bounds = [tuple(item) for item in bounds_arr]
        result = self.af_optimizer.run(ft.partial(self.cbo, obs_stage=obs_stage, desired_y=desired_y, with_grad=True),
                                       bounds=bounds, jac=True,
                                       minimize=minimize,
                                       skip_fzero_point=skip_fzero,
                                       f_to_init_eval=ft.partial(self.cbo, obs_stage=obs_stage, desired_y=desired_y,
                                                                 with_grad=False))
        return result

    def _cbo_decision(self, y_range: CBORange = 'true') -> DecisionResult:
        assert self._check_seq_state()
        if len(self.sorted_stocks) != 0:
            stock = self.sorted_stocks[0]
            next_stg = stock.out_stage + 1
            stock_i = 0
            next_x = np.concatenate([stock.y_value.flatten(), self.decided_param[-1].flatten()]).reshape(1, -1)
            self.decided_param.pop(-1)
            result = DecisionResult(next_stg=next_stg, next_input=next_x, af=None, stock_index=stock_i)
            return result

        af_hist = []

        i = self.cascade.n_stage - 1
        result = self._cbo_optimize(obs_stage=i, desired_y=None, y_range=y_range)
        decided_param = [result.x[self.model.out_dims[i - 1]:].flatten()]
        desired_y = result.x[:self.model.out_dims[i - 1]].flatten()
        af_hist.append(float(result.fun))
        with torch.no_grad():
            prediction = self.model.predict(i, torch.from_numpy(result.x.copy().reshape(1, -1)))
        print(f'stg={i}, EI={af_hist[-1]}, param={decided_param[-1]}, {desired_y=}, nfev={result.total_nfev}')
        print(f'\tprediction: mean={prediction.mean.clone().numpy()}, std={prediction.stddev.clone().numpy()}')
        i -= 1

        while i > 0:
            result = self._cbo_optimize(obs_stage=i, desired_y=desired_y, y_range=y_range)
            decided_param.append(result.x[self.model.out_dims[i - 1]:].flatten())
            desired_y = result.x[:self.model.out_dims[i - 1]].flatten()
            af_hist.append(float(result.fun))
            with torch.no_grad():
                prediction = self.model.predict(i, torch.from_numpy(result.x.copy().reshape(1, -1)))
            print(f'stg={i}, CBO={af_hist[-1]}, param={decided_param[-1]}, {desired_y=}, nfev={result.total_nfev}')
            print(f'\tprediction: mean={prediction.mean.clone().numpy()}, std={prediction.stddev.clone().numpy()}')
            i -= 1

        
        result = self._cbo_optimize(obs_stage=i, desired_y=desired_y, y_range=y_range)
        next_x = result.x.reshape(1, -1)
        af_hist.append(float(result.fun))

        with torch.no_grad():
            prediction = self.model.predict(i, torch.from_numpy(result.x.copy().reshape(1, -1)))
        print(f'stg={i}, CBO={af_hist[-1]}, param={decided_param[-1]}, nfev={result.total_nfev}')
        print(f'\tprediction: mean={prediction.mean.clone().numpy()}, std={prediction.stddev.clone().numpy()}')

        self.decided_param = decided_param
        result = DecisionResult(next_stg=0, next_input=next_x, af=af_hist[::-1], stock_index=None)
        return result

    def _af_joint(self, entire_ctrl_param: np.ndarray, joint_model: SingleTaskGP, with_grad: bool = False,
                  af: str = 'EI'):
        if af == 'EI':
            AF = EI(joint_model, self.y_best, maximize=self.maximize)
        elif af == 'UCB':
            AF = UCB(joint_model, root_beta=self.root_beta, lcb=not self.maximize)
        else:
            raise ValueError('Unknown acquisition function')

        if with_grad:
            x = to_float_tensor(entire_ctrl_param).reshape(1, -1)
            x.requires_grad = True

            af_value = AF.evaluate(x)
            grad = torch.autograd.grad(af_value, x)[0].detach().clone().numpy().astype(np.float64)
            af_value = af_value.detach().clone().numpy().astype(np.float64)
            return af_value, grad
        else:
            x = to_float_tensor(entire_ctrl_param)
            with torch.no_grad():
                af_value = AF.evaluate(x)
            return af_value.clone().numpy().astype(np.float64)

    def _joint_decision(self, af: str = 'EI') -> DecisionResult:
        assert self._check_seq_state()
        if len(self.sorted_stocks) != 0:
            stock = self.sorted_stocks[0]
            next_stg = stock.out_stage + 1
            stock_i = 0
            next_x = np.concatenate([stock.y_value.flatten(), self.decided_param[-1].flatten()]).reshape(1, -1)
            self.decided_param.pop(-1)
            result = DecisionResult(next_stg=next_stg, next_input=next_x, af=None, stock_index=stock_i)
            return result

        
        all_x = self.model.all_train_inputs
        all_param = [all_x[0]]
        if self.model.models[0].use_ard:
            all_ls_bounds = self.model.models[0].lengthscale_bounds
        else:
            ls_bound_i = self.model.models[0].lengthscale_bounds
            all_ls_bounds = [ls_bound_i[0] for _ in range(self.model.in_dims[0])]
        for i, xi in enumerate(all_x[1:]):
            out_dim = self.model.out_dims[i]  
            all_param.append(xi[:, out_dim:])

            ls_bound_i = self.model.models[i + 1].lengthscale_bounds
            if ls_bound_i is None:
                raise ValueError('lengthscale_bounds must be set.')
            if self.model.models[i].use_ard:
                all_ls_bounds += ls_bound_i[out_dim:]
            else:
                new_ls_bound_i = [ls_bound_i[0] for _ in range(self.model.in_dims[i + 1])]
                all_ls_bounds += new_ls_bound_i

        train_x = torch.cat(all_param, dim=1)
        train_y = torch.flatten(self.model.all_train_targets[-1])
        noise_var = self.model.models[-1].noise_var
        os_bound = self.model.models[-1].outputscale_bounds
        if isinstance(os_bound, list):
            os_bound = os_bound[0]
        joint_model = SingleTaskGP(train_x, train_y, noise_var,
                                   outputscale_bounds=os_bound, lengthscale_bounds=all_ls_bounds, use_ard=True)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(joint_model.likelihood, joint_model)
        fit_param(mll, max_retries=20)

        result = self.af_optimizer.run(ft.partial(self._af_joint, joint_model=joint_model, with_grad=True, af=af),
                                       bounds=self.cascade.ctrl_domain_to_end(0),
                                       jac=True, skip_fzero_point=True, minimize=False,
                                       f_to_init_eval=ft.partial(self._af_joint,
                                                                 joint_model=joint_model,
                                                                 with_grad=False, af=af))
        with torch.no_grad():
            prediction = joint_model.predict(torch.from_numpy(result.x.copy().reshape(1, -1)))
        decided_param = []
        i = 0
        for param_d in self.cascade.ctrl_dim:
            decided_param.insert(0, result.x[i:i + param_d].flatten())
            i += param_d
        next_x = decided_param.pop(-1).reshape(1, -1)
        print(f'stg=0, {af}={result.fun}, param={next_x.flatten()}, nfev={result.total_nfev}')
        print(f'prediction w.r.t. F(x1, ..., xn): mean={prediction.mean.clone().numpy()}, ', end='')
        print(f'std={prediction.stddev.clone().numpy()}')

        self.decided_param = decided_param
        result = DecisionResult(next_stg=0, next_input=next_x, af=result.fun, stock_index=None)
        return result

    def observe_next(self, fit_model: bool = True, *, enable_memory_profile=False, log_stream=None):
        begin = time.time()
        print(f'--- cost={self.cost_hist[-1]}, iter={self.iteration} ---')
        print(f'obs_best={self.y_best}, s_regret={self.s_regret_hist[-1]}, i_regret={self.i_regret_hist[-1]}, ', end='')
        total_suspend = np.count_nonzero(self.suspend_hist)
        total_discard = sum(self.n_discarded)
        if len(self.sorted_stocks) == 0:
            prev_stock_index = None
        else:
            prev_stock_index = int(np.argmax([item.obtained_iteration for item in self.sorted_stocks]))
            if self.sorted_stocks[prev_stock_index].obtained_iteration != self.iteration - 1:
                prev_stock_index = None
        print(f'stock={len(self.sorted_stocks)}, prev_stock={prev_stock_index}, {total_suspend=}, {total_discard=}')
        print('--- decide next observation ---')
        if enable_memory_profile:
            result = memory_profiler.profile(self._get_next, stream=log_stream)()
        else:
            result = self._get_next()

        next_stg = result.next_stg
        next_x = result.next_input
        af = result.af
        next_f = self.cascade.bench[next_stg].eval_f(next_x)
        next_y = next_f + self.rng.normal(scale=self.noise_scale, size=next_f.shape)
        next_cost = self.cascade.each_cost[next_stg]
        index = result.stock_index

        print('--- observation ---')
        is_suspend = (prev_stock_index is not None) and (index != prev_stock_index)
        with torch.no_grad():
            prediction = self.model.predict(next_stg, to_float_tensor(next_x))
            pred_mean, pred_std = prediction.mean.clone().numpy(), prediction.stddev.clone().numpy()

        print(f'stg={next_stg}, AF={af}, input={next_x}, stock_index={index}, {is_suspend=}')
        print(f'output={next_y}, noiseless=({next_f}), : mean={pred_mean}, std={pred_std}')

        
        self.model.add_observations(next_stg, to_float_tensor(next_x), to_float_tensor(next_y))
        if next_stg == self.n_stage - 1:
            if self.maximize:
                self.f_best = float(max(self.f_best, next_f))
                self.y_best = float(max(self.y_best, next_y))
            else:
                self.f_best = float(min(self.f_best, next_f))
                self.y_best = float(min(self.y_best, next_y))
        s_regret = float(self.f_opt - self.f_best)
        if not self.maximize:
            s_regret *= -1
        self.s_regret_hist.append(s_regret)
        self.obs_best_hist.append(float(self.y_best))

        self.prev_stage = next_stg
        self.cost_hist.append(self.cost_hist[-1] + next_cost)
        self.n_stock_hist.append(len(self.sorted_stocks))
        self.suspend_hist.append(is_suspend)
        self.obs_stage_hist.append(next_stg)

        if index is not None:
            if self.sorted_stocks[index].remain_times == 1:
                self.sorted_stocks.pop(index)
            else:
                self.sorted_stocks[index].remain_times -= 1

        for stock in self.sorted_stocks:
            if stock.out_stage < next_stg:
                stock.need_update = True
                stock.target_lower.need_update = True
                stock.target_upper.need_update = True

        if next_stg != self.n_stage - 1:
            new_stock = OutputStock(next_stg, next_y, need_update=True, remain_times=self.stock_lifetime,
                                    obtained_iteration=self.iteration,
                                    suggested_ctrl_param=result.remain_param)
        else:
            new_stock = None

        
        if fit_model:
            print('--- fit model ---')
            self.model.fit_hyper_param(next_stg)

        
        if self.eval_inference_r:
            print('--- calculate inference regret---')
            suggest_x, lower = self.infer_best(additional_point=self.suggested_x)
            f_inferred = self.cascade.observe_to_end(suggest_x)
            self.infer_best_hist.append(float(f_inferred))
            i_regret = float(self.f_opt - f_inferred)
            if not self.maximize:
                i_regret *= -1
            self.i_regret_hist.append(i_regret)
            self.suggested_x = suggest_x
            print(f'max_cLCB{lower}, f_inferred={float(f_inferred)}')
            self.target_lower = float(lower)
        else:
            self.i_regret_hist.append(None)

        
        begin_discard = time.time()
        n_discard = self._discard_strategy(new_stock)
        self.n_discarded.append(n_discard)
        end_discard = time.time()
        time_discard = datetime.timedelta(seconds=end_discard - begin_discard)
        self.discard_time.append(time_discard)
        print(f'{n_discard=}')
        print(f'(discard) elapsed time: {time_discard}')

        
        self.update_base_sample()
        end = time.time()
        time_delta = datetime.timedelta(seconds=end - begin)
        self.elapsed_time.append(time_delta)
        total_second = sum([item.total_seconds() for item in self.elapsed_time])
        print(f'iteration time={time_delta}, total={datetime.timedelta(seconds=total_second)}')
        self.iteration += 1
        return

    def update_base_sample(self):
        if not self.is_suspendable():
            self.base_samples.clear()
            for i in range(self.n_stage):
                bs = self.rng.standard_normal(size=(self.mc_samples, self.model.out_dims[i]))
                self.base_samples.append(to_float_tensor(bs))
        else:
            end = min(self.prev_stage + 1, self.n_stage)
            for i in range(end):
                bs = self.rng.standard_normal(size=(self.mc_samples, self.model.out_dims[i]))
                self.base_samples[i] = to_float_tensor(bs)
        return

    def add_new_stock(self, stock: typing.Optional[OutputStock]):
        if stock is None:
            return
        keys = [self.ci_sort_key(item) for item in self.sorted_stocks]
        insert_i = bisect.bisect_right(keys, self.ci_sort_key(stock))
        self.sorted_stocks.insert(insert_i, stock)
        return

    def _no_discard(self, new_stock: typing.Optional[OutputStock]):
        self.add_new_stock(new_stock)
        return 0

    def _discard_all(self, new_stock: typing.Optional[OutputStock]):
        
        if new_stock is None:
            return 0
        n_discard = len(self.sorted_stocks)
        self.sorted_stocks.clear()
        self.add_new_stock(new_stock)
        return n_discard

    def _discard_stocks_by_fifo(self, new_stock: typing.Optional[OutputStock]):
        if new_stock is None:
            return 0
        if len(self.sorted_stocks) + 1 <= self.max_stocks:
            self.add_new_stock(new_stock)
            return 0
        oldest_index = np.argmin([item.obtained_iteration for item in self.sorted_stocks])
        self.sorted_stocks.pop(int(oldest_index))
        self.add_new_stock(new_stock)
        return 1

    def _discard_stocks_by_af(self, new_stock: typing.Optional[OutputStock]):
        if new_stock is None:
            return 0
        if len(self.sorted_stocks) + 1 <= self.max_stocks:
            self.add_new_stock(new_stock)
            return 0
        stg = new_stock.out_stage + 1
        result = self._ei_optimize(obs_stage=stg, prev_y=new_stock.y_value,
                                   additional_point=new_stock.suggested_ctrl_param)
        ei_val = result.fun
        norm_ei = ei_val / self.cascade.remain_cost[stg]
        nfev = result.total_nfev
        batch_x = result.x
        new_stock.opt_x = batch_x[:self.cascade.ctrl_dim[stg]].reshape(1, -1)
        new_stock.af_value = norm_ei
        new_stock.need_update = False
        print(f'\tcalc AF of new stock:{nfev=}')
        self.add_new_stock(new_stock)

        discard_index = np.argmin([item.af_value for item in self.sorted_stocks])
        self.sorted_stocks.pop(int(discard_index))
        return 1

    def _discard_stocks_by_ci(self, new_stock: typing.Optional[OutputStock]):
        assert self.discard_stock_interval > 0
        print('--- check potential maximizer ---')
        check_all_stocks = self.iteration % self.discard_stock_interval == 0
        if not check_all_stocks:
            self.add_new_stock(new_stock)
            return 0

        
        self.update_target_lower()
        print(f'max lcb : {self.target_lower}')

        
        discard_index = []
        for index, stock in enumerate(self.sorted_stocks):
            if self.maximize:
                upper = stock.target_upper
            else:
                upper = stock.target_lower
            if upper.need_update or upper.value is None:
                print(upper.value)
                result = self._ucb_optimize(obs_stage=stock.out_stage + 1, prev_y=stock.y_value, lcb=not self.maximize,
                                            additional_point=upper.x)
                
                
                
                
                upper.x = result.x
                upper.value = result.fun
                upper.need_update = False
            print(upper.value)
            diff = upper.value - self.target_lower
            if not self.maximize:
                diff *= -1
            if diff < 0:
                discard_index.append(index)
                print(f'{index}: max ucb = {upper.value}')

        discard_index.sort(reverse=True)
        for i in discard_index:
            self.sorted_stocks.pop(i)
        self.add_new_stock(new_stock)
        return len(discard_index)

    @staticmethod
    def ci_sort_key(stock: OutputStock):
        return stock.out_stage
