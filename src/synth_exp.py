import argparse
import atexit
import gc
import pathlib
import pickle
import re
import sys

import gpytorch
import memory_profiler
import numpy as np
import torch

import benchmark
import benchmark as bm
import util
from bo import MultioutCascadeBO, SynthCascade
from models import CascadeMOGP, IndependentMOGP




def one_exp(seed: int,  setting_name: str, method: str, func: str,
            max_cycle: int, n_thread: int = 1, n_init: int = 10,
            cost_policy='same', noise_scale=1e-2,
            stock_lifetime: int = 1, discard_strategy: str = 'save_all', max_stocks: int = 10,
            discard_interval: int = 1, root_beta: int = 2,
            print_to_file: bool = False, profile_memory: bool = False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if discard_strategy == 'save_all':
        discard_abbr = ''
    elif discard_strategy == 'discard_all':
        discard_abbr = '_Dall'
    elif discard_strategy == 'fifo':
        discard_abbr = f'_Dfifo{max_stocks}'
    elif discard_strategy == 'af':
        discard_abbr = f'_Daf{max_stocks}'
    elif discard_strategy == 'ci':
        discard_abbr = f'_Dci_intv{discard_interval}_b{root_beta:.2f}'
    else:
        raise ValueError(f'Unknown discard strategy: {discard_strategy}')

    id_str = f'{method}_l{stock_lifetime}{discard_abbr}_seed{seed:0>3}_{cost_policy}'

    id_dir = result_dir / id_str
    id_dir.mkdir(exist_ok=True)
    model_dir = id_dir / 'model'
    model_dir.mkdir(exist_ok=True)
    log_dir = result_dir / 'out'
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True)

    def close_file(file_obj):
        file_obj.close()
        return

    f_out = None
    f_err = None
    log_stream = None
    if print_to_file:
        f_out = open(log_dir / (id_str + '.o'), mode='w', buffering=1)
        f_err = open(log_dir / (id_str + '.e'), mode='w', buffering=1)

        sys.stdout = f_out
        sys.stderr = f_err

        atexit.register(close_file, f_out)
        atexit.register(close_file, f_err)

        
        if profile_memory:
            log_stream = open(log_dir / (id_str + '_mem_profile.log'), mode='w', buffering=1)
            atexit.register(close_file, log_stream)

    
    n_stage = int(re.search(r'(?<=s)\d+', setting_name).group())
    each_param = int(re.search(r'(?<=d)\d+', setting_name).group())
    each_cost = [1 for _ in range(n_stage)]
    if cost_policy == 'same':
        pass
    elif cost_policy == 'cons':
        each_cost[-1] = 10
    elif cost_policy == 'head':
        each_cost[0] = 10
    else:
        print(f'Unknown cost_policy: {cost_policy}')
        sys.exit(1)

    output_dim = 1
    ctrl_dim = [each_param for _ in range(n_stage)]
    input_dim = [each_param + output_dim for _ in range(n_stage)]
    input_dim[0] = each_param

    if func == 'ackley':
        base_bounds = [[(-2., 2.) for _ in range(ctrl_dim[i])] for i in range(n_stage)]
        minmax_range = (-2., 2.)
    elif func == 'rosenbrock':
        base_bounds = [[(-2., 2.)  for _ in range(ctrl_dim[i])] for i in range(n_stage)]
        minmax_range = (-2., 2.)
    elif func == 'levy':
        base_bounds = [[(-10., 10.)  for _ in range(ctrl_dim[i])] for i in range(n_stage)]
        minmax_range = (-10., 10.)
    elif func == 'schwefel':
        base_bounds = [[(-500., 500.)  for _ in range(ctrl_dim[i])] for i in range(n_stage)]
        minmax_range = (-500., 500.)
    elif func == 'rastrigin':
        base_bounds = [[(-5.12, 5.12)  for _ in range(ctrl_dim[i])] for i in range(n_stage)]
        minmax_range = (-5.12, 5.12)
    elif func == 'sphere':
        base_bounds = [[(-5.12, 5.12)  for _ in range(ctrl_dim[i])] for i in range(n_stage)]
        minmax_range = (-5.12, 5.12)
    elif func == 'beale':
        input_dim[0] = 2
        ctrl_dim[0] = 2
        base_bounds = [[(-4.5, 4.5)  for _ in range(ctrl_dim[i])] for i in range(n_stage)]
        minmax_range = (-4.5, 4.5)
    elif func == 'matyas':
        input_dim[0] = 2
        ctrl_dim[0] = 2
        base_bounds = [[(-10, 10)  for _ in range(ctrl_dim[i])] for i in range(n_stage)]
        minmax_range = (-10, 10)
    else:
        raise ValueError('Unknown benchmark function')

    output_range = []
    func_list = []
    ctrl_domain = []

    for i in range(n_stage):
        ctrl_domain.append(base_bounds[i].copy())
        bounds = base_bounds[i].copy()
        if i > 0:
            bounds.insert(0, output_range[-1])
        if func == 'ackley':
            f_i = bm.NegAckley(dim=input_dim[i], bounds=bounds, minmax_range=minmax_range)
        elif func == 'rosenbrock':
            f_i = bm.NegRosenbrock(dim=input_dim[i], bounds=bounds, minmax_range=minmax_range)
        elif func == 'levy':
            f_i = bm.NegLevy(dim=input_dim[i], bounds=bounds, minmax_range=minmax_range)
        elif func == 'schwefel':
            f_i = bm.NegSchwefel(dim=input_dim[i], bounds=bounds, minmax_range=minmax_range)
        elif func == 'rastrigin':
            f_i = bm.NegRastrigin(dim=input_dim[i], bounds=bounds, minmax_range=minmax_range)
        elif func == 'sphere':
            f_i = bm.NegSphere(dim=input_dim[i], bounds=bounds, minmax_range=minmax_range)
        elif func == 'beale':
            f_i = bm.NegBeale(dim=input_dim[i], bounds=bounds, minmax_range=minmax_range)
        elif func == 'matyas':
            f_i = bm.NegMatyas(dim=input_dim[i], bounds=bounds, minmax_range=minmax_range)
        else:
            raise ValueError('Unknown benchmark function')
        func_list.append(f_i)
        output_range.append((f_i.min, f_i.max))

    cascade = SynthCascade(func_list, ctrl_dim=ctrl_dim, each_cost=each_cost, ctrl_domain=ctrl_domain)

    all_ctrl_domain = cascade.ctrl_domain_to_end(0)
    bounds = np.array(all_ctrl_domain)
    init_x = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, len(all_ctrl_domain)))
    init_x_sp = cascade.to_ctrl_split(init_x)
    x_train = []
    y_train = []
    for i in range(cascade.n_stage):
        if i == 0:
            x_train.append(init_x_sp[0])
            y_train.append(cascade.bench[0].eval_f(init_x_sp[0]))
        else:
            concat_x = np.hstack([y_train[i - 1], init_x_sp[i]])
            x_train.append(concat_x)
            y_train.append(cascade.bench[i].eval_f(concat_x))
    init_f_max = np.max(y_train[-1])
    for yt in y_train:
        yt += rng.normal(scale=noise_scale, size=yt.shape)

    gp_list = []
    os_base = 4.  
    outputscale_bounds = [(os_base / 2, os_base * 2)]
    lengthscale_bounds = [[((b[1] - b[0]) / 20., (b[1] - b[0]) / 2) for b in fi.bounds] for fi in func_list]

    for i in range(cascade.n_stage):
        xt = util.to_float_tensor(x_train[i])
        yt = util.to_float_tensor(y_train[i])
        ob = outputscale_bounds
        lb = lengthscale_bounds[i]
        gp_i = IndependentMOGP(xt, yt, noise_var=noise_scale ** 2, use_ard=True,
                               lengthscale_bounds=lb, outputscale_bounds=ob,
                               prior_mean=None)
        gp_list.append(gp_i)
    gp = CascadeMOGP(gp_list)
    gp.fit_hyper_param(max_retries=20)
    bo = MultioutCascadeBO(cascade=cascade, model=gp, method=method, seed=seed, maximize=True,
                           stock_lifetime=stock_lifetime, f_best=init_f_max, n_thread=n_thread,
                           discard_strategy=discard_strategy, max_stocks=max_stocks,
                           discard_stocks_interval=discard_interval,
                           ucb_root_beta=root_beta,
                           eval_inference_r=True)

    print(f'--- init ---')
    print(f'{n_init=}, {max_cycle=}')
    print(f'{func}: \n')
    print(f'optimal objective value={cascade.max}')

    max_cost = max_cycle * np.sum(each_cost)
    while bo.current_cost < max_cost:
        
        if profile_memory:
            memory_profiler.profile(bo.observe_next, stream=log_stream)(fit_model=False, enable_memory_profile=True,
                                                                        log_stream=log_stream)
        else:
            bo.observe_next(fit_model=True)
        if bo.iteration % 10 == 0:
            with open(model_dir / f'cost{bo.current_cost:0>3}.pkl', 'wb') as writer:
                pickle.dump(bo, writer)
            
            result_dict = {'cost_hist': bo.cost_hist,
                           'obs_hist': bo.obs_best_hist,
                           's_regret': bo.s_regret_hist,
                           'elapsed_time': bo.elapsed_time,
                           'n_stock': bo.n_stock_hist,
                           'is_suspend': bo.suspend_hist,
                           'observation_stage': bo.obs_stage_hist,
                           'n_discarded': bo.n_discarded,
                           'discard_time': bo.discard_time,
                           'each_cost': each_cost,
                           'max_cycle': max_cycle,
                           'i_regret': bo.i_regret_hist,
                           'root_beta': bo.root_beta,
                           'lipschitz': bo.lipschitz
                           }
            with open(id_dir / f'result_tmp.pkl', 'wb') as writer:
                pickle.dump(result_dict, writer)
        gc.collect()
        print()
    print(f'--- cost={bo.cost_hist[-1]}, iter={bo.iteration} ---')
    n_suspend = np.count_nonzero(bo.suspend_hist)
    print(f'best_f={bo.f_best}, s_regret={bo.s_regret_hist[-1]}, i_regret={bo.i_regret_hist[-1]}, ', end='')
    print(f'stock={len(bo.sorted_stocks)}, suspend={n_suspend}')

    with open(model_dir / f'cost{bo.current_cost:0>3}.pkl', 'wb') as writer:
        pickle.dump(bo, writer)

    
    result_dict = {'cost_hist': bo.cost_hist,
                   'obs_hist': bo.obs_best_hist,
                   's_regret': bo.s_regret_hist,
                   'elapsed_time': bo.elapsed_time,
                   'n_stock': bo.n_stock_hist,
                   'is_suspend': bo.suspend_hist,
                   'observation_stage': bo.obs_stage_hist,
                   'n_discarded': bo.n_discarded,
                   'discard_time': bo.discard_time,
                   'each_cost': each_cost,
                   'max_cycle': max_cycle,
                   'i_regret': bo.i_regret_hist,
                   'root_beta': bo.root_beta,
                   'lipschitz': bo.lipschitz
                   }
    with open(id_dir / f'result.pkl', 'wb') as writer:
        pickle.dump(result_dict, writer)
    print('\n BO finished successfully.')
    if print_to_file:
        f_out.close()
        f_err.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        atexit.unregister(close_file)
    return


def main():
    one_exp(seed=args.seed, setting_name=args.setting, method=args.method, func=args.func,
            max_cycle=args.max_cycle, n_thread=args.thread, n_init=args.n_init, cost_policy=args.policy,
            stock_lifetime=args.lifetime, discard_strategy=args.discard_method, max_stocks=args.max_stock,
            discard_interval=args.discard_interval, root_beta=args.root_beta,
            print_to_file=(not args.debug), profile_memory=args.mprof)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('setting', type=str, help='name of cascade settings')
    parser.add_argument('seed', type=int, help='random seed for numpy & pytorch')
    parser.add_argument('method', type=str, help='sampling method, nearly equal to acquisition function')
    parser.add_argument('policy', type=str, help='cost setting policy')
    parser.add_argument('--func', '-f', required=True, type=str, help='benchmark function name')
    parser.add_argument('--lifetime', '-l', type=int, default=1, help='lifetime of stocks')
    parser.add_argument('--n_init', '-i', type=int, default=10, help='number of initial observation')
    parser.add_argument('--max_cycle', '-m', type=int, default=100, help='Maximum observation cycle')
    parser.add_argument('--thread', '-t', type=int, default=1, help='number of thread for parallel calculation')
    parser.add_argument('--debug', action='store_true', help='print log to console')
    parser.add_argument('--mprof', action='store_true', help='logging memory usage')
    parser.add_argument('--discard_method', '-d', type=str, default='save_all',
                        choices=['save_all', 'discard_all', 'fifo', 'af', 'ci'],
                        help='discard strategy. used only in suspendable settings.')
    parser.add_argument('--max_stock', type=int, default=10,
                        help='maximum number of stocks to store. used only FIFO or AF strategy.')
    parser.add_argument('--discard_interval', type=int, default=1,
                        help='interval for discard. used only CI strategy.')
    parser.add_argument('--root_beta', type=int, default=2,
                        help='beta^{1/2} of ucb & lcb. used for inference regret, cUCB, stock reduction.')

    args = parser.parse_args()

    root_dir = pathlib.Path(__file__).parent.parent.resolve()
    result_dir = root_dir / f'{args.func}_{args.setting}_cycle{args.max_cycle}'
    result_dir.mkdir(exist_ok=True)
    np.set_printoptions(linewidth=150)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    benchmark.set_n_cpu(1)
    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type(torch.DoubleTensor)
    with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False,
                                             solves=False), gpytorch.settings.max_cholesky_size(2000):
        main()
