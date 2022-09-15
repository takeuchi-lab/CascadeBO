import argparse
import atexit
import gc
import pathlib
import pickle
import sys

import gpytorch
import memory_profiler
import numpy as np
import torch

import benchmark
import benchmark as bm
import util
from bo import RealCascade, RealMultioutCascadeBO
from models import CascadeMOGP, IndependentMOGP




def one_exp(seed: int, method: str, max_cycle: int, n_thread: int = 1, n_init: int = 5,
            cost_policy='same', noise_scale=1e-2,
            stock_lifetime: int = 1, discard_strategy: str = 'save_all', max_stocks: int = 10,
            discard_interval: int = 1, root_beta: int = 3, lipschitz=0.1,
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

    id_str = f'{method}_l{stock_lifetime}{discard_abbr}_seed{seed:0>2}_{cost_policy}'

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

    n_stage = 3
    each_cost = [1 for _ in range(n_stage)]
    if cost_policy == 'same':
        pass
    elif cost_policy == 'cons':
        each_cost[-1] = 10
    elif cost_policy == 'heand':
        each_cost[0] = 10
    else:
        print(f'Unknown cost_policy: {cost_policy}')
        sys.exit(1)

    ctrl_dim = [3, 2, 2]

    f1 = bm.PDiffusion1st(seed=seed, log_fit=True, shape='gaussian', parallel=n_thread)
    f2 = bm.PDiffusion2nd(seed=seed, log_fit=True, shape='gaussian', parallel=n_thread)
    f3 = bm.PC1DEmu()
    func_list = [f1, f2, f3]
    ctrl_domain = [f1.bounds, f2.bounds[-2:], f3.bounds[-2:]]
    out_dim = [4, 4, 1]
    cascade = RealCascade(func_list, ctrl_dim=ctrl_dim, each_cost=each_cost, ctrl_domain=ctrl_domain, out_dim=out_dim)

    all_ctrl_domain = cascade.ctrl_domain_to_end(0)
    bounds = np.array(all_ctrl_domain)
    init_x = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, len(all_ctrl_domain)))
    init_x_sp = cascade.to_ctrl_split(init_x)
    x_train = []
    y_train = []

    info = None
    for i in range(cascade.n_stage):
        if i == 0:
            x_train.append(init_x_sp[0])
            tmp_out = [cascade.bench[0].eval_f(xi) for xi in init_x_sp[0]]
            tmp_y = [item[0] for item in tmp_out]
            info = [item[1] for item in tmp_out]
            y_train.append(np.array(tmp_y).reshape(n_init, -1))
        else:
            concat_x = np.hstack([y_train[i - 1], init_x_sp[i]])
            x_train.append(concat_x)
            if info is not None:
                tmp_y = [cascade.bench[i].eval_f(xi, item) for xi, item in zip(concat_x, info)]
            else:
                tmp_y = [cascade.bench[i].eval_f(xi.reshape(1, -1)) for xi in concat_x]
            info = None
            y_train.append(np.array(tmp_y).reshape(n_init, -1))

    init_f_max = np.max(y_train[-1])
    
    
    gp_list = []
    outputscale_bounds = [[(0.04, 2), (0.01, 2), (0.04, 2), (0.01, 2)],
                          [(0.04, 2), (0.01, 2), (0.04, 2), (0.01, 2)],
                          
                          [(0.25, 25)]]
    lengthscale_bounds = [[((b[1] - b[0]) / 20., (b[1] - b[0]) / 1) for b in f1.bounds],
                          [((b[1] - b[0]) / 20., (b[1] - b[0]) / 1) for b in f2.bounds],
                          [((b[1] - b[0]) / 20., (b[1] - b[0]) / 1) for b in f3.bounds]]
    for i in range(cascade.n_stage):
        xt = util.to_float_tensor(x_train[i])
        yt = util.to_float_tensor(y_train[i])
        ob = outputscale_bounds[i]
        lb = lengthscale_bounds[i]
        gp_i = IndependentMOGP(xt, yt, noise_var=noise_scale ** 2, use_ard=True,
                               lengthscale_bounds=lb, outputscale_bounds=ob,
                               prior_mean=None)
        gp_list.append(gp_i)
    gp = CascadeMOGP(gp_list)
    gp.fit_hyper_param(max_retries=20)
    bo = RealMultioutCascadeBO(cascade=cascade, model=gp, method=method, seed=seed, maximize=True,
                               stock_lifetime=stock_lifetime, f_best=init_f_max, n_thread=n_thread,
                               discard_strategy=discard_strategy, max_stocks=max_stocks,
                               discard_stocks_interval=discard_interval,
                               ucb_root_beta=root_beta, lipschitz=lipschitz,
                               eval_inference_r=True,
                               f_opt=25.)

    print(f'--- init ---')
    print(f'{n_init=}, {max_cycle=}')
    print(f'SolarCell 3stage, HP: optimize (mll)\n')

    max_cost = max_cycle * np.sum(each_cost)
    while bo.current_cost < max_cost:
        
        if profile_memory:
            memory_profiler.profile(bo.observe_next, stream=log_stream)(fit_model=True, enable_memory_profile=True,
                                                                        log_stream=log_stream)
        else:
            bo.observe_next(fit_model=True)
        if bo.iteration % 10 == 0:
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
    one_exp(seed=args.seed, method=args.method,
            max_cycle=args.max_cycle, n_thread=args.thread, n_init=args.n_init, cost_policy=args.cost,
            stock_lifetime=args.lifetime, discard_strategy=args.discard_method, max_stocks=args.max_stock,
            discard_interval=args.discard_interval, root_beta=args.root_beta, lipschitz=args.lipschitz,
            print_to_file=(not args.debug), profile_memory=args.mprof)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, help='random seed for numpy & pytorch')
    parser.add_argument('method', type=str, help='sampling method, nearly equal to acquisition function')
    parser.add_argument('cost', type=str, help='cost setting policy')
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
    parser.add_argument('--root_beta', type=int, default=3,
                        help='beta^{1/2} of ucb & lcb. used for inference regret, cUCB, stock reduction.')
    parser.add_argument('--lipschitz', type=float, default=0.1,
                        help='lipschitz constant. used for inference regret, cUCB, stock reduction.')

    args = parser.parse_args()

    root_dir = pathlib.Path(__file__).parent.parent.resolve()
    result_dir = root_dir / f'solarcell3s_noise_m2_cycle{args.max_cycle}'
    result_dir.mkdir(exist_ok=True)
    np.set_printoptions(linewidth=150)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    benchmark.set_n_cpu(1)
    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type(torch.DoubleTensor)
    with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False,
                                             solves=False), gpytorch.settings.max_cholesky_size(
        2000), gpytorch.settings.cholesky_jitter(double=1e-6):
        main()
