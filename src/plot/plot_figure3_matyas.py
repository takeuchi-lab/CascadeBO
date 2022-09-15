import copy
import dataclasses as dc
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d


def copy_figure(fig) -> plt.Figure:
    return pickle.loads(pickle.dumps(fig))


@dc.dataclass
class Method:
    prefix: str
    label: str
    color: str
    ls: str
    result: dict[str, list] = dc.field(default_factory=dict)


def get_result(method: Method, cost: str = 'same'):
    results = sorted(result_dir.glob(f'{method.prefix}_seed*_{cost}'))
    valid_results = []
    for each_dir in results:
        model_file = (each_dir / 'result.pkl')
        if not model_file.exists():
            continue
        valid_results.append(each_dir)

    n_results = len(valid_results)
    print(f'{method.prefix}, {cost}: {n_results=}')
    selected_results = valid_results.copy()
  
    result_file = []
    for item in selected_results:
        with open(item / 'result.pkl', 'rb') as reader:
            result_file.append(pickle.load(reader))
    return result_file


def get_stats(target, results, cost_plot):
    stats = []
    for item in results:
        if target == 'regret':
            value = item['s_regret']
        elif target == 'i_regret':
            value = item['i_regret']
        elif target == 'obs_best':
            value = item['obs_hist']
        else:
            raise ValueError
        value = value[:len(item['cost_hist'])]
        r_interp = interp1d(item['cost_hist'], value, kind='previous', fill_value='extrapolate')
        stats.append(r_interp(cost_plot))
    stats = np.array(stats)
    if target in {'regret', 'i_regret'} and mode == 'solarcell':
        stats = -stats + 25
    if target in {'regret', 'i_regret'} and mode == 'equivalent_circuit':
        stats = -stats + 25
    return stats.mean(axis=0), stats.std(axis=0)


def plot_regret(methods: list[Method], cost: str, std_coef=None, i_regret: bool = False):
    fig, ax = plt.subplots()
    each_cost = methods[0].result[cost][0]['each_cost'].copy()
    cost_plot = np.arange(plot_cycle * np.sum(each_cost) + 1)

    if i_regret:
        stat_name = 'i_regret'
        _title = 'Regret'
    else:
        stat_name = 'regret'
        _title = 'Simple regret'
    for method in methods:
        cp = cost_plot.copy()
        mean, std = get_stats(stat_name, method.result[cost], cost_plot)
        mean = mean.copy().flatten()
        std = std.copy().flatten()
        c = method.color
        ls = method.ls
        label = method.label
        n_res = len(method.result[cost])
        if std_coef is None:
            coef = np.sqrt(1 / n_res)
        else:
            coef = std_coef
        err = coef * std
        if cost == 'same':
            itv = np.arange(0, cp.size, n_stage)
            cp = np.arange(itv.size)
            mean = mean[itv]
            err = err[itv]
        ax.plot(cp, mean, lw=2, ls=ls, c=c, label=label, alpha=0.7)
        err_itv = np.linspace(0, cp.size - 1, num=11, dtype=int)
        ax.errorbar(cp[err_itv], mean[err_itv], yerr=err[err_itv],
                    ls='', lw=1, capsize=3, color=c, alpha=0.7)

    if legend_flag:
        ax.legend(fontsize=legend_fontsize, framealpha=0, ncol=ncol)

    xlabel = 'Iteration'
    if cost != 'same':
        xlabel = 'Total cost'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_title)
    ax.set_title(title)
    fig_log = copy_figure(fig)
    fig.tight_layout()

    fig_log.axes[0].set_yscale('log')
    fig_log.axes[0].grid(which='minor', ls='-')
    fig_log.tight_layout()
    return [fig_log]



def plot_sequential(cost_list: list, sus=bool):
    seq_methods = [
        Method('ei_seq_l1', 'EI-based', 'tab:green', '-'),
        Method('cb_upd_l1', 'CI-based', 'tab:purple', '-'),
        Method('eifn_l1', 'EI-FN', 'tab:pink', '-'),
        Method('cbo_wide_l1', 'CBO', 'tab:orange', '-'),
        Method('joint_l1', 'FB-EI', 'tab:red', '-'),
        Method('jointUCB_l1', 'FB-UCB', 'tab:brown', '-'),
        Method('random_l1', 'Random', 'tab:gray', '-'),
    ]

    main_methods = seq_methods

    remain_cost = cost_list.copy()
    remain_cost.remove('same')
    converted_cost_hist = dict()
    for item in seq_methods:
        item.result['same'] = get_result(item, 'same')
        for cost in remain_cost:
            item.result[cost] = copy.deepcopy(item.result['same'])
            for res in item.result[cost]:
                res['cost_hist'] = converted_cost_hist[cost]

    for cost in cost_list:
        save_dir = base_dir/'figure'
        save_dir.mkdir(exist_ok=True)
        pdf_path = save_dir / f'figure3_matyas.pdf'

        with PdfPages(pdf_path) as pdf:  
            figs = plot_regret(main_methods, cost)
            for item in figs:
                pdf.savefig(item)
    return


if __name__ == '__main__':
    seed_pattern = re.compile(r'(?<=seed)[0-9]+(?=_)')

    sns.set_style('darkgrid')
    plt.rcParams['font.size'] = 16
    plt.rcParams['figure.figsize'] = (5., 4.5)
    plt.rcParams['figure.dpi'] = 70
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'

    plt.rcParams['pdf.fonttype'] = 42 
    plt.rcParams['ps.fonttype'] = 42 

    plt.rcParams['legend.borderaxespad'] = 0.15
    plt.rcParams['legend.borderpad'] = 0.2
    plt.rcParams['legend.columnspacing'] = 0.5
    plt.rcParams["legend.handletextpad"] = 0.5
    plt.rcParams['legend.handlelength'] = 1.5
    plt.rcParams['legend.handleheight'] = 1.


    ncol = 2 
    legend_fontsize = 13

    plot_obs_best = False
    plot_i_regret = False


    max_cycle = 100  
    plot_cycle = max_cycle  
    mode='matyas'


    cost_setting = ['same']
    legend_flag = False
    sus_flag = False
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    n_stage = 3
    ctrl_dim = 1
    result_dir = base_dir / f'matyas_s{n_stage}d{ctrl_dim}_cycle{max_cycle}/'
    title = 'Matyas (N = 3)'

    plot_sequential(cost_setting, sus=sus_flag)
