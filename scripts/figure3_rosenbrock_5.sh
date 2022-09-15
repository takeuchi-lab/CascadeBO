#!/bin/bash

n_cpu=1
if [ $# = 0 ]; then
  n_parallel=1
else
  n_parallel=$1
fi

declare -a methods=('cb_upd' 'ei_seq' 'random' 'joint' 'jointUCB' 'cbo_wide' 'eifn')

func='rosenbrock'
max_cycle=100
setting='s5d2'
n_init=20
reuse=1
cost="same"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

make_args() {
  for seed in $(seq 0 19); do
    for method in "${methods[@]}"; do
      echo "${setting}" "${seed}" "${method}" "${cost}" -f "${func}" -t "${n_cpu}" -m ${max_cycle} -l ${reuse} --n_init=${n_init} -d=save_all --root_beta=2
    done
  done
}

make_args | xargs -t -IXXX -P "${n_parallel}" bash -c "python -u src/synth_exp.py XXX"
python src/plot/plot_figure3_rosenbrock_5.py
