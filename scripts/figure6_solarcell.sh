#!/bin/bash

n_cpu=1
if [ $# = 0 ]; then
  n_parallel=1
else
  n_parallel=$1
fi

max_cycle=50
n_init=20
cost="same"


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

make_args() {
  for seed in $(seq 0 19); do

    reuse=1
    echo "${seed}" ei "${cost}" -t "${n_cpu}" -m ${max_cycle} -l ${reuse} --n_init=${n_init} -d=save_all

    reuse=1000
    echo "${seed}" ei "${cost}" -t "${n_cpu}" -m ${max_cycle} -l ${reuse} --n_init=${n_init} -d=save_all
  done
}

make_args | xargs -t -IXXX -P "${n_parallel}" bash -c "python -u src/solarcell3s_exp.py XXX"

python src/plot/plot_figure6_solarcell.py