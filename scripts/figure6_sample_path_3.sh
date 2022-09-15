#!/bin/bash

n_cpu=1
if [ $# = 0 ]; then
  n_parallel=1
else
  n_parallel=$1
fi

max_cycle=100
setting='s3d2'
n_init=10
reuse=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

make_args() {
  for gpseed in $(seq 0 9); do
    for seed in $(seq 0 1); do
      cost="same"
      echo "${setting}" "${seed}" ei "${cost}" -g "${gpseed}" -t "${n_cpu}" -m ${max_cycle} -l ${reuse} --n_init=${n_init} -d=save_all --length_scale=3
      echo "${setting}" "${seed}" ei "${cost}" -g "${gpseed}" -t "${n_cpu}" -m ${max_cycle} -l ${reuse} --n_init=${n_init} -d=ci --discard_interval=1 --root_beta=2 --length_scale=3

      cost="cons"
      echo "${setting}" "${seed}" ei "${cost}" -g "${gpseed}" -t "${n_cpu}" -m ${max_cycle} -l ${reuse} --n_init=${n_init} -d=save_all --length_scale=3
      echo "${setting}" "${seed}" ei "${cost}" -g "${gpseed}" -t "${n_cpu}" -m ${max_cycle} -l ${reuse} --n_init=${n_init} -d=ci --discard_interval=1 --root_beta=2 --length_scale=3
    done
  done
}

make_args | xargs -t -IXXX -P "${n_parallel}" bash -c "python -u src/sample_path_exp.py XXX"
python src/plot/plot_figure6_sample_path_3.py
