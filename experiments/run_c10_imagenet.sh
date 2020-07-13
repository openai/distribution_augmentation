#!/bin/bash

mkdir -p logs

function multiply() {
  echo $(( $1 * $2 ))
}

# Baseline
CUDA_VISIBLE_DEVICES=0 python train.py --seed 43 --aug_seed 314 --desc c10-small-rp10 --hps c10_small --resid_pdrop 0.10 &> logs/c10-small-rp10.out &
CUDA_VISIBLE_DEVICES=1 python train.py --seed 43 --aug_seed 314 --desc c10-small-rp25 --hps c10_small --resid_pdrop 0.25 &> logs/c10-small-rp25.out &

# without prompting

# equivalent to rot/tr/c => 4 x 2 x 6 = 48. 1 / 48 ~= 0.02.
CUDA_VISIBLE_DEVICES=2 python train.py --seed 43 --aug_seed 314 --desc c10-small-i32-f98-nocond --hps c10_small_ss_i32_nocond --auxiliary_dataset_fraction 0.98 --auxiliary_dataset_subset_size `multiply 48000 47` --resid_pdrop 0.005 &> logs/c10-small-i32-f98-nocond.out &
# equivalent to rot/tr
CUDA_VISIBLE_DEVICES=3 python train.py --seed 43 --aug_seed 314 --desc c10-small-i32-f875-nocond --hps c10_small_ss_i32_nocond --auxiliary_dataset_fraction 0.875 --auxiliary_dataset_subset_size `multiply 48000 7` --resid_pdrop 0.005 &> logs/c10-small-i32-f875-nocond.out &
# equivalent to rot
CUDA_VISIBLE_DEVICES=4 python train.py --seed 43 --aug_seed 314 --desc c10-small-i32-f75-nocond --hps c10_small_ss_i32_nocond --auxiliary_dataset_fraction 0.75 --auxiliary_dataset_subset_size `multiply 48000 3` --resid_pdrop 0.005 &> logs/c10-small-i32-f75-nocond.out &

# with prompting
CUDA_VISIBLE_DEVICES=5 python train.py --seed 43 --aug_seed 314 --desc c10-small-i32-f98-cond --hps c10_small_ss_i32_cond --auxiliary_dataset_fraction 0.98 --auxiliary_dataset_subset_size `multiply 48000 47` --resid_pdrop 0.005 &> logs/c10-small-i32-f98-cond.out &
CUDA_VISIBLE_DEVICES=6 python train.py --seed 43 --aug_seed 314 --desc c10-small-i32-f875-cond --hps c10_small_ss_i32_cond --auxiliary_dataset_fraction 0.875 --auxiliary_dataset_subset_size `multiply 48000 7` --resid_pdrop 0.005 &> logs/c10-small-i32-f875-cond.out &
CUDA_VISIBLE_DEVICES=7 python train.py --seed 43 --aug_seed 314 --desc c10-small-i32-f75-cond --hps c10_small_ss_i32_cond --auxiliary_dataset_fraction 0.75 --auxiliary_dataset_subset_size `multiply 48000 3` --resid_pdrop 0.005 &> logs/c10-small-i32-f75-cond.out &
