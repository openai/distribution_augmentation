#!/bin/bash

mkdir -p logs

# Baseline
CUDA_VISIBLE_DEVICES=0 python train.py --seed 43 --aug_seed 314 --desc c10-small --hps c10_small --resid_pdrop 0.005 &> logs/c10-small.out &

# Horizontal flips
CUDA_VISIBLE_DEVICES=3 python train.py --seed 43 --aug_seed 314 --desc c10-small-lr-rp05 --hps c10_small_lr --resid_pdrop 0.05 &> logs/c10-small-lr-rp05.out &
CUDA_VISIBLE_DEVICES=1 python train.py --seed 43 --aug_seed 314 --desc c10-small-lr-rp10 --hps c10_small_lr --resid_pdrop 0.10 &> logs/c10-small-lr-rp10.out &
CUDA_VISIBLE_DEVICES=2 python train.py --seed 43 --aug_seed 314 --desc c10-small-lr-rp20 --hps c10_small_lr --resid_pdrop 0.20 &> logs/c10-small-lr-rp20.out &

# Rand-augment
CUDA_VISIBLE_DEVICES=4 python train.py --seed 43 --aug_seed 314 --desc c10-small-ra-n2-m3-nocond --hps c10_small_ra_n2_m3 --resid_pdrop 0.005 &> logs/c10-small-ra-n2-m3-nocond.out &
CUDA_VISIBLE_DEVICES=5 python train.py --seed 43 --aug_seed 314 --desc c10-small-ra-n2-m3-cond --hps c10_small_ra_n2_m3 --resid_pdrop 0.005 --rand_augment_conditioning &> logs/c10-small-ra-n2-m3-cond.out &
CUDA_VISIBLE_DEVICES=6 python train.py --seed 43 --aug_seed 314 --desc c10-small-ra-n1-m2-nocond --hps c10_small_ra_n1_m2 --resid_pdrop 0.005 &> logs/c10-small-ra-n1-m2-nocond.out &
CUDA_VISIBLE_DEVICES=7 python train.py --seed 43 --aug_seed 314 --desc c10-small-ra-n1-m2-cond --hps c10_small_ra_n1_m2 --resid_pdrop 0.005 --rand_augment_conditioning &> logs/c10-small-ra-n1-m2-cond.out &
