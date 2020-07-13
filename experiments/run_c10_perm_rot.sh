#!/bin/bash

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python train.py --seed 43 --aug_seed 314 --desc c10-small-rot-nocond-nope --hps c10_small_rot --no_permute_embeddings --resid_pdrop 0.005 --use_unconditional_augmentation &> logs/c10-small-rot-nocond-nope.out &

CUDA_VISIBLE_DEVICES=1 python train.py --seed 43 --aug_seed 314 --desc c10-small-rot-nocond-pe --hps c10_small_rot --permute_embeddings --resid_pdrop 0.005 --use_unconditional_augmentation &> logs/c10-small-rot-nocond-pe.out &

CUDA_VISIBLE_DEVICES=2 python train.py --seed 43 --aug_seed 314 --desc c10-small-rot-cond-nope --hps c10_small_rot --no_permute_embeddings --resid_pdrop 0.005 &> logs/c10-small-rot-cond-nope.out &

CUDA_VISIBLE_DEVICES=3 python train.py --seed 43 --aug_seed 314 --desc c10-small-rot-cond-pe --hps c10_small_rot --permute_embeddings --resid_pdrop 0.005 &> logs/c10-small-rot-cond-pe.out &
