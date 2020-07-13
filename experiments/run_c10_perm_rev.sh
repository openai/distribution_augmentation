#!/bin/bash

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python train.py --seed 43 --aug_seed 314 --desc c10-small-rev-nocond-nope --hps c10_small_rev --no_permute_embeddings --resid_pdrop 0.005 --use_unconditional_augmentation &> logs/c10-small-rev-nocond-nope.out &

CUDA_VISIBLE_DEVICES=1 python train.py --seed 43 --aug_seed 314 --desc c10-small-rev-nocond-pe --hps c10_small_rev --permute_embeddings --resid_pdrop 0.005 --use_unconditional_augmentation &> logs/c10-small-rev-nocond-pe.out &

CUDA_VISIBLE_DEVICES=2 python train.py --seed 43 --aug_seed 314 --desc c10-small-rev-cond-nope --hps c10_small_rev --no_permute_embeddings --resid_pdrop 0.005 &> logs/c10-small-rev-cond-nope.out &

CUDA_VISIBLE_DEVICES=3 python train.py --seed 43 --aug_seed 314 --desc c10-small-rev-cond-pe --hps c10_small_rev --permute_embeddings --resid_pdrop 0.005 &> logs/c10-small-rev-cond-pe.out &
