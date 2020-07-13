#!/bin/bash

mkdir -p logs

CUDA_VISIBLE_DEVICES=4 python train.py --seed 43 --aug_seed 314 --desc c10-small-rot-tr-nocond-nope --hps c10_small_rot_tr --no_permute_embeddings --resid_pdrop 0.005 --use_unconditional_augmentation &> logs/c10-small-rot-tr-nocond-nope.out &

CUDA_VISIBLE_DEVICES=5 python train.py --seed 43 --aug_seed 314 --desc c10-small-rot-tr-nocond-pe --hps c10_small_rot_tr --permute_embeddings --resid_pdrop 0.005 --use_unconditional_augmentation &> logs/c10-small-rot-tr-nocond-pe.out &

CUDA_VISIBLE_DEVICES=6 python train.py --seed 43 --aug_seed 314 --desc c10-small-rot-tr-cond-nope --hps c10_small_rot_tr --no_permute_embeddings --resid_pdrop 0.005 &> logs/c10-small-rot-tr-cond-nope.out &

CUDA_VISIBLE_DEVICES=7 python train.py --seed 43 --aug_seed 314 --desc c10-small-rot-tr-cond-pe --hps c10_small_rot_tr --permute_embeddings --resid_pdrop 0.005 &> logs/c10-small-rot-tr-cond-pe.out &
