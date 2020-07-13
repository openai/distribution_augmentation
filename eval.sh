#!/bin/bash

for dataset in cifar10; do
for partition in test; do
  args="--eval_test --no_sample_during_eval --dataset $dataset"

  # c10 15m runs
  python train.py --desc c10-15m-baseline-${partition} --hps c10_15m --restore_path gs://openai-distribution-augmentation-assets/models/c10-15m-baseline.npz $args
  python train.py --desc c10-15m-rot-${partition} --hps c10_15m_rot --restore_path gs://openai-distribution-augmentation-assets/models/c10-15m-rot-dist-aug.npz $args
  python train.py --desc c10-15m-rot-tr-${partition} --hps c10_15m_rot_tr --restore_path gs://openai-distribution-augmentation-assets/models/c10-15m-rot-tr-dist-aug.npz $args

  # c10 58m runs
  python train.py --desc c10-58m-baseline-${partition} --hps c10-58m --restore_path gs://openai-distribution-augmentation-assets/models/c10-58m-baseline.npz $args
  python train.py --desc c10-58m-rot-${partition} --hps c10-58m-rot --restore_path gs://openai-distribution-augmentation-assets/models/c10-58m-rot-dist-aug.npz $args

  # c10 150m runs
  python train.py --desc c10-150m-baseline-${partition} --hps c10_150m_baseline --restore_path gs://openai-distribution-augmentation-assets/models/c10-150m-baseline.npz $args

  # randaugment runs
  python train.py --desc c10-150m-randaugment-dataaug-${partition} --hps c10_150m_randaugment_dataaug --restore_path gs://openai-distribution-augmentation-assets/models/c10-150m-randaugment-data-aug.npz $args
  python train.py --desc c10-150m-randaugment-distaug-${partition} --hps c10_150m_randaugment_distaug --restore_path gs://openai-distribution-augmentation-assets/models/c10-150m-randaugment-dist-aug.npz $args

  # permutation runs
  python train.py --desc c10-150m-rot-${partition} --hps c10-150m-rot --restore_path gs://openai-distribution-augmentation-assets/models/c10-150m-rot-dist-aug.npz $args
  python train.py --desc c10-150m-rot-tr-${partition} --hps c10-150m-rot-tr --restore_path gs://openai-distribution-augmentation-assets/models/c10-150m-rot-tr-dist-aug.npz $args
  python train.py --desc c10-150m-rot-c-tr-js-${partition} --hps c10-150m-rot-c-tr-js --restore_path gs://openai-distribution-augmentation-assets/models/c10-150m-rot-tr-c-js-dist-aug.npz $args
done
done

for dataset in imagenet64; do
for partition in test; do
  # i64 150m
  python train.py --desc i64-150m-baseline-${partition} --hps i64_150m_32gpu --restore_path gs://openai-distribution-augmentation-assets/models/i64-150m-baseline.npz $args
  python train.py --desc i64-150m-32gpu-rot-${partition} --hps i64_150m_32gpu_rot_32gpu --restore_path gs://openai-distribution-augmentation-assets/models/i64-150m-rot-dist-aug.npz $args
  python train.py --desc i64-150m-32gpu-rot-tr-${partition} --hps i64_150m_32gpu_rot_tr_32gpu --restore_path gs://openai-distribution-augmentation-assets/models/i64-150m-rot-tr-dist-aug.npz $args

  # i64 300m
  python train.py --desc i64-300m-baseline-${dataset}-converged-${partition} --hps i64_300m_64gpu --restore_path gs://openai-distribution-augmentation-assets/models/i64-300m-baseline.npz $args
  python train.py --desc i64-300m-rot-${dataset}-converged-${partition} --hps i64_300m_64gpu_rot --restore_path gs://openai-distribution-augmentation-assets/models/i64-300m-rot-dist-aug.npz $args
done
done
