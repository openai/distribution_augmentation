#!/bin/bash

wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-15m-baseline.npz
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-15m-rot-dist-aug.npz
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-15m-rot-tr-dist-aug.npz
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-58m-baseline.npz
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-58m-rot-dist-aug.npz
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-150m-baseline.npz
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-150m-randaugment-data-aug.npz
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-150m-randaugment-dist-aug.npz
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-150m-rot-dist-aug.npz
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-150m-rot-tr-dist-aug.npz
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-150m-rot-tr-c-js-dist-aug.npz

for dataset in cifar10; do
for partition in test; do
  args="--eval_test --no_sample_during_eval --dataset $dataset"

  # c10 15m runs
  python train.py --desc c10-15m-baseline-${partition} --hps c10_15m --restore_path c10-15m-baseline.npz $args
  python train.py --desc c10-15m-rot-${partition} --hps c10_15m_rot --restore_path c10-15m-rot-dist-aug.npz $args
  python train.py --desc c10-15m-rot-tr-${partition} --hps c10_15m_rot_tr --restore_path c10-15m-rot-tr-dist-aug.npz $args

  # c10 58m runs
  python train.py --desc c10-58m-baseline-${partition} --hps c10-58m --restore_path c10-58m-baseline.npz $args
  python train.py --desc c10-58m-rot-${partition} --hps c10-58m-rot --restore_path c10-58m-rot-dist-aug.npz $args

  # c10 150m runs
  python train.py --desc c10-150m-baseline-${partition} --hps c10_150m_baseline --restore_path c10-150m-baseline.npz $args

  # randaugment runs
  python train.py --desc c10-150m-randaugment-dataaug-${partition} --hps c10_150m_randaugment_dataaug --restore_path c10-150m-randaugment-data-aug.npz $args
  python train.py --desc c10-150m-randaugment-distaug-${partition} --hps c10_150m_randaugment_distaug --restore_path c10-150m-randaugment-dist-aug.npz $args

  # permutation runs
  python train.py --desc c10-150m-rot-${partition} --hps c10-150m-rot --restore_path c10-150m-rot-dist-aug.npz $args
  python train.py --desc c10-150m-rot-tr-${partition} --hps c10-150m-rot-tr --restore_path c10-150m-rot-tr-dist-aug.npz $args
  python train.py --desc c10-150m-rot-c-tr-js-${partition} --hps c10-150m-rot-c-tr-js --restore_path c10-150m-rot-tr-c-js-dist-aug.npz $args
done
done
