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

args="--sample_and_exit --samples_to_generate 4 --sample_batch 4"

# c10 15m runs
python train.py --desc samples-c10-15m-baseline --hps c10_15m --restore_path c10-15m-baseline.npz $args
python train.py --desc samples-c10-15m-rot --hps c10_15m_rot --restore_path c10-15m-rot-dist-aug.npz $args
python train.py --desc samples-c10-15m-rot-tr --hps c10_15m_rot_tr --restore_path c10-15m-rot-tr-dist-aug.npz $args

# c10 58m runs
python train.py --desc samples-c10-58m-baseline --hps c10-58m --restore_path c10-58m-baseline.npz $args
python train.py --desc samples-c10-58m-rot --hps c10-58m-rot --restore_path c10-58m-rot-dist-aug.npz $args

# c10 150m baseline and randaugment runs
python train.py --desc samples-c10-150m-baseline --hps c10_150m_baseline --restore_path c10-150m-baseline.npz $args
python train.py --desc samples-c10-150m-randaugment-dataaug --hps c10_150m_randaugment_dataaug --restore_path c10-150m-randaugment-data-aug.npz $args
python train.py --desc samples-c10-150m-randaugment-distaug --hps c10_150m_randaugment_distaug --restore_path c10-150m-randaugment-dist-aug.npz $args

# c10 150m perm runs
python train.py --desc samples-c10-150m-rot --hps c10-150m-rot --restore_path c10-150m-rot-dist-aug.npz $args
python train.py --desc samples-c10-150m-rot-tr --hps c10-150m-rot-tr --restore_path c10-150m-rot-tr-dist-aug.npz $args
python train.py --desc samples-c10-150m-rot-c-tr-js --hps c10-150m-rot-c-tr-js --restore_path c10-150m-rot-tr-c-js-dist-aug.npz $args
