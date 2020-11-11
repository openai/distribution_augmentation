# Distribution Augmentation for Generative Modeling

This is the code repository for `Distribution Augmentation for Generative Modeling`, presented at ICML 2020.  

Paper link: https://proceedings.icml.cc/static/paper_files/icml/2020/6095-Paper.pdf

Unconditional samples from our autoregressive CIFAR-10 model. (150m params, t=0.94)
![Samples from our model](https://github.com/openai/distribution_augmentation/blob/master/samples.png?raw=true)


# Setup
This code is tested on Python 3.7.3, Ubuntu 16.04, Anaconda 4.7.11, Tensorflow version 1.13.1, and CUDA 10. It requires V100 GPUs.

It requires installing `blocksparse` from source:

```bash
git clone git@github.com:openai/blocksparse.git
cd blocksparse
git checkout 89074c5ccf78e3a88b4aa2aefc9e208d4773dcbc  # master at time of release
make compile
pip install dist/*.whl
```


# Running experiments
Hyperparameters for experiments live in `hyperparams.py`. They can be selected in a training run using the `--hps [name]` flag. For instance:

1 gpu runs can be run as so:
`CUDA_VISIBLE_DEVICES=0 python train.py --hps c10_small`

8 gpu runs can be run as so:
`mpiexec -n 8 python train.py --hps c10_150m_baseline`

64 gpu runs depend on the specifics of your network and system. We use the `-f` flag with a hostfile, but see options in `mpiexec` for whatever works for you.

If you run imagenet64 or imagenet32, you need to download the datasets. See the corresponding class in `datasets.py` to see how to do that. We dumped copies of the datasets in Azure buckets for convenience.

The specific code for experiments we reported in the paper live in the `experiments/` folder.

# Reloading, evaluating, and sampling from saved models

We also stored some model weights in storage buckets, which can be loaded with this codebase to generate samples, run evaluations, or to run a custom training procedure.

For instance, to generate samples from a trained 15m parameter model, you can run the following:

```
wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/models/c10-15m-baseline.npz
python train.py --desc samples-c10-15m-baseline --hps c10_15m --restore_path c10-15m-baseline.npz --sample_and_exit --samples_to_generate 4 --sample_batch 4
```

This will dump samples in the model directory.

Other examples of loading from saved models can be found in `eval.sh` or `sample.sh`, where we include saved versions of various models reported in the paper.

# Citation
If you find this work useful in your research, consider citing us with the following bibtex entry:
```
@incollection{icml2020_6095,
 abstract = {We present distribution augmentation (DistAug), a simple and powerful method of regularizing generative models. Core to our approach is applying augmentation functions to data and then conditioning the generative model on the specific function used. Unlike typical data augmentation, distribution augmentation allows usage of functions which modify the target density, enabling aggressive augmentations more commonly seen in supervised and self-supervised learning. We demonstrate this is a more effective regularizer than standard methods, and use it to train a 150M parameter autoregressive model on CIFAR-10 to 2.56 bits per dim (relative to the state-of-the-art 2.80). Samples from this model attain FID 12.75 and IS 8.40, outperforming the majority of GANs. We further demonstrate the technique is broadly applicable across model architectures, objectives, and problem domains.},
 author = {Jun, Heewoo and Child, Rewon and Chen, Mark and Schulman, John and Ramesh, Aditya and Radford, Alec and Sutskever, Ilya},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {10563--10576},
 title = {Distribution Augmentation for Generative Modeling},
 year = {2020}
}
```
