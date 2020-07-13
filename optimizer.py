'''
Optimizers should take the arguments
    grads, variables, learning_rate, grad_scale, max_grad_norm, and **kwargs.

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import blocksparse as bs
from mpi_utils import mpi_rank


def get_optimizer(name):
    mapping = {
        'bs_adafactor': bs_adafactor,
        'bs_adam': bs_adam,
    }
    return mapping[name]


def bs_adafactor(grads, variables, learning_rate, grad_scale=1.0,
                 beta2=0.999, max_grad_norm=1.0, norm_scale=1.0,
                 static_loss_scaling=False, **kwargs):

    # set to large value to disable clipping, but still collect global norm
    # we also use this for dynamic loss scaling
    if not max_grad_norm:
        max_grad_norm = 9e9

    fp16_args = dict(saturate=65504.0,
                     zero_nans=True) if static_loss_scaling else dict()

    global_norm, norm_scale = bs.clip_by_global_norm(grads,
                                                     grad_scale=grad_scale,
                                                     clip_norm=max_grad_norm,
                                                     **fp16_args)

    # use Adam for gains/biases
    adam = bs.AdamOptimizer(
        learning_rate=learning_rate,
        beta2=beta2,
        norm_scale=norm_scale,
        grad_scale=grad_scale,
        zero_init_variables=mpi_rank() != 0, **fp16_args)

    fact = bs.AdafactorOptimizer(
        learning_rate=learning_rate,
        beta2=beta2,
        norm_scale=norm_scale,
        grad_scale=grad_scale,
        zero_init_variables=mpi_rank() != 0, **fp16_args)

    adam_pairs = list()
    fact_pairs = list()
    for g, v in zip(grads, variables):
        if len(v.shape) < 2:
            adam_pairs.append((g, v))
        else:
            fact_pairs.append((g, v))

    adam = adam.apply_gradients(adam_pairs)
    fact = fact.apply_gradients(fact_pairs)

    return tf.group(adam, fact), global_norm


def bs_adam(grads, variables, learning_rate, beta2=0.999,
            grad_scale=1.0, max_grad_norm=1.0,
            fp16_mean_var=True, static_loss_scaling=False, **kwargs):

    # set to large value to disable clipping, but still collect global norm
    # we also use this for dynamic loss scaling
    if not max_grad_norm:
        max_grad_norm = 9e9

    if static_loss_scaling:
        global_norm, norm_scale = bs.clip_by_global_norm(grads,
                                                         grad_scale=grad_scale,
                                                         clip_norm=max_grad_norm,
                                                         saturate=65504.0,
                                                         zero_nans=True)
    else:
        # We first calculate whether its nan, then also clip.
        global_norm, _ = bs.clip_by_global_norm(grads,
                                                grad_scale=grad_scale,
                                                clip_norm=max_grad_norm)
        # Try zeroing infs.
        grads = [bs.filter_tensor(g, zero_infs=True, zero_nans=True) for g in grads]
        _, norm_scale = bs.clip_by_global_norm(grads,
                                               grad_scale=grad_scale,
                                               clip_norm=max_grad_norm)

    adam = bs.AdamOptimizer(
        learning_rate=learning_rate,
        beta2=beta2,
        norm_scale=norm_scale,
        grad_scale=grad_scale,
        fp16=fp16_mean_var,
        zero_init_variables=mpi_rank() != 0,
        saturate=65504.0, zero_nans=True)

    return adam.apply_gradients(zip(grads, variables)), global_norm
