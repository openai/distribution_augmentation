# the base model is the optimized version of the Sparse Transformer
# presented at https://arxiv.org/abs/1904.10509
# if hacking on the model, be sure to use the mpi init functions
# (random_or_zeros_init, constant_or_zeros_init, etc) or
# else the models wont be synced across ranks
from collections import namedtuple
import itertools
import os
import pdb
import sys
import time
import math
import argparse
import numpy as np
import tensorflow as tf
import blocksparse as bs
from blocksparse.nccl import serialize_allreduce_ops
import subprocess
from utils import logger, save_params, load_variables_from_file
from utils import maybe_download
from utils import log_gradient_values, shape_list, go_over
from hyperparams import Hyperparams, add_arguments
from hyperparams import parse_args_and_update_hparams
from mpi_utils import random_or_zeros_init, constant_or_zeros_init, zeros_init
from mpi_utils import get_session, allreduce, group_allreduce, sync_variables
from mpi_utils import mpi_size, mpi_rank, local_mpi_rank, mpi_allgather, mpi_barrier
from optimizer import get_optimizer
from datasets import get_dataset, iter_data_mpi, JankySubsampledDataset
from autoaugment import distort_image_with_randaugment


H = Hyperparams()


AugmentationType = namedtuple("AugmentationType", ("sos_name", "description", "num_tokens", "is_used", "fn"))


def f32_storage_getter(getter, name, shape=None, dtype=tf.float32,
                       initializer=None, regularizer=None,
                       trainable=True, *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/
    index.html#mptrain
    """
    var = H.var_cache.get(name)
    if var is None:
        with tf.control_dependencies(None):
            var = getter(name, shape, dtype=tf.float32,
                         initializer=initializer,
                         regularizer=regularizer,
                         trainable=trainable,
                         *args, **kwargs)
        H.var_cache[name] = var

    if H.ema is not None:
        var = H.ema.average(var)

    if dtype != var.dtype.base_dtype:
        var = bs.float_cast(var, dtype=dtype, dx_dtype=dtype, name=f"{name}/cast")

    return var


def split_states(x, heads):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [heads, m // heads]
    return tf.reshape(x, new_x_shape)


def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


def split_heads(scope, x, heads):
    """
    (batch, pixel, state) -> (batch, head, pixel, head_state)
    """
    with tf.name_scope(scope):
        return bs.transpose_0213(split_states(x, heads))


def merge_heads(scope, x):
    """
    (batch, head, pixel, head_state) -> (batch, pixel, state)
    """
    with tf.name_scope(scope):
        return merge_states(bs.transpose_0213(x))


def get_dense_attn_mask(n, attn_mode):
    '''a is dense attention, b is local attention (previous k),
    bT is strided (every kth element), implemented as a transpose'''
    key = f'{n}-{attn_mode}'
    dense_mask = H.dense_mask_cache.get(key)
    if dense_mask is not None:
        return dense_mask

    if attn_mode == 'a_all':
        b = tf.ones([n, n], dtype=tf.float32)
    elif attn_mode == 'a':
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    elif attn_mode == 'b':
        bandwidth = H.local_attn_ctx
        ctx = tf.minimum(n - 1, bandwidth - 1)
        b = tf.matrix_band_part(tf.ones([n, n]), ctx, 0)
    elif attn_mode in ['c', 'bT']:
        stride = H.local_attn_ctx
        x = tf.reshape(tf.range(n, dtype=tf.int32), [n, 1])
        y = tf.transpose(x)
        z = tf.zeros([n, n], dtype=tf.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = tf.equal(tf.floormod(q - k, stride), 0)
        c3 = tf.logical_and(c1, c2)
        b = tf.cast(c3, tf.float32)
    else:
        raise ValueError('Not yet implemented')
    b = tf.reshape(b, [1, 1, n, n])

    H.dense_mask_cache[key] = b

    return b


def get_callback(attn_mode):
    def cb(blk_shape, head_idx, qry_idx, key_idx, blk_idx):
        mask = np.ones(blk_shape, dtype=np.bool)
        qdim, kdim = blk_shape
        assert qdim == kdim
        if attn_mode in ['a_all', 'b_all', 'bT_all']:
            return mask
        if qry_idx == key_idx:
            for q in range(qdim):
                mask[q, q + 1:] = 0
        if attn_mode in ['a', 'bT', 'b0']:
            return mask
        if attn_mode == 'b':
            bandwidth = H.local_attn_ctx
            # convert group indices to absolute indices and mask
            # according to that
            q_pos = blk_shape[0] * qry_idx
            k_pos = blk_shape[1] * key_idx
            for q in range(qdim):
                q_ = q + q_pos
                maxw = max(-1, q_ - k_pos - bandwidth)
                mask[q, :maxw + 1] = 0
                if qry_idx == key_idx:
                    mask[q, q + 1:] = 0
            if H.print_attn_layout:
                for i in range(qdim):
                    print(' '.join([str(x) for x in mask[i, 0:kdim].astype(np.int32)]))
                print(qry_idx, key_idx)
                pdb.set_trace()
            return mask
        raise ValueError
    return cb


def get_blocksparse_obj(n_ctx, n_heads, attn_mode):
    '''a is dense attention, b is local attention (previous k),
    bT is strided (every kth element), implemented as a transpose'''
    key = f'{n_ctx}-{n_heads}-{attn_mode}'
    bst = H.bst_cache.get(key)
    if bst is not None:
        return bst

    blocksize = H.blocksize
    n_bctx = n_ctx // blocksize

    if attn_mode in ['b', 'bT', 'b0']:
        if attn_mode in ['b']:
            assert H.local_attn_ctx % blocksize == 0
            extra_diagonals = H.local_attn_ctx // blocksize
        elif attn_mode in ['bT', 'b0']:
            bT_ctx = H.attn_ctx // H.local_attn_ctx
            assert bT_ctx % blocksize == 0
            block_chunks = bT_ctx // blocksize
        layout = np.ones([n_bctx, n_bctx], dtype=np.bool)
        for q_idx in range(n_bctx):
            # Causal queries cannot attend to keys above them
            layout[q_idx, q_idx + 1:] = 0
            if attn_mode == 'b':
                start = max(0, q_idx - extra_diagonals)
                layout[q_idx, :start] = 0
            elif attn_mode in ['bT', 'b0']:
                offset = q_idx % block_chunks
                layout[q_idx, :q_idx - offset] = 0

    elif attn_mode == 'a':
        # standard causal attention
        layout = np.ones([n_bctx, n_bctx], dtype=np.bool)
        for q_idx in range(n_bctx):
            layout[q_idx, q_idx + 1:] = 0
    elif attn_mode == 'a_all':
        layout = np.ones([n_bctx, n_bctx], dtype=np.bool)
        if H.mem_block and H.block_memory:
            # Block attention over the memory block
            layout[:-1, -1] = 0
    elif attn_mode in ['b_all', 'bT_all']:
        assert H.blocksize == 32
        assert H.local_attn_ctx == 32
        assert n_bctx == 32
        layout = np.zeros([n_bctx, n_bctx], dtype=np.bool)
        for q_idx in range(n_bctx):
            layout[q_idx, q_idx] = 1.0
    else:
        raise NotImplementedError

    if H.print_attn_layout:
        width = H.attn_cols_to_print
        for i in range(min(width, n_bctx)):
            print(' '.join([str(x) for x in layout[i, 0:width].astype(np.int32)]))
        pdb.set_trace()

    bst = bs.BlocksparseTransformer(
        layout, block_size=blocksize,
        mask_callback=get_callback(attn_mode), heads=n_heads)

    H.bst_cache[key] = bst
    return bst


def linear(scope, x, nf, std, relu=False, fast_gelu=False):

    with tf.variable_scope(scope):

        nx = x.shape[-1].value

        # delay w casting operation just prior to use
        # This can save a lot of memory for large param models.
        with tf.control_dependencies([x]):
            w = tf.get_variable("w", [nx, nf], dtype=H.dtype,
                                initializer=random_or_zeros_init(stddev=std))
            b = tf.get_variable("b", [nf], dtype=tf.float32,
                                initializer=zeros_init())

        ndims = x.shape.ndims
        if ndims > 2:
            h_shape = tf.concat([tf.shape(x)[:ndims - 1], [nf]], axis=0)
            x = tf.reshape(x, [-1, nx])

        h = tf.matmul(x, w)
        h = bs.bias_relu(h, b, relu=relu, fast_gelu=fast_gelu)
        if ndims > 2:
            h = tf.reshape(h, h_shape)
    return h


def norm(scope, x, epsilon=1e-5):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        g = tf.get_variable("g", [nx], dtype=tf.float32,
                            initializer=constant_or_zeros_init(1.0))
        b = tf.get_variable("b", [nx], dtype=tf.float32,
                            initializer=zeros_init())
        return bs.layer_norm(x, g, b, axis=-1, epsilon=epsilon, relu=False)


def embedding_dropout(x, train):
    if train and H.embd_pdrop > 0.0:
        x, _ = bs.dropout(x, keep_prob=1.0 - H.embd_pdrop)
    return x


def residual_dropout(x, train, key, pdrop=None):
    resid_pdrop = pdrop if pdrop else H.resid_pdrop

    if train and resid_pdrop > 0.0:
        mask_shape = x.shape.as_list()
        key += str(mask_shape)
        mask_shape = None

        x, H.dropout_cache[key] = bs.dropout(
            x, keep_prob=1.0 - resid_pdrop,
            mask=H.dropout_cache.get(key),
            mask_shape=mask_shape)

    return x


@bs.recomputable
def dense_attention(x, n_heads, attn_mode, use_cache=False, train=False, pdrop=None):

    nx = x.shape[-1].value
    n_state = int(nx * H.qk_ratio)
    if n_state % n_heads != 0:
        raise ValueError('nx must be divisible by head state')

    h = norm("attn_input", x)

    qh = h[:, -1:, :] if use_cache else h

    q = linear('q_proj', qh, n_state, std=np.sqrt(H.qk_w / nx))
    k = linear('k_proj', h, n_state, std=np.sqrt(H.qk_w / nx))
    v = linear('v_proj', h, nx, std=np.sqrt(H.v_w / nx))

    q = split_heads("q_split", q, n_heads)
    k = split_heads("k_split", k, n_heads)
    v = split_heads("v_split", v, n_heads)

    if use_cache:
        if attn_mode not in ['a', 'b', 'c', 'bT']:
            raise NotImplementedError
        mask = None
        if attn_mode == 'b':
            k = k[:, :, -H.local_attn_ctx:, :]
            v = v[:, :, -H.local_attn_ctx:, :]
        elif attn_mode in ['c', 'bT']:
            k = k[:, :, ::-H.local_attn_ctx, :][:, :, ::-1, :]
            v = v[:, :, ::-H.local_attn_ctx, :][:, :, ::-1, :]
    else:
        n_timesteps = k.shape[2].value
        mask = get_dense_attn_mask(n_timesteps, attn_mode)
    if H.float16:
        # These products can overflow, so we do it in float32.
        k = bs.float_cast(k, dtype=tf.float32)
        q = bs.float_cast(q, dtype=tf.float32)
        v = bs.float_cast(v, dtype=tf.float32)
    w = tf.matmul(q, k, transpose_b=True)
    w = bs.masked_softmax(w, mask=mask, scale=1.0 / np.sqrt(q.shape[-1].value))
    a = tf.matmul(w, v)
    a = merge_heads("merge_attn", a)
    if H.float16:
        a = bs.float_cast(a, dtype=tf.float16)

    return post_attention(x, a, use_cache=use_cache, train=train, pdrop=pdrop)


@bs.recomputable
def sparse_attention(x, n_heads, attn_mode, use_cache=False, train=False, pdrop=None):

    if use_cache:
        raise NotImplementedError
    if not H.float16:
        raise ValueError("sparse_attention requires fp16")

    nx = x.shape[-1].value
    n_state = int(nx * H.qk_ratio)
    if n_state % n_heads != 0:
        raise ValueError('nx must be divisible by head state')

    h = norm("attn_input", x)

    if attn_mode in ['bT', 'bT_all']:
        ctx = H.local_attn_ctx
        bT_ctx = H.attn_ctx // ctx
        assert bT_ctx % H.blocksize == 0, f'{bT_ctx}, {H.blocksize}'
        n, t, embd = shape_list(h)
        h = tf.reshape(h, [n, bT_ctx, ctx, embd])
        h = bs.transpose_0213(h)
        h = tf.reshape(h, [n, t, embd])

    q = linear('q_proj', h, n_state, std=np.sqrt(H.qk_w / nx))
    k = linear('k_proj', h, n_state, std=np.sqrt(H.qk_w / nx))
    v = linear('v_proj', h, nx, std=np.sqrt(H.v_w / nx))

    bst = get_blocksparse_obj(H.attn_ctx, n_heads, attn_mode)

    w = bst.query_key_op(q, k)
    w = bst.masked_softmax(w, scale=1.0 / np.sqrt(n_state // n_heads))
    a = bst.weight_value_op(w, v)

    if attn_mode in ['bT', 'bT_all']:
        a = tf.reshape(a, [n, ctx, bT_ctx, embd])
        a = bs.transpose_0213(a)
        a = tf.reshape(a, [n, t, embd])

    return post_attention(x, a, train=train, pdrop=pdrop)


def post_attention(x, a, use_cache=None, train=False, pdrop=None):

    nx = x.shape[-1].value

    a = linear('post_proj', a, nx,
               std=np.sqrt(H.post_w * 0.5 / nx / H.n_layer))
    scopename = tf.get_variable_scope().name
    a = residual_dropout(a, train, key=f'{scopename}-a', pdrop=pdrop)

    x = x[:, -1:, :] if use_cache else x
    x = bs.add(x, a)

    inner_dim = int(nx * H.mlp_multiple)

    m = norm("mlp", x)
    m = linear('mlp_proj1', m, inner_dim,
               std=np.sqrt(H.mlp_w1 / nx), fast_gelu=True)
    m = linear('mlp_proj2', m, nx,
               std=np.sqrt(H.mlp_w2 / inner_dim / H.n_layer * 0.5))
    m = residual_dropout(m, train, key=f'{scopename}-m', pdrop=pdrop)

    return bs.add(x, m)


def add_position_embedding(x, x_emb, train, step):
    num_e = H.emb_number
    emb_std = H.pos_embd_std * np.sqrt(1.0 / num_e)
    for idx in range(H.emb_number):
        vsize = H.emb_vocabs[idx]
        name = f"pos_emb_{idx}"
        we = tf.get_variable(
            name, [vsize, H.n_embd], dtype=H.dtype,
            initializer=random_or_zeros_init(stddev=emb_std))
        e = bs.embedding_lookup(we, x_emb[:, idx, :])
        e = embedding_dropout(e, train)
        x += e
    return x


def stack(X, X_emb, train, step=None, cache=None):
    with tf.name_scope('input_processing'):
        we = tf.get_variable(
            "we", [H.n_vocab, H.n_embd], dtype=H.dtype,
            initializer=random_or_zeros_init(stddev=H.w_embd_std))
        h = bs.embedding_lookup(we, X)
        H.we = we
        H.we_x = h
        h = embedding_dropout(h, train)

    h = add_position_embedding(h, X_emb, train, step=step)
    if step is None:
        h = tf.reshape(h, [H.n_batch, H.attn_ctx, H.n_embd])
    else:
        h = tf.reshape(h, [H.sample_batch, -1, H.n_embd])

    with tf.variable_scope('sos_token'):
        if H.num_self_gen_in_use > 0 and not H.use_unconditional_augmentation:
            y_gen_idx = 0
            sos_tok = 0
            for typ in H.self_gen_types:
                if not typ.is_used:
                    if mpi_rank() == 0:
                        print(f" [self-gen] not using {typ.description}")
                    continue
                if mpi_rank() == 0:
                    print(f" [self-gen] using {typ.description}")
                this_sos_var = tf.get_variable(
                    typ.sos_name,
                    [typ.num_tokens, H.n_embd],
                    dtype=H.dtype,
                    initializer=random_or_zeros_init(stddev=H.w_embd_std))
                this_sos_tok = bs.embedding_lookup(this_sos_var, H.Y_gen_ph[:, y_gen_idx:y_gen_idx + 1])
                assert this_sos_tok.shape[1:] == (1, H.n_embd)
                sos_tok += this_sos_tok
                y_gen_idx += 1
            assert y_gen_idx == H.num_self_gen_in_use
        else:
            sos = tf.get_variable(
                'sos', [1, 1, H.n_embd], dtype=H.dtype,
                initializer=random_or_zeros_init(stddev=H.w_embd_std))
            batch_size = H.n_batch if step is None else H.sample_batch
            sos_tok = tf.ones(shape=[batch_size, 1, H.n_embd], dtype=H.dtype) * sos
    if step is None:
        h = tf.concat([sos_tok, h[:, :-1, :]], axis=1)
        if H.randomly_determined_order_use_lookahead:
            print("lookahead_embd")
            with tf.variable_scope("lookahead_embedding"):
                h = add_position_embedding(h, X_emb, train, step=step)
    else:
        h = tf.concat([sos_tok, h], axis=1)[:, -1:, :]

    new_cache = []
    modes = H.attention_layers.split(',')
    assert H.n_layer % len(modes) == 0

    for layer_idx in range(H.n_layer):
        mode = modes[layer_idx % len(modes)]
        name = f'h{layer_idx}'
        if cache is not None:
            # We only cache the pre qkv tensor, as it takes up
            # too much memory otherwise on long sequences.
            h = tf.concat([cache[layer_idx], h], axis=1)
            new_cache.append(h)
            use_cache = True
        else:
            use_cache = False

        with tf.variable_scope(name):
            recompute = H.recompute and train
            if H.float16 and H.blocksparse_op and not use_cache:
                h = sparse_attention(h, H.n_head, mode, use_cache=use_cache,
                                     train=train, recompute=recompute)
            else:
                h = dense_attention(h, H.n_head, mode, use_cache=use_cache,
                                    train=train, recompute=recompute)

    if cache is not None:
        return h, new_cache

    return h


def get_logits(name, h, n_out, train=False):
    n, t, nx = shape_list(h)
    w_std = np.sqrt(H.logits_w / nx)
    with tf.variable_scope(name):
        w = tf.get_variable(
            "logits_proj", [nx, n_out], dtype=H.dtype,
            initializer=random_or_zeros_init(stddev=w_std))
        w = embedding_dropout(w, train)
        h = tf.reshape(h, [-1, nx])
        logits = tf.matmul(h, w)
    return tf.reshape(logits, [n, t, n_out])


def get_losses(logits, labels, mask=None):
    with tf.name_scope('loss'):

        n, t, nx = shape_list(logits)
        ln, lt = shape_list(labels)
        assert lt == t
        labels = tf.reshape(labels, [-1])
        logits = tf.reshape(logits, [-1, nx])

        if H.float16 and logits.shape[-1].value <= 65536 and logits.dtype == tf.float16:
            # much faster fused fp16 implementation that also saves memory
            losses = bs.softmax_cross_entropy(logits=logits, labels=labels)
        else:
            logits = tf.cast(logits, tf.float32)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)

        losses = tf.reshape(losses, [n, t])
        if mask is not None:
            # X_mask can be either boolean or scalar (weighted) mask
            return (tf.reduce_sum(losses * mask) / tf.reduce_sum(mask)), losses
        return tf.reduce_mean(losses), losses


def model(train=False):
    with tf.variable_scope('model', custom_getter=f32_storage_getter):
        network_input = H.X_ph
        network_target = H.X_ph
        if H.rand_augment and train:
            assert network_input.shape[-1] == 3072, 'TODO: support other image sizes'
            network_input = tf.reshape(tf.cast(network_input, tf.uint8), [-1, 32, 32, 3])
            if H.rand_augment_conditioning:
                if H.use_unconditional_augmentation:
                    raise NotImplementedError
                rand_augment_idx = [t.sos_name for t in H.self_gen_types if t.is_used].index('sos_aa')
                batch = []
                with tf.device('/cpu:0'):
                    for i in range(H.n_batch):
                        example = network_input[i]
                        with_randaug = distort_image_with_randaugment(example, H.rand_augment_n, H.rand_augment_m)
                        without_randaug = example
                        should_autoaugment = tf.cast(H.Y_gen_ph[i, rand_augment_idx], tf.bool)
                        example = tf.cond(should_autoaugment, lambda: with_randaug, lambda: without_randaug)
                        batch.append(example)
                network_input = batch
            else:
                with tf.device('/cpu:0'):
                    network_input = [distort_image_with_randaugment(network_input[i], H.rand_augment_n, H.rand_augment_m) for i in range(H.n_batch)]
            network_input = tf.cast(tf.reshape(tf.concat(network_input, axis=0), [-1, 3072]), H.X_ph.dtype)
            network_target = network_input

        h = stack(network_input, H.X_emb_ph, train=train)
        h = norm('final_norm', h, epsilon=1e-6)
        targets = network_target
        gen_logits = get_logits('gen_logits', h, H.n_vocab, train=train)
        gen_loss, gen_losses = get_losses(gen_logits, targets)
        return gen_loss, gen_losses


def sample_model():
    X = tf.zeros(shape=[H.sample_batch, 0], dtype=tf.int32)
    current_step = tf.constant(0, dtype=tf.int64)
    accumulated_output = X[:, :current_step]   # Everything up til now.
    current_input = X[:, current_step - 1:current_step]
    cache_vars = [tf.zeros(shape=[H.sample_batch, 0, H.n_embd],
                           dtype=H.dtype) for _ in range(H.n_layer)]
    cacheshapes = [tf.TensorShape([H.sample_batch, None, H.n_embd])
                   for _ in range(H.n_layer)]
    embd_index = tf.constant([0] * H.sample_batch, dtype=tf.int32)
    first_embd = tf.zeros(shape=[H.sample_batch, H.emb_number, 0],
                          dtype=tf.int32)

    loop_vars = [current_step, accumulated_output, current_input,
                 first_embd, embd_index, cache_vars]
    shape_invariants = [current_step.get_shape(),
                        tf.TensorShape([H.sample_batch, None]),
                        tf.TensorShape([H.sample_batch, None]),
                        tf.TensorShape([H.sample_batch, H.emb_number, None]),
                        embd_index.get_shape(),
                        cacheshapes]
    embd_shapes = tf.constant(H.emb_vocabs, dtype=tf.int32)

    def cond(step, acc, curr, curr_embd, embd_index, cache):
        return step < H.attn_ctx

    def body(step, acc, curr, curr_embd, embd_index, cache):
        with tf.variable_scope('model', custom_getter=f32_storage_getter):
            h, cache = stack(curr, curr_embd, train=False, step=step,
                             cache=cache)
            h = norm('final_norm', h, epsilon=1e-6)

            h = h[:, -1:, :]
            logits = tf.cast(get_logits('gen_logits', h, H.n_vocab), tf.float32)
            logits = tf.reshape(logits, [H.sample_batch, H.n_vocab])
            temp = H.temperature
            symbol = tf.cast(tf.multinomial(logits / temp, 1), tf.int32)
            with tf.device('/cpu:0'):
                next_embd = tf.unravel_index(embd_index, embd_shapes)
                # unravel_index yields a embd_size, n_batch tensor
                next_embd = tf.transpose(next_embd, [1, 0])
                next_embd = tf.reshape(next_embd, [
                    H.sample_batch, H.emb_number, 1])
                next_index = embd_index + 1
        return (step + 1, tf.concat([acc, symbol], axis=1), symbol,
                next_embd, next_index, cache)

    _, output_seq, _, _, _, _ = tf.while_loop(
        cond=cond, body=body, loop_vars=loop_vars, back_prop=False,
        shape_invariants=shape_invariants, parallel_iterations=1)

    # Now, we want to gather the images across all ranks which have generated
    # them. We will just allreduce a sparse tensor.
    all_samples = [tf.zeros_like(output_seq) for _ in range(mpi_size())]
    all_samples[mpi_rank()] = output_seq
    all_samples = tf.cast(allreduce(tf.cast(
        tf.concat(all_samples, axis=0), tf.float32)), tf.int32)
    return all_samples


def warmup_cosine(current_iter):
    current_iter = tf.cast(current_iter, tf.float32) + 1.0
    warmup_iters = tf.cast(H.warmup_iters, tf.float32)
    s = tf.cast(tf.less(current_iter, warmup_iters), tf.float32)
    current_fraction = ((current_iter - warmup_iters) /
                        (H.n_updates_total - warmup_iters))
    return (s * (current_iter / warmup_iters) +
            (1 - s) * (0.5 * (1 + tf.cos(math.pi * current_fraction))))


def warmup_linear_decay(current_iter):
    current_iter = tf.cast(current_iter, tf.float32) + 1.0
    warmup_iters = tf.cast(H.warmup_iters, tf.float32)
    s = tf.cast(tf.less(current_iter, warmup_iters), tf.float32)
    current_fraction = tf.minimum(
        ((current_iter - warmup_iters) / (H.n_updates_total - warmup_iters)),
        tf.cast(1, tf.float32))
    return (s * (current_iter / warmup_iters) +
            (1 - s) * (1.0 - current_fraction))


def mpi_train():

    with tf.device('/cpu:0'), tf.name_scope('optimizer'):
        if H.decay_lr_linearly:
            lr_at_time = H.lr * warmup_linear_decay(H.global_step - H.lr_offset)
        else:
            lr_at_time = H.lr * warmup_cosine(H.global_step - H.lr_offset)
        rcp_mpi_size = tf.constant(1.0 / mpi_size())
        grad_scale = tf.reciprocal(H.curr_loss_scale)

    with tf.device("/gpu:0"):
        avg_loss_gen, _ = model(train=True)
        H.train_gen_loss = avg_loss_gen
        # n_updates_per_epoch H.global_step
        loss_to_optimize = avg_loss_gen
        params = tf.trainable_variables()
        grads = bs.gradients(bs.scale_tensor(loss_to_optimize, H.curr_loss_scale), params)

        if H.merge_layer_allreduce > 0:
            search_strings = list()
            stride = H.merge_layer_allreduce
            for l in range(H.n_layer - 1, -1, -stride):
                search_strings.append([f"model/h{j}" for j in range(l, l - stride, -1)])
        else:
            logprint('Not interleaving allreduce with backprop! Is slow.')
            search_strings = None

        if mpi_size() > 1:

            H.train_gen_loss = allreduce(bs.scale_tensor(avg_loss_gen, rcp_mpi_size))

            # Pre-scale the gradients to give all-reduce some room.
            # After gradients are computed on this device scaling here can be rather aggressive.
            # But 1/mpi_size should be enough.
            grads = [bs.filter_tensor(x, rcp_mpi_size) for x in grads]

            cast_all = tf.float16 if H.fp16_allreduce else None
            grads = group_allreduce(grads, params, search_strings=search_strings, cast_all=cast_all)

            serialize_allreduce_ops([H.train_gen_loss] + grads)

        if H.log_grad_stats and mpi_rank() == 0:
            grads = log_gradient_values(grads, params, H.global_step, model_dir=H.model_dir)

        train_op, global_norm = get_optimizer(H.optimizer)(
            grads, params,
            learning_rate=lr_at_time,
            grad_scale=grad_scale,
            fp16_mean_var=H.fp16_mean_var,
            max_grad_norm=H.max_grad_norm,
            static_loss_scaling=H.float16 and not H.dynamic_loss_scaling,
            beta2=H.beta2)

        if H.l2_loss > 0:
            # AdamW
            logprint('enabling l2 loss with value', H.l2_loss)
            updates = [train_op]
            l2_updates = []
            for p in params:
                if len(shape_list(p)) > 1:
                    l2_updates.append(p.assign(p - lr_at_time * H.l2_loss * p))
            updates.extend(l2_updates)
            train_op = tf.group(*updates)

        if not H.disable_ema_vars:
            # Polyak average of params. Stores an extra copy.
            # NOTE: this assignment is stateful -- graphs created after this will use the EMA var, see
            # the variable getter, so the order of mpi_train and eval model creation cannot be swapped.
            # TODO: remove this constraint
            H.ema = bs.Ema(decay=H.weights_beta)
            with tf.control_dependencies([train_op]):
                train_op = H.ema.apply(params)

    return train_op, lr_at_time, global_norm


def eval(test=False, epoch=None):
    if test:
        tx = dataset.teX
    else:
        tx = dataset.vaX
    losses = []
    for data in iter_data_mpi(tx, n_batch=H.n_batch, log=logprint,
                              split_by_rank=dataset.full_dataset_valid):
        feeds = {H.X_ph: data[0], H.X_emb_ph: H.x_emb}
        if H.num_self_gen_in_use > 0 and not H.use_unconditional_augmentation:
            feeds[H.Y_gen_ph] = np.zeros((data[0].shape[0], H.num_self_gen_in_use), dtype=np.int32)
        losses.append(sess.run(H.avg_eval_loss_gen, feeds))

    avg_loss = sum(losses) / len(losses)
    content = dict(epoch=epoch, series='eval_loss', loss=avg_loss, bits=avg_loss / np.log(2.))
    logprint(**content)
    mpi_barrier()
    return avg_loss


def get_data(partition):
    return {
        "train": (dataset.trX, dataset.trY),
        "valid": (dataset.vaX, dataset.vaY),
        "test": (dataset.teX, dataset.teY),
    }[partition]


def aug_eval(partition, epoch):
    tx, ty = get_data(partition)
    if H.aug_eval_n_examples is not None:
        tx = tx[:H.aug_eval_n_examples]
        if ty is not None:
            ty = ty[:H.aug_eval_n_examples]
    gen_in_use = [gen for gen in H.self_gen_types if gen.is_used]
    if not gen_in_use:
        gen_in_use = [AugmentationType("sos", "identity", 1, True, identity)]
    aug_choices = [gen.num_tokens for gen in gen_in_use]
    for aug_types in go_over(aug_choices):
        fname = os.path.join(
            H.model_dir,
            f"{H.desc}_" + "_".join(map(str, aug_types)) + "_losses.npz")
        if os.path.exists(fname):
            if mpi_rank() == 0:
                print(f" Evaluated {fname}")
            continue
        if mpi_rank() == 0:
            print(f"Evaluating {fname}")
        losses = []
        imgs = []
        for data in iter_data_mpi(tx, n_batch=H.n_batch, log=logprint,
                                  split_by_rank=dataset.full_dataset_valid):
            feeds = {H.X_ph: data[0], H.X_emb_ph: H.x_emb}
            x_emb = np.concatenate([H.x_emb.copy() for _ in range(H.n_batch)], axis=0)
            d_in = data[0]
            if H.num_self_gen_in_use > 0:
                y_gen_list = []
                for aug_type, gen in zip(aug_types, gen_in_use):
                    if gen.sos_name == 'sos_data':
                        raise NotImplementedError("sos_data is not supported in aug_eval")
                    yy = np.full((H.n_batch, 1), aug_type, dtype=np.int32)
                    d_in, x_emb, y_gen = gen.fn(d_in, x_emb, yy=yy)
                    assert (y_gen == yy).all()
                    y_gen_list.append(y_gen)
                feeds[H.X_ph] = d_in
                if H.permute_embeddings:
                    feeds[H.X_emb_ph] = x_emb
                if not H.use_unconditional_augmentation:
                    feeds[H.Y_gen_ph] = np.concatenate(y_gen_list, axis=1)
                    assert (feeds[H.Y_gen_ph] == np.stack([aug_types] * H.n_batch)).all()
            imgs.append(d_in)
            cur_loss = sess.run(H.eval_gen_losses, feeds)
            assert cur_loss.shape[0] == H.n_batch
            losses.append(cur_loss)

        losses = np.concatenate(losses, axis=0).astype(np.float32)
        assert losses.shape[0] == tx.shape[0] // mpi_size()
        mpi_barrier()
        losses = mpi_allgather(losses)
        assert losses.shape[0] == tx.shape[0]
        loss = losses.mean()

        content = dict(epoch=epoch, aug_types=aug_types, loss=loss, bpd=loss / np.log(2.0))
        logprint(**content)
        content["losses"] = losses
        if mpi_rank() == 0:
            np.savez(fname, **content)

        imgs = np.concatenate(imgs, axis=0)
        assert imgs.shape[0] == tx.shape[0] // mpi_size()
        mpi_barrier()
        imgs = mpi_allgather(imgs)
        assert imgs.shape == tx.shape
        if mpi_rank() == 0 and partition != "test":
            fname = os.path.join(H.model_dir, f"{H.desc}_" + "_".join(map(str, aug_types)) + "_imgs.npz")
            np.savez(fname, imgs=imgs.reshape(dataset.orig_shape))
        mpi_barrier()


def sample(name):
    sample_batches = []
    assert H.samples_to_generate % (H.sample_batch * mpi_size()) == 0
    for idx in range(H.samples_to_generate // (H.sample_batch * mpi_size())):
        feeds = {}
        if H.num_self_gen_in_use > 0 and not H.use_unconditional_augmentation:
            feeds[H.Y_gen_ph] = np.zeros((H.sample_batch, H.num_self_gen_in_use), dtype=np.int32)
        samples = sess.run(sample_output, feeds)
        sample_batches.append(samples)
        logprint(f'generated {sum([a.shape[0] for a in sample_batches])} / {H.samples_to_generate} samples')
        if idx == 0 and H.samples_memorycheck:
            mem = sess.run(tf.contrib.memory_stats.MaxBytesInUse())
            logprint('Runtime memory usage so far (bytes):', f'{mem:,}')
            logprint(memory_usage=mem)
    if mpi_rank() == 0:
        samples = np.concatenate(sample_batches, axis=0)
        nppath = os.path.join(H.model_dir, f'{H.desc}-samples-{H.samples_to_generate}-t{H.temperature}.npy')
        np.save(nppath, samples)


def sample_augmentation_type(n, size=None, nprng=None):
    """
    Sample one of `n` augmentation types. Index 0 is reserved for not
    augmenting.
    """
    if nprng is None:
        nprng = np.random
    if H.unaugmented_data_rate is None:
        y = nprng.randint(n, size=size)
    else:
        # We draw multiple augmentation types independently, so the probability
        # of not using augmentation has to be discounted accordingly.
        n_types = max(H.num_self_gen_in_use, 1)
        p = H.unaugmented_data_rate ** (1.0 / n_types)
        pmf = [p] + [(1.0 - p) / (n - 1)] * (n - 1)
        y = nprng.choice(n, size=size, p=pmf)
    return y.astype(np.int32)


def data_aug(xx, nprng=None, yy=None):
    """just hflip"""
    if nprng is None:
        nprng = np.random
    xx = xx.reshape(dataset.orig_shape)
    if yy is None:
        yy = sample_augmentation_type(2, size=xx.shape[0], nprng=nprng)
    assert yy.shape[0] == xx.shape[0]
    # n = len(xx)
    # xx = np.pad(xx, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='reflect')
    xx = [np.fliplr(x) if y else x for x, y in zip(xx, yy)]
    # ii = nprng.randint(low=0, high=4 * 2 + 1, size=n)
    # jj = nprng.randint(low=0, high=4 * 2 + 1, size=n)
    # xx = [x[i:i + 32, j:j + 32] for x, i, j in zip(xx, ii, jj)]
    xx = np.asarray(xx).reshape(dataset.shape)
    return xx


def identity(xx, x_emb, nprng=None, yy=None):
    return xx, x_emb, yy


def rotate(xx, x_emb, nprng=None, yy=None):
    b = xx.shape[0]
    b_emb, n_emb, n_ctx = x_emb.shape
    assert b == b_emb
    assert n_ctx == np.prod(dataset.orig_shape[1:])
    if yy is None:
        yy = sample_augmentation_type(4, size=(b, 1), nprng=nprng)
    assert yy.shape[0] == xx.shape[0]
    xx = xx.reshape(dataset.orig_shape)
    xx = [np.rot90(x, k=yy[i, 0], axes=(1, 0)) for i, x in enumerate(xx)]
    xx = np.asarray(xx).reshape(dataset.shape)
    x_emb = x_emb.reshape((b_emb, n_emb, *dataset.orig_shape[1:]))
    x_emb = [np.rot90(x, k=yy[i, 0], axes=(2, 1)) for i, x in enumerate(x_emb)]
    x_emb = np.asarray(x_emb).reshape((b_emb, n_emb, n_ctx))
    return xx, x_emb, yy


def transpose(xx, x_emb, nprng=None, yy=None):
    b = xx.shape[0]
    b_emb, n_emb, n_ctx = x_emb.shape
    assert b == b_emb
    assert n_ctx == np.prod(dataset.orig_shape[1:])
    if yy is None:
        yy = sample_augmentation_type(2, size=(b, 1), nprng=nprng)
    assert yy.shape[0] == xx.shape[0]
    xx = xx.reshape(dataset.orig_shape)
    xx = [np.transpose(x, [1, 0, 2]) if yy[i, 0] == 1 else x for i, x in enumerate(xx)]
    xx = np.asarray(xx).reshape(dataset.shape)
    x_emb = x_emb.reshape((b_emb, n_emb, *dataset.orig_shape[1:]))
    x_emb = [np.transpose(x, [0, 2, 1, 3]) if yy[i, 0] == 1 else x for i, x in enumerate(x_emb)]
    x_emb = np.asarray(x_emb).reshape((b_emb, n_emb, n_ctx))
    return xx, x_emb, yy


def reverse(xx, x_emb, nprng=None, yy=None):
    b = xx.shape[0]
    b_emb, n_emb, n_ctx = x_emb.shape
    assert b == b_emb
    assert n_ctx == np.prod(dataset.orig_shape[1:])
    if yy is None:
        yy = sample_augmentation_type(2, size=(b, 1), nprng=nprng)
    assert yy.shape[0] == xx.shape[0]
    xx = xx.reshape(dataset.orig_shape)
    xx = [np.rot90(x, k=yy[i, 0] * 2, axes=(1, 0)) for i, x in enumerate(xx)]
    xx = np.asarray(xx).reshape(dataset.shape)
    x_emb = x_emb.reshape((b_emb, n_emb, *dataset.orig_shape[1:]))
    x_emb = [np.rot90(x, k=yy[i, 0] * 2, axes=(2, 1)) for i, x in enumerate(x_emb)]
    x_emb = np.asarray(x_emb).reshape((b_emb, n_emb, n_ctx))
    return xx, x_emb, yy


def autoaugment_conditioning(rate):
    def fn(xx, x_emb, nprng=None, yy=None):
        if nprng is None:
            nprng = np.random
        b = xx.shape[0]
        # 1 when augment is applied
        if yy is None:
            yy = (nprng.uniform(size=(b, 1)) < rate).astype(np.int32)
        assert yy.shape[0] == xx.shape[0]
        return xx, x_emb, yy

    return fn


def permute_arbitrarily(random_perms):
    perms = [np.arange(dataset.ctx)] + random_perms
    n = len(perms)

    def fn(xx, x_emb, nprng=None, yy=None):
        b, n_ctx = xx.shape
        b_emb, n_emb, n_emb_ctx = x_emb.shape
        assert b == b_emb
        assert n_ctx == n_emb_ctx
        if yy is None:
            yy = sample_augmentation_type(n, size=(b, 1), nprng=nprng)
        assert yy.shape[0] == xx.shape[0]
        xx_new = []
        x_emb_new = []
        for i, y in enumerate(yy):
            xx_new.append(xx[i][perms[y[0]]])
            x_emb_new.append(x_emb[i][:, perms[y[0]]])
        xx = np.concatenate(xx_new, axis=0).reshape(dataset.shape)
        x_emb = np.concatenate(x_emb_new, axis=0)
        x_emb = x_emb.reshape(b_emb, n_emb, n_emb_ctx)
        return xx, x_emb, yy
    return fn


def remap_c(xx, order):
    new = np.zeros_like(xx)
    a, b, c = [(0, 1, 2),
               (0, 2, 1),
               (1, 0, 2),
               (1, 2, 0),
               (2, 0, 1),
               (2, 1, 0)
               ][order]
    new[:, :, 0] = xx[:, :, a]
    new[:, :, 1] = xx[:, :, b]
    new[:, :, 2] = xx[:, :, c]
    return new


def color_swap(xx, x_emb, nprng=None, yy=None):
    b = xx.shape[0]
    b_emb, n_emb, n_ctx = x_emb.shape
    assert b == b_emb
    assert n_ctx == np.prod(dataset.orig_shape[1:])
    if yy is None:
        yy = sample_augmentation_type(6, size=(b, 1), nprng=nprng)
    assert yy.shape[0] == xx.shape[0]
    xx = xx.reshape(dataset.orig_shape)
    x_emb = x_emb.reshape((b, n_emb * dataset.orig_shape[1], *dataset.orig_shape[2:]))
    xx_new = []
    x_emb_new = []
    for i, order in enumerate(yy):
        xx_new.append(remap_c(xx[i], order[0]))
        x_emb_new.append(remap_c(x_emb[i], order[0]))
    xx = np.concatenate(xx_new, axis=0).reshape(dataset.shape)
    x_emb = np.concatenate(x_emb_new, axis=0).reshape((b_emb, n_emb, n_ctx))
    return xx, x_emb, yy


def remap_jigsaw(x, order):
    r, c, ch = x.shape
    g = H.jigsaw_grid_size
    gr, gc = r // g, c // g
    x = x.reshape((g, gr, g, gc, ch))
    x = np.transpose(x, [0, 2, 1, 3, 4])
    x = x.reshape([g * g, gr, gc, ch])
    perm = H.jigsaw_perms[order]
    x = x[perm, :, :, :]
    x = x.reshape([g, g, gr, gc, ch])
    x = np.transpose(x, [0, 2, 1, 3, 4])
    x = x.reshape((r, c, ch))
    return x


def jigsaw(xx, x_emb, nprng=None, yy=None):
    b = xx.shape[0]
    b_emb, n_emb, n_ctx = x_emb.shape
    r, c, ch = dataset.orig_shape[1:]
    assert b == b_emb
    assert n_ctx == np.prod(dataset.orig_shape[1:])
    xx = xx.reshape(dataset.orig_shape)
    if yy is None:
        yy = sample_augmentation_type(H.jigsaw_num_perms, size=(b, 1), nprng=nprng)
    assert yy.shape[0] == xx.shape[0]
    x_emb = x_emb.reshape(b, n_emb, r, c, ch)
    x_emb = np.transpose(x_emb, [0, 2, 1, 3, 4])
    x_emb = x_emb.reshape((b, n_emb * r, c, ch))
    xx_new = []
    x_emb_new = []
    for i, order in enumerate(yy):
        xx_new.append(remap_jigsaw(xx[i], order[0]))
        x_emb_new.append(remap_jigsaw(x_emb[i], order[0]))
    xx = np.concatenate(xx_new, axis=0).reshape(dataset.shape)
    x_emb = np.concatenate(x_emb_new, axis=0)
    x_emb = x_emb.reshape(b, r, n_emb, c, ch)
    x_emb = np.transpose(x_emb, [0, 2, 1, 3, 4])
    x_emb = x_emb.reshape((b_emb, n_emb, n_ctx))
    return xx, x_emb, yy


if __name__ == '__main__':
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_mpi_rank())
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parse_args_and_update_hparams(H, parser)

    H.model_dir = os.path.join(H.out_dir, H.desc)
    os.makedirs(H.model_dir, exist_ok=True)
    H.log_path = os.path.join(H.model_dir, 'log')
    logprint = logger(H.log_path)

    logprint(hyperparams=H, pprint=True)

    # Same numpy seed so we can shuffle the data across ranks similarly
    np.random.seed(H.seed)
    # Different seed for TF to randomize model sampling/dropout across ranks
    tf.set_random_seed(H.seed * mpi_rank())
    # Augmentation nprng
    aug_nprng = np.random.RandomState(H.aug_seed + mpi_rank())

    # Cache for objects/tensors that should persist through recompute, eval, and/or samples
    H.bst_cache = dict()
    H.dropout_cache = dict()
    H.dense_mask_cache = dict()
    H.var_cache = dict()

    H.ema = None
    H.reduced_targets = H.kmeans_targets or H.mse_targets
    H.attn_ctx = H.n_ctx
    H.dtype = tf.float16 if H.float16 else tf.float32

    bs.set_entropy()  # for bs.dropout

    if mpi_size() == 1:
        logprint("WARNING: Only one MPI rank, did you forget to run w/ MPI?")

    dataset = get_dataset(H.dataset)(H, logprint)
    if H.auxiliary_dataset is not None:
        if mpi_rank() == 0:
            logprint("")
        pmf = [1.0 - H.auxiliary_dataset_fraction, H.auxiliary_dataset_fraction]
        aux_dataset = get_dataset(H.auxiliary_dataset)(H, logprint)
        if H.auxiliary_dataset_subset_size is not None:
            n_train = H.auxiliary_dataset_subset_size
            aux_dataset.trX = aux_dataset.trX[:n_train]
            if mpi_rank() == 0:
                logprint(f"taking a subset of auxiliary dataset {len(aux_dataset.trX)}")
            aux_dataset.iters_per_epoch = n_train // (mpi_size() * aux_dataset.n_batch)
        datasets = (dataset, aux_dataset)
        dataset = JankySubsampledDataset(datasets, pmf, seed=H.auxiliary_dataset_seed)

    H.emb_number = dataset.num_embeddings
    H.emb_vocabs = dataset.embedding_sizes
    H.n_classes = dataset.n_classes
    H.X_emb_shape = [None] + [H.emb_number] + dataset.shape[1:]
    H.x_emb = dataset.x_emb

    # Making n_vocab the nearest multiple of 128 allows the usage of
    # tensor cores with fp16 on V100's, which speeds up large vocab problems
    if H.no_vocab_rounding:
        H.n_vocab = dataset.n_vocab
    else:
        H.n_vocab = (math.ceil(dataset.n_vocab / 128)) * 128

    H.X_shape = [None] + dataset.shape[1:]

    with tf.device("/gpu:0"), tf.name_scope('placeholders'):
        H.X_ph = tf.placeholder(tf.int32, H.X_shape)
        H.X_emb_ph = tf.placeholder(tf.int32, H.X_emb_shape)
        H.jigsaw_perms = list(itertools.permutations(list(range(H.jigsaw_grid_size ** 2))))
        H.jigsaw_num_perms = len(H.jigsaw_perms)
        nprng = np.random.RandomState(H.randomly_determined_order_seed)
        random_perms = [
            nprng.permutation(dataset.ctx) for _ in range(H.randomly_determined_order_num_perms)
        ]
        H.self_gen_types = [
            AugmentationType("sos_rot", "rotation", 4, H.use_rotation, rotate),
            AugmentationType("sos_c", "color swapping", 6, H.use_color, color_swap),
            AugmentationType("sos_tr", "transposition", 2, H.use_transposition, transpose),
            AugmentationType("sos_rev", "reverse", 2, H.use_reverse, reverse),
            AugmentationType("sos_js", f"jigsaw with grid size {H.jigsaw_grid_size}", H.jigsaw_num_perms, H.use_jigsaw, jigsaw),
            AugmentationType("sos_aa", "autoaugment", 2, H.rand_augment_conditioning, autoaugment_conditioning(H.rand_augment_rate)),
            AugmentationType("sos_rd", "randomly determined order", H.randomly_determined_order_num_perms + 1, H.use_randomly_determined_order, permute_arbitrarily(random_perms)),
            AugmentationType("sos_data", "dataset", 2, H.use_dataset_conditioning, None),
        ]
        H.num_self_gen_in_use = sum(typ.is_used for typ in H.self_gen_types)
        if mpi_rank() == 0:
            for typ in H.self_gen_types:
                if typ.is_used:
                    logprint(f"Using [{typ.description}]")
                else:
                    logprint(f"Not using [{typ.description}]")
            if H.use_unconditional_augmentation:
                logprint(f"Training without augmentation prompting")
            else:
                logprint(f"Training with augmentation prompting")
            if H.permute_embeddings:
                logprint("Permuting embeddings")
            else:
                logprint("Not permuting embeddings")
        if H.num_self_gen_in_use > 0 and not H.use_unconditional_augmentation:
            H.Y_gen_ph = tf.placeholder(tf.int32, [None, H.num_self_gen_in_use])

    with tf.device("/cpu:0"):
        loss_scale_ph = tf.placeholder(
            tf.float32, shape=[], name="loss_scale")

        H.global_step = tf.get_variable(
            'global_step', initializer=zeros_init(), shape=tuple(),
            trainable=False, dtype=tf.int64)

        num_epochs = tf.get_variable(
            'num_epochs', initializer=zeros_init(), shape=tuple(),
            trainable=False, dtype=tf.int64)

        num_examples_processed = tf.get_variable(
            'num_examples_processed', initializer=zeros_init(), shape=tuple(),
            trainable=False, dtype=tf.int64)

        curr_loss_scale = tf.get_variable(
            'curr_loss_scale', initializer=constant_or_zeros_init(H.fp16_loss_scale),
            shape=tuple(), trainable=False, dtype=tf.float32)

        H.curr_loss_scale = curr_loss_scale

        best_val_loss = tf.get_variable(
            'best_val_loss', initializer=constant_or_zeros_init(99999),
            shape=tuple(), trainable=False, dtype=tf.float32)

        val_loss = tf.placeholder(tf.float32, shape=[], name="val_loss")
        update_val_loss = tf.assign(best_val_loss, val_loss)
        update_loss_scale = tf.assign(curr_loss_scale, loss_scale_ph)
        increment_epochs = tf.assign_add(num_epochs, 1)
        increment_examples = tf.assign_add(num_examples_processed, H.n_batch * mpi_size())
        increment_step = tf.assign_add(H.global_step, 1)

    n_updates_per_epoch = dataset.iters_per_epoch
    n_updates_total = H.total_epochs * n_updates_per_epoch
    H.n_updates_total = n_updates_total

    train_op, lr_at_step, global_norm = mpi_train()

    num_params = 0
    for p in tf.trainable_variables():
        num_params += np.prod(p.shape.as_list())
        if H.print_params:
            logprint(f'{p.name}, {p.shape.as_list()}, {np.prod(p.shape.as_list()):,}')

    with tf.name_scope('eval_model'), tf.device('/gpu:0'):
        avg_eval_loss_gen, eval_gen_losses = model(train=False)
        H.eval_gen_losses = eval_gen_losses
        H.avg_eval_loss_gen = allreduce(avg_eval_loss_gen) * (1.0 / mpi_size())

    if H.sample_and_exit or H.sample_during_eval:
        logprint('Creating sampling graph.')
        with tf.name_scope('sample_model'), tf.device('/gpu:0'):
            sample_output = sample_model()
        logprint('Done with sampling graph creation.')

    sess = get_session(mpi=True, disable_swapping=True, log=logprint)
    sess.run(tf.global_variables_initializer())

    logprint(f'Total number trainable parameters: {num_params:,}')
    logprint(num_params=num_params, n_vocab=H.n_vocab, n_batch=H.n_batch,
             n_ctx=H.n_ctx, effective_minibatch=mpi_size() * H.n_batch,
             n_updates_total=n_updates_total, n_updates_per_epoch=n_updates_per_epoch,
             pprint=True)

    if H.restore_path:
        if mpi_rank() == 0:
            localpath = maybe_download(H.restore_path)
            logprint("loading from " + localpath)
            load_variables_from_file(sess, localpath, ema=False)
            logprint("Done loading from " + localpath)

    with tf.name_scope('sync_variables'):
        if mpi_size() > 1:
            logprint('Syncing initial variables across gpus')
            sync_variables(sess)
            logprint('Finishing syncing variables')

    ema_loss = None
    steps_since_starting = 0
    save_dir = os.path.join(H.out_dir, H.desc)
    os.makedirs(save_dir, exist_ok=True)

    n_updates, n_epochs, curr_val_loss, loss_scale_t, examples_processed_t = sess.run([
        H.global_step, num_epochs, best_val_loss, curr_loss_scale, num_examples_processed])

    logprint(f"Starting at {n_updates} updates, {n_epochs} epochs, " +
             f"{curr_val_loss} best val loss, examples {examples_processed_t}")

    if H.sample_and_exit or H.sample_during_eval:
        sample('onload')
        if H.sample_and_exit:
            sys.exit(0)

    if H.eval_test or not H.skip_initial_evals or H.eval_and_exit:
        eval(test=H.eval_test, epoch=n_epochs)
        if H.eval_test or H.eval_and_exit:
            sys.exit(0)

    if H.aug_eval is not None:
        aug_eval(partition=H.aug_eval, epoch=n_epochs)
        sys.exit(0)

    # Free up some python memory
    H.bst_cache = None
    H.dropout_cache = None
    H.dense_mask_cache = None
    H.var_cache = None
    bs.clear_bst_constants()

    avg_t = 9999.0
    loss_count = 0
    if H.eval_after_n_examples:
        last_set_processed = examples_processed_t // H.eval_after_n_examples
    loss_scale_t = H.fp16_loss_scale
    times = []
    losses = []
    gns = []

    for i in range(n_epochs, H.total_epochs):
        t0 = time.time()
        args = [dataset.trX]
        if H.use_dataset_conditioning:
            args.append(dataset.auxX)
        for data in iter_data_mpi(*args, n_batch=H.n_batch, log=logprint,
                                  iters=n_updates_per_epoch, shuffle=True, seed=i,
                                  split_by_rank=dataset.full_dataset_train):
            outputs = [train_op, H.train_gen_loss, lr_at_step, global_norm]
            d_in = data_aug(data[0], nprng=aug_nprng) if H.aug else data[0]
            feeds = {H.X_ph: d_in, H.X_emb_ph: H.x_emb}
            if H.num_self_gen_in_use > 0:
                y_gen_list = []
                x_emb = np.concatenate([H.x_emb.copy() for _ in range(H.n_batch)], axis=0)
                d_gen = d_in.copy()
                for gen in H.self_gen_types:
                    if not gen.is_used:
                        continue
                    if gen.fn is None and gen.sos_name == 'sos_data':
                        y_gen = data[-1]
                    else:
                        d_gen, x_emb, y_gen = gen.fn(d_gen, x_emb, nprng=aug_nprng)
                    assert d_gen.shape == d_in.shape
                    assert y_gen.shape == (d_in.shape[0], 1)
                    y_gen_list.append(y_gen)
                feeds[H.X_ph] = d_gen
                if H.permute_embeddings:
                    feeds[H.X_emb_ph] = x_emb
                if not H.use_unconditional_augmentation:
                    feeds[H.Y_gen_ph] = np.concatenate(y_gen_list, axis=1)
            is_rank0 = mpi_rank() == 0

            if steps_since_starting == 2 or steps_since_starting == 65:
                mem = sess.run(tf.contrib.memory_stats.MaxBytesInUse())
                logprint('Runtime memory usage so far (bytes):', f'{mem:,}')
                logprint(memory_usage=mem)

            t1 = time.time()
            _, loss_t, lr_t, gn_t = sess.run(outputs, feeds)
            t2 = time.time()

            if H.dynamic_loss_scaling and H.float16:
                # slowly increase loss scale but quickly drop it when inf or nan is detected in the gradients
                # global_norm will be nan/inf when this happens
                if np.isfinite(loss_t) and np.isfinite(gn_t):
                    # Case: No infs or nans, roughly double the loss scale every 2k iters
                    loss_scale_t = sess.run(update_loss_scale, {loss_scale_ph: loss_scale_t * 1.0003466337})
                elif not np.isfinite(loss_t):
                    # Incurred some nans on the forward pass, don't do anything.
                    pass
                else:
                    # gn_t is nan/inf and loss_t is non-nan, meaning the grad scaling was too high
                    # Reduce by half and move to the next minibatch
                    if loss_scale_t > H.min_loss_scale:
                        loss_scale_t = sess.run(update_loss_scale, {loss_scale_ph: loss_scale_t * 0.5})

            step_t = sess.run(increment_step)
            examples_processed = sess.run(increment_examples)

            n_updates += 1
            gns.append(gn_t)
            times.append(t2 - t0)
            losses.append(loss_t)
            steps_since_starting += 1

            if (steps_since_starting in [2**n for n in range(9)] or
                    n_updates % H.iters_per_log == 0):
                loss_to_avg = [x for x in losses if np.isfinite(x)]
                if len(loss_to_avg) > 0:
                    avg_loss = sum(loss_to_avg) / len(loss_to_avg)
                else:
                    avg_loss = None
                avg_t = sum(times) / len(times)
                gns_so_far = [x for x in gns if np.isfinite(x)]
                if len(gns_so_far) > 0:
                    max_gn_so_far = max([x for x in gns if np.isfinite(x)])
                else:
                    max_gn_so_far = -1
                logprint(step=step_t, lr=lr_t, loss=loss_t, loss_avg=avg_loss,
                         t_iter=t2 - t1, t_iter_avg=avg_t, t_data=t1 - t0, gn=gn_t,
                         nans=len(losses) - len(loss_to_avg), loss_scale="2^%.0f" % np.log2(loss_scale_t),
                         max_gn=max_gn_so_far, series='train_loss',
                         examples=examples_processed)
                times = []
                losses = []
                gns = []

            t0 = time.time()

            if H.eval_after_n_examples:
                sets_processed = examples_processed // H.eval_after_n_examples
                if sets_processed > last_set_processed:
                    vl = eval(epoch=sets_processed)
                    if H.sample_during_eval:
                        sample(f'epoch-{sets_processed}')
                    if vl < curr_val_loss:
                        curr_val_loss = vl
                        sess.run(update_val_loss, {val_loss: vl})
                        logprint(f'Saving model with val loss of {vl} at epoch {sets_processed}')
                        save_params(sess, os.path.join(save_dir, 'model_best'))
                    save_params(sess, os.path.join(save_dir, 'model_latest'))
                    n = 12
                    if sets_processed in [2**i for i in range(n)] + [2**(n - 1) + 2 ** i for i in range(n)]:
                        save_params(sess, os.path.join(save_dir, f'model_epoch{sets_processed}'))

                    last_set_processed = sets_processed

        n_epochs = sess.run(increment_epochs)

        if not H.eval_after_n_examples:
            if n_epochs % H.epochs_per_eval == 0:
                vl = eval(epoch=n_epochs)
                if H.sample_during_eval:
                    sample(f'epoch-{n_epochs}')

                if vl < curr_val_loss:
                    curr_val_loss = vl
                    sess.run(update_val_loss, {val_loss: vl})
                    logprint(f'Saving model with val loss of {vl} at epoch {n_epochs}')
                    save_params(sess, os.path.join(save_dir, 'model_best'))

            if n_epochs % H.epochs_per_save == 0 and n_epochs > 0:
                save_params(sess, os.path.join(save_dir, 'model_latest'))

            if n_epochs in [2**i for i in range(12)]:
                save_params(sess, os.path.join(save_dir, f'model_epoch{n_epochs}'))

            if H.exit_after_n_epochs:
                if n_epochs >= H.exit_after_n_epochs:
                    time.sleep(20)
                    logprint(f'Exiting now, epoch={n_epochs}')
                    sys.exit(0)

    save_params(sess, os.path.join(save_dir, 'model_latest'))
    logprint('Finished training.')
