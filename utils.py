import os
import itertools
import json
import tempfile
import numpy as np
import tensorflow as tf
import blocksparse as bs
import time
import subprocess
from mpi_utils import mpi_rank


def logger(log_prefix):
    'Prints the arguments out to stdout, .txt, and .jsonl files'

    jsonl_path = f'{log_prefix}.jsonl'
    txt_path = f'{log_prefix}.txt'

    def log(*args, pprint=False, **kwargs):
        if mpi_rank() != 0:
            return
        t = time.ctime()
        argdict = {'time': t}
        if len(args) > 0:
            argdict['message'] = ' '.join([str(x) for x in args])
        argdict.update(kwargs)

        txt_str = []
        args_iter = sorted(argdict) if pprint else argdict
        for k in args_iter:
            val = argdict[k]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            argdict[k] = val
            if isinstance(val, float):
                if k == 'lr':
                    val = f'{val:.6f}'
                else:
                    val = f'{val:.4f}'
            txt_str.append(f'{k}: {val}')
        txt_str = ', '.join(txt_str)

        if pprint:
            json_str = json.dumps(argdict, sort_keys=True)
            txt_str = json.dumps(argdict, sort_keys=True, indent=4)
        else:
            json_str = json.dumps(argdict)

        print(txt_str, flush=True)

        with open(txt_path, "a+") as f:
            print(txt_str, file=f, flush=True)
        with open(jsonl_path, "a+") as f:
            print(json_str, file=f, flush=True)

    return log


def go_over(choices):
    return itertools.product(*[range(n) for n in choices])


def get_git_revision():
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return git_hash.strip().decode('utf-8')


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def rsync_data(from_path, to_path):
    subprocess.check_output(['rsync', '-r', from_path, to_path,
                             '--update'])


def maybe_download(path):
    '''If a path is a gsutil path, download it and return the local link,
    otherwise return link'''
    if not path.startswith('gs://'):
        return path
    local_dest = tempfile.mkstemp()[1]
    subprocess.check_output(['gsutil', '-m', 'cp', path, local_dest])
    return local_dest


def upload_to_gcp(from_path, to_path, is_async=False):
    if is_async:
        cmd = f'bash -exec -c "gsutil -m rsync -r {from_path} {to_path}"&'
        subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)
    else:
        subprocess.check_output(['gsutil', '-m', 'rsync', from_path, to_path])


def check_identical(from_path, to_path):
    try:
        subprocess.check_output(['git', 'diff', '--no-index', '--quiet',
                                 from_path, to_path])
        return True
    except subprocess.CalledProcessError:
        return False


def wait_until_synced(from_path, to_path):
    while True:
        if check_identical(from_path, to_path):
            break
        else:
            time.sleep(5)


def is_gcp():
    try:
        subprocess.check_output(['curl', '-s',
                                 'metadata.google.internal', '-i'])
        return True
    except subprocess.CalledProcessError:
        return False


def backup_files(save_dir, save_dir_gcp, path=None):
    if mpi_rank() == 0:
        if not path:
            print(f'Backing up {save_dir} to {save_dir_gcp}',
                  'Will execute silently in another thread')
            upload_to_gcp(save_dir, save_dir_gcp, is_async=True)
        else:
            upload_to_gcp(path, save_dir_gcp, is_async=True)


def log_gradient_values(grads, variables, global_step, model_dir):
    loggrads = []
    with tf.name_scope("log_gradient_values"):
        for i, (grad, param) in enumerate(zip(grads, variables)):
            name = param.op.name + "_" + "_".join(
                str(x) for x in param.shape.as_list())
            loggrads.append(bs.log_stats(
                grad, step=global_step, name=name,
                logfile=os.path.join(model_dir, 'grad_stats.txt')))
    return loggrads


def tf_print(t, name, summarize=10, first_n=None, mv=False, maxmin=False):
    # Useful for debugging!
    axes = [i for i in range(len(t.shape))]
    if mv:
        m, v = tf.nn.moments(t, axes=axes)
    if maxmin:
        maxi = tf.reduce_max(t)
        mini = tf.reduce_min(t)
    prefix = f'{tf.get_variable_scope().name}-{name}'
    with tf.device('/cpu:0'):
        if mv:
            t = tf.Print(t, [tf.shape(t), m, v], prefix,
                         summarize=summarize, first_n=first_n)
        elif maxmin:
            t = tf.Print(t, [tf.shape(t), mini, maxi, t], prefix,
                         summarize=summarize, first_n=first_n)
        else:
            t = tf.Print(t, [tf.shape(t), t], prefix,
                         summarize=summarize, first_n=first_n)
    return t


def get_variables(trainable=False):
    if trainable:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    else:
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return variables


def load_variables(sess, weights, ignore=None, trainable=False, ema=True):
    '''ema refers to whether the exponential moving averaged weights are used to
    initialize the true weights or not.'''
    weights = {os.path.normpath(key): value for key, value in weights.items()}
    ops = []
    feed_dict = {}
    if ema:
        gvs_map = {v.name: v for v in tf.global_variables()}
    for i, var in enumerate(get_variables(trainable=trainable)):
        var_name = os.path.normpath(var.name)
        if ignore:
            do_not_load = False
            for ignore_substr in ignore:
                if ignore_substr in var_name:
                    do_not_load = True
            if do_not_load:
                continue
        ph = tf.placeholder(dtype=var.dtype, shape=var.shape)
        ops.append(var.assign(ph))
        if ema:
            ema_name = f'{var_name[:-2]}/Ema/ema:0'
            # We assign the EMA value to the current value
            try:
                feed_dict[ph] = weights[ema_name]
            except KeyError:
                print(f'warning: ema var not found for {var_name}')
                feed_dict[ph] = weights[var_name]
            # We also assign the EMA value to the current EMA, which will otherwise
            # use the initialized value of the variable (random)
            ema_var = gvs_map[ema_name]
            ph = tf.placeholder(dtype=ema_var.dtype, shape=ema_var.shape)
            ops.append(ema_var.assign(ph))
            feed_dict[ph] = weights[ema_name]
        else:
            feed_dict[ph] = weights[var_name]
    sess.run(ops, feed_dict)


def save_params(sess, path):
    if mpi_rank() == 0:
        tf_vars = dict(zip([var.name for var in get_variables()],
                           sess.run(get_variables())))
        np.savez(path + '.npz', **tf_vars)


def load_variables_from_file(sess, path, ignore=None, trainable=False, ema=True):
    weights = dict(np.load(path))
    load_variables(sess, weights, ignore, trainable=trainable, ema=ema)
