from mpi4py import MPI
import numpy as np
import tensorflow as tf
import blocksparse as bs
from blocksparse import nccl


def mpi_init(initializer):
    'Variable initializer for MPI. Used such that allreduce '
    'syncs variables at the beginning of training. '
    'This is better than multiplying the values by 0, which requires'
    'extra memory. Alternatively, a broadcast can be used.'
    if mpi_rank() == 0:
        return initializer
    return tf.zeros_initializer()


def random_or_zeros_init(stddev):
    return mpi_init(tf.random_normal_initializer(stddev=stddev))


def constant_or_zeros_init(constant):
    return mpi_init(tf.constant_initializer(constant))


def zeros_init():
    return tf.zeros_initializer()


def num_comms():
    # perhaps make this editable later
    return 2


def mpi_size():
    return MPI.COMM_WORLD.Get_size()


def mpi_rank():
    return MPI.COMM_WORLD.Get_rank()


def num_nodes():
    # works only w 8 gpu nodes
    if mpi_size() > 8:
        return mpi_size() // 8
    return 1


def gpus_per_node():
    size = mpi_size()
    if size > 1:
        return max(size // num_nodes(), 1)
    return 1


def local_mpi_rank():
    return mpi_rank() % gpus_per_node()


def prereduce_size():
    if mpi_size() > 8:
        if mpi_size() % num_nodes() != 0:
            raise ValueError('MPI size not evenly divisible across nodes')
        return gpus_per_node()
    return 0


def allreduce(val):
    if mpi_size() == 1:
        return val
    return nccl.allreduce(val, num_comms=num_comms(), prereduce=prereduce_size())


def sync_variables(sess):
    sess.run(bs.nccl.sync_globals_zero_init_op(
        num_comms=num_comms(), prereduce=prereduce_size()))


def group_allreduce(grads, params, search_strings=None, cast_all=None):
    if mpi_size() == 1:
        return grads
    return nccl.group_allreduce(
        grads, params,
        search_strings=search_strings,
        cast_all=cast_all,
        num_comms=num_comms(),
        prereduce=prereduce_size())


def mpi_dtype(dtype):
    return {
        "float32": MPI.FLOAT,
        "float64": MPI.DOUBLE,
        "int8": MPI.CHAR,
        "uint8": MPI.UNSIGNED_CHAR,
        "int16": MPI.SHORT,
        "uint16": MPI.UNSIGNED_SHORT,
        "int32": MPI.INT,
        "uint32": MPI.UNSIGNED,
        "int64": MPI.LONG,
        "uint64": MPI.UNSIGNED_LONG,
    }[dtype]


def mpi_barrier():
    MPI.COMM_WORLD.Barrier()


def mpi_allgather(arr):
    comm = MPI.COMM_WORLD
    n = comm.Get_size()
    bs, *other = arr.shape
    out = np.zeros((bs * n, *other), dtype=arr.dtype)
    dtype = mpi_dtype(arr.dtype.name)
    comm.Allgather([arr, dtype], [out, dtype])
    return out


def get_session(mpi=True, disable_swapping=True, log=print):
    config = tf.ConfigProto()
    # if mpi:
    #     log('local rank', local_mpi_rank(), 'rank', mpi_rank())
    #     config.gpu_options.visible_device_list = str(local_mpi_rank())
    config.allow_soft_placement = False
    if disable_swapping:
        # Disables the swapping heuristic used by TF to reduce memory;
        # it is faster to recompute gradients rather than swap out params
        config.graph_options.rewrite_options.memory_optimization = 1

    # Dont need the timeout session if mpi4py is used when invoking mpi
    # sess = TimeoutSession(timeout=timeout, config=config, log=log)
    sess = tf.Session(config=config)
    return sess
