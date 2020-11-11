import pickle
import os
import numpy as np
import imageio
try:
    from sklearn.cross_validation import train_test_split
except ModuleNotFoundError:
    from sklearn.model_selection import train_test_split
from mpi_utils import mpi_size, mpi_rank
from janky_stuff import JankySubsampler


mpisize = mpi_size()
mpirank = mpi_rank()


def get_dataset(name):
    return {
        'cifar10': Cifar10,
        'imagenet64': Imagenet64,
        'imagenet32': Imagenet32,
    }[name]


def tile_images(images, d1=4, d2=4, border=1):
    id1, id2, c = images[0].shape
    out = np.ones([d1 * id1 + border * (d1 + 1),
                   d2 * id2 + border * (d2 + 1),
                   c], dtype=np.uint8)
    out *= 255
    if len(images) != d1 * d2:
        raise ValueError('Wrong num of images')
    for imgnum, im in enumerate(images):
        num_d1 = imgnum // d2
        num_d2 = imgnum % d2
        start_d1 = num_d1 * id1 + border * (num_d1 + 1)
        start_d2 = num_d2 * id2 + border * (num_d2 + 1)
        out[start_d1:start_d1 + id1, start_d2:start_d2 + id2, :] = im
    return out


def iter_data_mpi(*args, n_batch, log, shuffle=False, iters=None, seed=None, split_by_rank=True):
    'Take the tensors in *args and iterate through them across mpi ranks if split_by_rank, otherwise iter normally'
    if not args:
        raise ValueError
    size = args[0].shape[0]
    for idx in range(1, len(args)):
        if args[idx].shape[0] != size:
            raise ValueError(f'mismatch in arg {idx}, shape {args[idx].shape[0]} vs {size}')

    if seed:
        np.random.seed(seed)

    if shuffle:
        idxs = np.random.permutation(np.arange(size))
    else:
        idxs = np.arange(size)

    ms = mpisize
    mr = mpirank
    if not split_by_rank:
        ms = 1
        mr = 0

    # Truncate the data if it does not divide evenly
    sequences_per_batch = ms * n_batch
    length = (idxs.size // sequences_per_batch) * sequences_per_batch
    if length != idxs.size:
        log('Truncating {}/{} sequences'.format(idxs.size - length, idxs.size))
    idxs = idxs[:length]
    # Reshape starting indices to K*mpi_size*n_batch
    idxs = idxs.reshape([-1, ms, n_batch])
    log(f'Number of minibatches in this dataset: {len(idxs)}')
    for mb_idx in range(len(idxs)):
        indices = idxs[mb_idx, mr]
        vals = [t[indices] for t in args]
        yield vals
        if iters and mb_idx > iters:
            break


class ImageDataset(object):
    'Non-jpeg images'

    def decode(self, samples, logname):
        H = self.H
        out_samples = self.samples_to_image(samples)
        n_examples = out_samples.shape[0]
        d2 = H.sample_grid_dim
        if d2 > n_examples:
            d2 = n_examples
        d1 = n_examples // d2
        tiled_image = tile_images(out_samples, d1=d1, d2=d2)
        imname = f'{H.desc}-samples-{logname}.png'
        out_path = os.path.join(H.model_dir, imname)
        imageio.imwrite(out_path, tiled_image)
        self.logprint(f'Saved samples in file {out_path}')

    def initialize_image_embedding(self):
        w, h, c = self.embedding_sizes
        embedding = []
        for i in range(w):
            for j in range(h):
                for k in range(c):
                    embedding.append([i, j, k])
        self.x_emb = np.array(embedding).T.reshape([1, 3, self.ctx])

    def samples_to_image(self, samples):
        return samples.reshape(self.orig_shape)


class JankySubsampledDataset(ImageDataset):
    def __init__(self, datasets, pmf, seed=42):
        assert len(pmf) == len(datasets)
        if seed is None:
            raise ValueError("seed can't be None")
        self.datasets = datasets
        self.pmf = pmf
        # Some basic sanity-checks.
        attrs = (
            "orig_shape",
            "shape",
            "ctx",
            "num_embeddings",
            "embedding_sizes",
            "n_vocab",
            "x_emb",
        )
        for attr in attrs:
            assert hasattr(self.ref, attr), f"{attr} is missing in the main dataset."
            ref_attr = getattr(self.ref, attr)
            setattr(self, attr, ref_attr)
            for oth in self.oth:
                assert hasattr(oth, attr), f"{attr} is missing in the auxiliary dataset"
                oth_attr = getattr(oth, attr)
                assert type(ref_attr) == type(oth_attr)
                if isinstance(ref_attr, np.ndarray):
                    assert (ref_attr == oth_attr).all(), f"expected {attr} to be the same."
                else:
                    assert ref_attr == oth_attr, f"expected {attr} to be the same."
        # Perform model selection and evaluation using the main dataset.
        attrs = (
            "H",
            "logprint",
            "vaX",
            "vaY",
            "teX",
            "teY",
            "n_classes",
            "full_dataset_valid",
            "full_dataset_train",
            "iters_per_epoch",
        )
        for attr in attrs:
            setattr(self, attr, getattr(self.ref, attr, None))
        trX = [ds.trX for ds in datasets]
        auxX = [np.zeros_like(tr[:, 0:1]) + idx for idx, tr in enumerate(trX)]
        self.trX = JankySubsampler(trX, pmf, seed=seed)
        self.auxX = JankySubsampler(auxX, pmf, seed=seed)

    @property
    def ref(self):
        return self.datasets[0]

    @property
    def oth(self):
        return self.datasets[1:]


class Imagenet64(ImageDataset):
    '''To download, if your data dir is /root/data:

    mkdir -p /root/data
    cd /root/data
    wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/imagenet64-train.npy
    wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/imagenet64-valid.npy
    '''

    def __init__(self, H, logprint):
        self.logprint = logprint
        self.H = H
        # Whether the full dataset is loaded on each rank, or just its own partition
        self.full_dataset_train = True
        self.full_dataset_valid = True
        n_train = 1231149
        self.n_batch = H.n_batch
        self.orig_shape = [-1, 64, 64, 3]
        self.orig_pixels = 64 * 64 * 3
        self.num_embeddings = 3
        self.n_vocab = 256
        self.embedding_sizes = [64, 64, 3]
        self.iters_per_epoch = n_train // (mpisize * self.n_batch)
        tr = np.load('/root/data/imagenet64-train.npy', mmap_mode='r').reshape([-1, 12288])
        self.trX = tr[:n_train]
        self.trY = None
        self.vaY = None
        self.teY = None
        self.vaX = tr[n_train:]
        self.n_classes = None
        self.teX = np.load('/root/data/imagenet64-valid.npy', mmap_mode='r').reshape([-1, 12288])
        self.n_vocab = 256
        self.ctx = 12288
        self.shape = [-1, self.ctx]
        assert self.ctx == H.n_ctx, f'n_ctx should be {self.ctx}'
        self.initialize_image_embedding()


class Imagenet32(Imagenet64):
    '''To download, if your data dir is /root/data:
    mkdir -p /root/data
    cd /root/data
    wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/imagenet32-train.npy
    wget https://openaipublic.blob.core.windows.net/distribution-augmentation-assets/imagenet32-valid.npy
    '''

    def __init__(self, H, logprint):
        self.logprint = logprint
        self.H = H
        # 1281167 << dataset has this many examples
        # We will use 10k examples for dev
        n_train = 1281167 - 10000
        self.full_dataset_train = True
        self.full_dataset_valid = True
        self.n_batch = H.n_batch
        self.orig_shape = [-1, 32, 32, 3]
        self.trY = None
        self.vaY = None
        self.teY = None
        self.n_classes = None
        self.orig_pixels = 32 * 32 * 3
        self.num_embeddings = 3
        self.n_vocab = 256
        self.embedding_sizes = [32, 32, 3]
        self.iters_per_epoch = n_train // (mpisize * self.n_batch)
        # we are dumb and saved imagenet32 in 3x32x32, unlike ImageNet64, which we saved in transposed format, sorry about the inconsistency
        tr = np.load('/root/data/imagenet32-train.npy').reshape([-1, 3, 32, 32]).transpose(
            [0, 2, 3, 1]).reshape([-1, 3072])
        self.trX = tr[:n_train]
        self.vaX = tr[n_train:]
        self.teX = np.load('/root/data/imagenet32-valid.npy').reshape([-1, 3, 32, 32]).transpose(
            [0, 2, 3, 1]).reshape([-1, 3072])
        self.n_vocab = 256
        self.ctx = 3072
        self.shape = [-1, self.ctx]
        assert self.ctx == H.n_ctx, f'n_ctx should be {self.ctx}'
        self.initialize_image_embedding()


def flatten(outer):
    return [el for inner in outer for el in inner]


def unpickle_cifar10(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    data = dict(zip([k.decode() for k in data.keys()], data.values()))
    return data


def cifar10(data_dir, one_hot=True, test_size=None):
    test_size = test_size or 5000
    tr_data = [unpickle_cifar10(os.path.join(data_dir, 'data_batch_%d' % i)) for i in range(1, 6)]
    trX = np.vstack(data['data'] for data in tr_data)
    trY = np.asarray(flatten([data['labels'] for data in tr_data]))
    te_data = unpickle_cifar10(os.path.join(data_dir, 'test_batch'))
    teX = np.asarray(te_data['data'])
    teY = np.asarray(te_data['labels'])
    trX = trX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape([-1, 3072])
    teX = teX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape([-1, 3072])
    trX, vaX, trY, vaY = train_test_split(trX, trY, test_size=test_size, random_state=11172018)
    if one_hot:
        trY = np.eye(10, dtype=np.float32)[trY]
        vaY = np.eye(10, dtype=np.float32)[vaY]
        teY = np.eye(10, dtype=np.float32)[teY]
    else:
        trY = np.reshape(trY, [-1, 1])
        vaY = np.reshape(vaY, [-1, 1])
        teY = np.reshape(teY, [-1, 1])
    return (trX, trY), (vaX, vaY), (teX, teY)


class Cifar10(ImageDataset):
    def __init__(self, H, logprint):
        self.logprint = logprint
        self.H = H
        self.full_dataset_train = True
        self.full_dataset_valid = True
        # 5k examples for valid
        n_train = 45000
        if H.datapoints:
            n_train = H.datapoints
        self.n_batch = H.n_batch
        self.iters_per_epoch = n_train // (mpisize * self.n_batch)
        self.orig_shape = [-1, 32, 32, 3]
        self.n_classes = 10
        self.orig_pixels = 32 * 32 * 3
        self.num_embeddings = 3
        self.n_vocab = 256
        self.embedding_sizes = [32, 32, 3]
        self.n_batch = H.n_batch
        self.iters_per_epoch = n_train // (mpisize * self.n_batch)
        (self.trX, self.trY), (self.vaX, self.vaY), (self.teX, self.teY) = cifar10('/root/data/cifar10/', one_hot=False, test_size=H.test_size)
        if H.datapoints:
            logprint(f'Only using {H.datapoints} examples')
            self.trX = self.trX[:n_train]
            self.trY = self.trY[:n_train]
        self.shape = [-1, 3072]
        self.ctx = 32 * 32 * 3
        assert self.ctx == H.n_ctx, f'n_ctx should be {self.ctx}'
        self.initialize_image_embedding()

    def preprocess(self, arr):
        arr = arr.reshape([-1, 3, 32, 32])
        arr = arr.transpose([0, 2, 3, 1])
        return arr.reshape([-1, 3072])
