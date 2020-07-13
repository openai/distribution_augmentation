import numpy as np


class JankySampler:
    def __init__(self, arr, seed=None):
        self.arr = arr
        self.nprng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.drawn = 0
        self.idx = self.nprng.permutation(len(self.arr))

    def draw(self, n):
        '''
        Shuffle the array if it's exhausted and draw `n` samples without
        replacement.
        '''
        if n > len(self.arr):
            raise ValueError("It looks like you tried to draw more than there are in the list")
        if self.drawn + n > len(self.arr):
            self.reset()
        end = self.drawn + n
        retval = self.arr[self.idx[self.drawn:end]]
        self.drawn = end
        return retval


class JankySubsampler:
    '''
    To be used with iter_data_mpi. This class reports it has the same number
    of examples as `arrays[0]`, but returns a mixed slice of examples from all
    `arrays`.
    '''
    def __init__(self, arrays, pmf, seed=None):
        assert len(pmf) == len(arrays)
        self.pmf = pmf
        self.arrays = arrays
        self.samplers = [JankySampler(arr, seed=seed) for arr in arrays]
        self.idxs = np.arange(len(self.pmf))
        self.nprng = np.random.RandomState(seed)
        for arr in arrays[1:]:
            assert arrays[0].shape[1:] == arr.shape[1:]

    @property
    def shape(self):
        return self.arrays[0].shape

    def __getitem__(self, key):
        n = len(key)
        arr = self.nprng.choice(self.idxs, size=n, p=self.pmf)
        ret = np.concatenate(
            [sampler.draw((arr == idx).sum())
             for idx, sampler in zip(self.idxs, self.samplers)],
            axis=0)
        return ret


if __name__ == "__main__":
    # Test 1
    arr = np.arange(9)
    sampler = JankySampler(arr, seed=42)
    for _ in range(4):
        print(sampler.draw(8))

    # Test 2
    pmf = [0.5, 0.5]
    arr1 = np.arange(4 * 2).reshape(4, 2) + 1
    arr2 = -np.arange(8 * 2).reshape(8, 2)
    subsampler = JankySubsampler((arr1, arr2), pmf, seed=42)
    aux_arr1 = np.ones((4,))
    aux_arr2 = np.zeros((8,))
    aux_subsampler = JankySubsampler((aux_arr1, aux_arr2), pmf, seed=42)
    dummy_indices = np.arange(4)  # Draw 4 arrays at a time
    for _ in range(10):
        print(subsampler[dummy_indices])
        print(aux_subsampler[dummy_indices])
