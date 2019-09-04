import numpy as np

class KFold_iterator:
    def __init__(self, input_dim, nr_splits=5):
        self.input_dim = input_dim
        self.nr_splits = nr_splits
        self.nr_iters_done = 0
        self.index_data = self.shuffle_split()

    def __iter__(self):
        return self

    def __next__(self):
        if self.nr_iters_done < self.nr_splits:
            self.iter = (np.concatenate(self.index_data[1:]), self.index_data[0])
            self.index_data = self.index_data[1:] + [self.index_data[0]]
            self.nr_iters_done += 1
            return self.iter
        else:
            raise StopIteration

    def shuffle_split(self):
        indexes = np.arange(self.input_dim)
        np.random.shuffle(indexes)
        return np.array_split(indexes, self.nr_splits)