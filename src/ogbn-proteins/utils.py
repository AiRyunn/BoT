import random

import numpy as np
import torch
import torch.nn.functional as F


class DataLoaderWrapper(object):
    def __init__(self, dataloader):
        self.iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter)
        except Exception:
            raise StopIteration() from None


class BatchSampler(object):
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            shuf = torch.randperm(self.n).split(self.batch_size)
            for shuf_batch in shuf:
                yield shuf_batch
            yield None
