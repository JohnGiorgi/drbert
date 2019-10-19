import copy

import numpy as np


class ProportionalBatchSampler(object):
    def __init__(self, tasks, partition='train'):
        self.tasks = copy.deepcopy(tasks)
        self.partition = partition

        # Compute % of total batches for each task to use proportional sampling
        # See: https://arxiv.org/abs/1811.06031
        total_batches = sum([len(task['iterators'][partition]) for task in tasks])
        self.p = [len(task['iterators'][partition]) / total_batches for task in tasks]

        # Convert each torchtext iterator to a pyton generator object
        for task in tasks:
            task['iterators'][partition] = iter(task['iterators'][partition])

    def get_batch(self):
        """"""
        # Training order of tasks is chosen using propotional sampling
        while True:
            task = np.random.choice(self.tasks, p=self.p)
            name = task['name']
            try:
                batch = next(task['iterators'][self.partition])
                break
            # Re-draw a batch if this tasks iterator is exhausted.
            except StopIteration:
                pass

        return name, batch
