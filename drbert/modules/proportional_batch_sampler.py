import logging

import numpy as np

from ..constants import PARTITIONS

logger = logging.getLogger(__name__)


class ProportionalBatchSampler(object):
    """A helper class to sample batches propotionally from our `tasks` object. Proportional sampling
    was introduced in: https://arxiv.org/abs/1811.06031.

    Args:
        tasks (list): A list of dictionaries, one for each task.
        partition (str, optional): Which partition to sample from. Must be one of 'train',
            'validation', or 'test'.

    Raises:
        ValueError: If 'partition' is not in `drbert.constants.PARTITIONS`.
    """
    def __init__(self, tasks, partition='train'):
        if not tasks or any(not task for task in tasks):
            err_msg = (f"'tasks' must be a list of non-empty dictionaries. Got '{tasks}'.")
            logger.error('ValueError: %s', err_msg)
            raise ValueError(err_msg)
        if partition not in PARTITIONS:
            err_msg = (f"Invalid argument for 'partitions'. Expected on of {PARTITIONS}. Got"
                       f" '{partition}'.")
            logger.error('ValueError: %s', err_msg)
            raise ValueError(err_msg)

        self.partition = partition

        # Compute % of total batches for each task to use proportional sampling
        # See: https://arxiv.org/abs/1811.06031
        total_batches = sum([len(task['iterators'][partition]) for task in tasks])
        self.p = [len(task['iterators'][partition]) / total_batches for task in tasks]

        # Convert each torchtext iterator to a pyton generator object
        self.iterators = []
        for task in tasks:
            self.iterators.append((task['name'], task['task'], iter(task['iterators'][partition])))

    def get_batch(self):
        """Returns a tuple containing the name and batch of the sampled task.
        """
        # Training order of tasks is chosen using propotional sampling
        while True:
            name, task, iterator = self.iterators[np.random.choice(len(self.iterators), p=self.p)]
            try:
                batch = next(iterator)
                break
            # Re-draw a batch if this tasks iterator is exhausted.
            except StopIteration:
                pass

        return name, task, batch
