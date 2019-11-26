import logging

import torch
from transformers import AdamW
from transformers import WarmupLinearSchedule

from ..constants import TASKS
from ..data.dataset_readers import NLIDatasetReader
from ..data.dataset_readers import RelationClassificationDatasetReader
from ..data.dataset_readers import SequenceLabellingDatasetReader
from ..data.dataset_readers import STSDatasetReader

logger = logging.getLogger(__name__)


def prepare_optimizer_and_scheduler(args, model, t_total=None):
    """Returns an Adam optimizer configured for optimization of a Transformers model (`model`).

    Args:
        args (ArgumentParser): Object containing objects parsed from the command line.
        model (nn.Module): The Transformers model to be fine-tuned.
        t_total (int): The total number of optimization steps.

    Returns:
        A 2-tuple containing an initialized `AdamW` optimizer and `WarmupLinearSchedule` scheduler
        for the fine-tuning of a Transformers (`model`).
    """
    # These are hardcoded because transformers named them to match to TF implementations
    decay_blacklist = {'LayerNorm.bias', 'LayerNorm.weight'}

    decay, no_decay = [], []

    for name, param in model.named_parameters():
        # Frozen weights
        if not param.requires_grad:
            continue
        # A shape of len 1 indicates a normalization layer
        # See: https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
        if len(param.shape) == 1 or name.endswith('.bias') or name in decay_blacklist:
            no_decay.append(param)
        else:
            decay.append(param)

    grouped_parameters = [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': args.weight_decay}
    ]

    optimizer = AdamW(grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon,
                      correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=args.warmup * t_total,
                                     t_total=t_total)

    return optimizer, scheduler


def generate_inputs(name, task, batch, tokenizer):
    """A helper function which returns a dictionary that can be fed to the forward hook of a
    Transformers PyTorch model.

    Args:
        name (str): The unique name used to register the classification head.
        task (str): Which type of task this batch corresponds to. Must be in
            `drbert.constants.TASKS`.
        batch (torchtext.data.batch.Batch): A single batch from a torchtext iterator.
        tokenizer (PretrainedTokenizer): A transformers tokenizer object.

    Raises:
        ValueError: If `task` is not in `drbert.constants.TASKS`.

    Returns:
        dict: A dictionary which can be fed to the forward hook of a Transformers PyTorch model.
    """
    inputs = {'name': name}

    # Generate inputs based on the task
    if task in {'sequence_labelling', 'relation_classification'}:
        inputs.update({'input_ids': batch.text, 'labels': batch.label})
    elif task == 'document_classification':
        err_msg = 'Document classification is not yet implemented.'
        logger.error('NotImplementedError: %s', err_msg)
        raise NotImplementedError(err_msg)
    elif task == 'nli':
        input_ids = torch.cat((batch.premise, batch.hypothesis[:, 1:]), dim=-1)
        inputs.update({'input_ids': input_ids, 'labels': batch.label})
    elif task == 'sts':
        input_ids = torch.cat((batch.sentence1, batch.sentence2[:, 1:]), dim=-1)
        inputs.update({'input_ids': input_ids, 'labels': batch.label})
    else:
        err_msg = f"'task' must be one of {TASKS}. Got '{task}'."
        logger.error('NotImplementedError: %s', err_msg)
        raise ValueError(err_msg)

    # Generate attention masks on the fly
    inputs['attention_mask'] = torch.where(
        inputs['input_ids'] == tokenizer.pad_token_id,
        torch.zeros_like(inputs['input_ids']),
        torch.ones_like(inputs['input_ids'])
    )

    return inputs


def get_iterators_for_task(task, tokenizer, lower=False, device='cpu'):
    """Convience function which will return the iterators for a given `task` and `tokenizer`.

    Args:
        task (dict): A dictionary containing all information for a task neccecary to create a
            dataset loader.
        tokenizer (PretrainedTokenizer): A transformers tokenizer object.
        device (str or torch.device, optional): A string or instance of torch.device specifying
            which device the Tensors are going to be created on. If left as default, the tensors
            will be created on cpu. Default: None.

    Returns:
        dict: A dictionary keyed by the partitions in `task['partitions']`, containing an iterator
            for each of those partitions.
    """
    inputs = {'tokenizer': tokenizer, 'lower': lower, 'device': device, **task}

    if task['task'] == 'sequence_labelling':
        iterators = SequenceLabellingDatasetReader(**inputs).textual_to_iterator()
    elif task['task'] == 'relation_classification':
        iterators = RelationClassificationDatasetReader(**inputs).textual_to_iterator()
    elif task['task'] == 'nli':
        iterators = NLIDatasetReader(**inputs).textual_to_iterator()
    elif task['task'] == 'sts':
        iterators = STSDatasetReader(**inputs).textual_to_iterator()
    elif task['task'] == 'document_classification':
        err_msg = 'Document classification is not yet implemented.'
        logger.error('NotImplementedError: %s', err_msg)
        raise NotImplementedError(err_msg)
    else:
        err_msg = f''''task["name"]' must be one of {TASKS}. Got '{task}'.'''
        logger.error('ValueError: %s', err_msg)
        raise ValueError(err_msg)

    return iterators
