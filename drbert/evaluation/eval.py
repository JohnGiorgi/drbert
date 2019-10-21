import logging

import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from ..constants import OUTSIDE
from ..constants import SEQUENCE_CLASSIFICATION_TASKS
from ..constants import TASKS
from ..training.util import generate_inputs
from .util import precision_recall_f1_support_sequence_labelling
from .util import print_evaluation
from .util import reverse_dict

logger = logging.getLogger(__name__)


def evaluate(tasks, model, tokenizer, partitions=None):
    """The interface for evaluating DrBERT.

    This function handles evaluation of DrBERT (`model`) for some tasks (`tasks`).

    Args:
        tasks (list): A list of dictionaries containing information about each task to evaluate.
        model (model.DrBERT): The DrBERT model to evaluate.
        tokenizer (PretrainedTokenizer): A transformers tokenizer object.
        partitions (list, optional): A list of partition names to evaluate. If None, each partition
            in task['partitions']` is evaluated for each task in tasks. Defaults to None.

    Returns:
        dict: A dictionary (keyed by task names, `task['name']`) of dictionaries (keyed by
            partitions).
    """
    model.eval()

    results = {}

    for task in tasks:
        results[task['name']] = {}
        partitions = task['partitions'] if partitions is None else partitions

        # task_step = {task['name']: 0 for task in tasks}
        # eval_loss = {task['name']: 0.0 for task in tasks}

        for partition in partitions:
            logger.info(f"***** Running {task['name']} evaluation on {partition} *****")
            logger.info(f"  Num examples = {len(task['iterators'][partition])}")
            logger.info(f"  Batch size = {task['iterators'][partition].batch_size}")

            y_true, y_pred = [], []
            if task['task'] == 'sequence_labelling':
                orig_tok_mask = []

            eval_iterator = tqdm(
                task['iterators'][partition], desc="Evaluating", unit='batch', dynamic_ncols=True
            )

            for batch in eval_iterator:
                inputs = generate_inputs(task['name'], task['task'], batch, tokenizer)

                outputs = model(**inputs)
                loss, logits = outputs[:2]

                y_true.extend(batch.label.tolist())
                y_pred.extend(torch.argmax(logits, dim=-1).tolist())

                if task['task'] == 'sequence_labelling':
                    orig_tok_mask.extend(batch.mask.tolist())

                del batch, inputs, outputs, loss, logits

            idx_to_label = \
                reverse_dict(task['iterators'][partition].dataset.fields['label'].vocab.stoi)

            if task['task'] == 'sequence_labelling':
                scores = evaluate_sequence_labelling(y_true, y_pred, idx_to_label)
            elif task['task'] in SEQUENCE_CLASSIFICATION_TASKS:
                scores = evaluate_sequence_classification(y_true, y_pred, idx_to_label)
            elif task['task'] == 'document_classification':
                err_msg = 'Document classification is not yet implemented.'
                logger.error('NotImplementedError: %s', err_msg)
                raise NotImplementedError('Document classification is not yet implemented.')
            else:
                err_msg = f"'task' must be one of {TASKS}. Got '{task}'."
                logger.error('ValueError: %s', err_msg)
                raise ValueError(err_msg)

            results[task['name']][partition] = scores
            print_evaluation(scores, title=task['name'])

    return results


def evaluate_sequence_labelling(y_true, y_pred, idx_to_label, orig_tok_mask):
    """Performs evaluation for sequence labelling tasks.

    Args:
        y_true (list): A list containing the gold labels for each example.
        y_pred (list): A list containing the predicted labels for each example.
        idx_to_label (dict): A dictionary mapping indices (ints) to labels (strings).
        orig_tok_mask (list): A list containing the indices of original tokens in `y_true`.

    Returns:
        dict: A dictionary of scores keyed by the labels in `labels` where each score is a 4-tuple
            containing precision, recall, f1 and support. Additionally includes the keys
            'macro avg' and 'micro avg' containing the macro and micro averages across scores.
    """
    # TODO (John): This is going to throw and error (I think) when it hits the pads.
    y_true = torch.masked_select(torch.as_tensor(y_true), torch.as_tensor(orig_tok_mask)).tolist()
    y_pred = torch.masked_select(torch.as_tensor(y_pred), torch.as_tensor(orig_tok_mask)).tolist()

    # Map predicted indices to labels
    y_true = [idx_to_label[idx] for idx in y_true]
    y_pred = [idx_to_label[idx] for idx in y_pred]

    scores = precision_recall_f1_support_sequence_labelling(y_true, y_pred)

    # Add binary F1 scores
    y_true_binary = \
        [label if label == OUTSIDE else f'{label.split("-")[0]}-binary' for label in y_true]
    y_pred_binary = \
        [label if label == OUTSIDE else f'{label.split("-")[0]}-binary' for label in y_pred]

    scores['binary'] = \
        precision_recall_f1_support_sequence_labelling(y_true_binary, y_pred_binary)['binary']

    return scores


def evaluate_sequence_classification(y_true, y_pred, idx_to_label):
    """Performs evaluation for sequence classification tasks.

    Args:
        y_true (list): A list containing the gold labels for each example.
        y_pred (list): A list containing the predicted labels for each example.
        idx_to_label (dict): A dictionary mapping indices (ints) to labels (strings).

    Returns:
        dict: A dictionary of scores keyed by the labels in `labels` where each score is a 4-tuple
            containing precision, recall, f1 and support. Additionally includes the keys
            'macro avg' and 'micro avg' containing the macro and micro averages across scores.
    """
    scores = {}

    # Unique labels
    labels = list(set(y_true))
    labels.sort()  # ensures labels displayed in same order across runs / partitions

    # Relation classification tasks contain a 'false' label indicating no relation. This is not
    # included in the evaluation metrics.
    if 'false' in set(idx_to_label.values()):
        labels.remove(reverse_dict(idx_to_label)['false'])

    precision, recall, f1, support = \
        precision_recall_fscore_support(y_true, y_pred, labels=labels)
    # Get macro and micro performance metrics averages
    macro_precision, macro_recall, macro_f1, _ = \
        precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro')
    micro_precision, micro_recall, micro_f1, _ = \
        precision_recall_fscore_support(y_true, y_pred, labels=labels, average='micro')

    total_support = 0
    for i, label in enumerate(labels):
        scores[idx_to_label[label]] = precision[i], recall[i], f1[i], support[i]
        total_support += support[i]

    scores['macro avg'] = macro_precision, macro_recall, macro_f1, total_support
    scores['micro avg'] = micro_precision, micro_recall, micro_f1, total_support

    return scores


def evaluate_document_classification(y_true, y_pred, idx_to_label):
    """Performs evaluation for document_classification tasks.

    Args:
        y_true (list): A list containing the gold labels for each example.
        y_pred (list): A list containing the predicted labels for each example.
        idx_to_label (dict): A dictionary mapping indices (ints) to labels (strings).

    Returns:
        dict: A dictionary of scores keyed by the labels in `labels` where each score is a 4-tuple
            containing precision, recall, f1 and support. Additionally includes the keys
            'macro avg' and 'micro avg' containing the macro and micro averages across scores.
    """
    disease_dict = {key: [[], []] for key, value in idx_to_label.items()}
    scores = {'micro': {value: [] for key, value in idx_to_label.items()},
              'macro': {value: [] for key, value in idx_to_label.items()},
              }

    for pred_list, lab_list in zip(y_pred, y_true):
        for i, _ in enumerate(pred_list):
            label = lab_list[i].item()
            pred = pred_list[i].item()

            disease_dict[i][0].append(label)
            disease_dict[i][1].append(pred)

    for disease_idx in disease_dict:
        scores['micro'][idx_to_label[disease_idx]] = precision_recall_fscore_support(
            y_true=disease_dict[disease_idx][0],
            y_pred=disease_dict[disease_idx][1],
            average="micro"
        )
        scores['macro'][idx_to_label[disease_idx]] = precision_recall_fscore_support(
            y_true=disease_dict[disease_idx][0],
            y_pred=disease_dict[disease_idx][1],
            average="macro"
        )

    # Get an arithmetic mean over all diseases and add it to the table
    for avg_type, disease_scores in scores.items():
        avg_precision, average_recall, average_f1 = 0, 0, 0
        for score in disease_scores.values():
            avg_precision += score[0]
            average_recall += score[1]
            average_f1 += score[2]

        avg_precision /= len(disease_scores)
        average_recall /= len(disease_scores)
        average_f1 /= len(disease_scores)

        scores[avg_type]['avg'] = (avg_precision, average_recall, average_f1, None)

    return scores
