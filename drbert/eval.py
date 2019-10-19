import logging

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import SequentialSampler
from tqdm import tqdm

from .utils import eval_utils
from .constants import OUTSIDE
from .constants import TASKS
from .constants import COHORT_DISEASE_CONSTANTS
from .constants import DEID_LABELS

logger = logging.getLogger(__name__)


# TODO (John): Find a way to remove args
def evaluate(args, model, dataset, task):
    """The interface for evaluating DrBERT.

    This function handles evaluation of DrBERT (`model`) for a given task (`task`).

    Args:
        args ([type]): [description]
        model ([type]): [description]
        dataset ([type]): [description]
        task (str, optional): [description]. Defaults to "sequence_labelling".

    Returns:
        [type]: [description]
    """
    if task.lower() not in TASKS:
        err_msg = f"'task' must be one of {TASKS}. Got: {task}"
        logger.error('ValueError: %s', err_msg)
        raise ValueError(err_msg)

    eval_dataloader = {}

    # bs 1 for document classification tasks because sentences are grouped by document
    batch_size = 1 if task == 'document_classification' else args.eval_batch_size

    for partition in dataset:
        sampler = (SequentialSampler(dataset[partition]) if args.local_rank == -1
                   # Note that DistributedSampler samples randomly
                   else DistributedSampler(dataset[partition]))

        eval_dataloader[partition] = \
            DataLoader(dataset[partition], sampler=sampler, batch_size=batch_size)

    evaluations = {}
    model.eval()

    for partition, dataloader in eval_dataloader.items():
        logger.info(f"***** Running {task} evaluation on {partition} *****")
        logger.info("  Num examples = %d", len(dataset[partition]))
        logger.info("  Batch size = %d", args.eval_batch_size)

        labels, predictions, orig_tok_mask = [], [], []

        for batch in tqdm(dataloader, desc="Evaluating"):
            # Dataloader introduces a first dimension of size one
            if task == 'document_classification':
                batch[0] = batch[0].squeeze(0)
                batch[1] = batch[1].squeeze(0)
                batch[2] = batch[2].squeeze(0)

            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[2],
                          'task':           task
                          }

                outputs = model(**inputs)
                loss, logits = outputs[:2]

                # Need to accumulate these for evaluation
                labels.append(inputs['labels'])
                predictions.append(logits.argmax(dim=-1))
                if task == 'sequence_labelling':
                    orig_tok_mask.append(batch[3])

        if task == 'sequence_labelling':
            labels = torch.cat(labels)
            predictions = torch.cat(predictions)
            orig_tok_mask = torch.cat(orig_tok_mask)

            evaluation = evaluate_sequence_labelling(labels, predictions, orig_tok_mask)
            eval_utils.print_evaluation(evaluation, title=partition.title())
        elif task == 'document_classification':
            evaluation = evaluate_document_classification(labels, predictions)

            # For document_classification, print both macro and micro averages per disease
            for avg_type in evaluation:
                eval_utils.print_evaluation(
                    evaluation=evaluation[avg_type],
                    title=f'{partition.title()} ({avg_type})'
                )

        evaluations[partition] = evaluation

        del batch, inputs, outputs

    return evaluations


def evaluate_sequence_labelling(labels, predictions, orig_tok_mask):
    """Performs evaluation for sequence labelling tasks.

    Args:
        labels (list): A list of Tensors containing the gold labels for each example.
        predictions (list): A list of Tensors containing the predicted labels for each example.

    Returns:
        dict: A dictionary of scores keyed by the labels in `labels` where each score is a 4-tuple
            containing precision, recall, f1 and support. Additionally includes the keys
            'Macro avg' and 'Micro avg' containing the macro and micro averages across scores.
    """
    idx_to_tag = eval_utils.reverse_dict(DEID_LABELS)

    y_true = torch.masked_select(labels, orig_tok_mask)
    y_pred = torch.masked_select(predictions, orig_tok_mask)

    # Map predictions to tags
    y_true = [idx_to_tag[idx.item()] for idx in y_true]
    y_pred = [idx_to_tag[idx.item()] for idx in y_pred]

    scores = eval_utils.precision_recall_f1_support_sequence_labelling(y_true, y_pred)

    # Add binary F1 scores
    y_true_binary = [tag if tag == OUTSIDE else f'{tag.split("-")[0]}-binary' for tag in y_true]
    y_pred_binary = [tag if tag == OUTSIDE else f'{tag.split("-")[0]}-binary' for tag in y_pred]

    scores['binary'] = \
        eval_utils.precision_recall_f1_support_sequence_labelling(y_true_binary, y_pred_binary)['binary']

    return scores


def evaluate_document_classification(labels, predictions):
    """Coordinates evaluation for the document_classification identification task.
    Args:
        labels (list): A list of Tensors containing the gold labels for each example.
        predictions (list): A list of Tensors containing the predicted labels for each example.
        average (flag): Set micro or macro mode of eval
    Returns:
        scores (dict): A dictionairy of dictionaries of diseases and their scores
            (precision, recall, F1, support) using sklearn precision_recall_fscore_support
    """
    idx_to_tag = eval_utils.reverse_dict(COHORT_DISEASE_CONSTANTS)

    disease_dict = {key: [[], []] for key, value in idx_to_tag.items()}
    scores = {'micro': {value: [] for key, value in idx_to_tag.items()},
              'macro': {value: [] for key, value in idx_to_tag.items()},
              }

    for pred_list, lab_list in zip(predictions, labels):
        for i, _ in enumerate(pred_list):
            label = lab_list[i].item()
            pred = pred_list[i].item()

            disease_dict[i][0].append(label)
            disease_dict[i][1].append(pred)

    for disease_idx in disease_dict:
        scores['micro'][idx_to_tag[disease_idx]] = precision_recall_fscore_support(
            y_true=disease_dict[disease_idx][0],
            y_pred=disease_dict[disease_idx][1],
            average="micro"
        )
        scores['macro'][idx_to_tag[disease_idx]] = precision_recall_fscore_support(
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
