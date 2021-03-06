import json
import os
from statistics import mean

from prettytable import PrettyTable

from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities

from ..constants import OUTSIDE


def reverse_dict(mapping):
    """Returns a dictionary composed of the reverse v, k pairs of a dictionary `mapping`.
    """
    return {v: k for k, v in mapping.items()}


def precision_recall_f1_support_sequence_labelling(y_true, y_pred):
    """Compute precision, recall, f1 and support for sequence labelling tasks.

    For given gold (`y_true`) and predicted (`y_pred`) sequence labels, returns the precision,
    recall, f1 and support per label, and the macro and micro average of these scores across
    labels. Expects `y_true` and `y_pred` to be a sequence of IOB1/2, IOE1/2, or IOBES formatted
    labels.

    Args:
        y_true (list): List of IOB1/2, IOE1/2, or IOBES formatted sequence labels.
        y_pred (list): List of IOB1/2, IOE1/2, or IOBES formatted sequence labels.

    Returns:
        A dictionary of scores keyed by the labels in `y_true` where each score is a 4-tuple
        containing precision, recall, f1 and support. Additionally includes the keys
        'Macro avg' and 'Micro avg' containing the macro and micro averages across scores.
    """
    scores = {}
    # Unique labels, not including NEG
    labels = list({tag.split('-')[-1] for tag in set(y_true) if tag != OUTSIDE})
    labels.sort()  # ensures labels displayed in same order across runs / partitions

    for label in labels:
        y_true_lab = [tag if tag.endswith(label) else OUTSIDE for tag in y_true]
        y_pred_lab = [tag if tag.endswith(label) else OUTSIDE for tag in y_pred]

        # TODO (John): Open a pull request to seqeval with a new function that returns all these
        # scores in one call. There is a lot of repeated computation here.
        precision = precision_score(y_true_lab, y_pred_lab)
        recall = recall_score(y_true_lab, y_pred_lab)
        f1 = f1_score(y_true_lab, y_pred_lab)
        support = len(set(get_entities(y_true_lab)))

        scores[label] = precision, recall, f1, support

    # Get macro and micro performance metrics averages
    macro_precision = mean([v[0] for v in scores.values()])
    macro_recall = mean([v[1] for v in scores.values()])
    macro_f1 = mean([v[2] for v in scores.values()])
    total_support = sum([v[3] for v in scores.values()])

    micro_precision = precision_score(y_true, y_pred)
    micro_recall = recall_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred)

    scores['Macro avg'] = macro_precision, macro_recall, macro_f1, total_support
    scores['Micro avg'] = micro_precision, micro_recall, micro_f1, total_support

    return scores


def classification_accuracy(y_true, y_pred):
    num_correct = 0
    for i, _ in enumerate(y_true):
        if y_true[i] == y_pred[i]:
            num_correct += 1
    return num_correct / len(y_true)


def print_evaluation(evaluation, title=None):
    """Prints an ASCII table of evaluation scores.

    Args:
        evaluation: A dictionary of label, score pairs where label is a class tag and
            scores is a 4-tuple containing precision, recall, f1 and support.
        title (str): Optional, the title of the table.

    Preconditions:
        Assumes the values of `evaluation` are 4-tuples, where the first three items are
        float representaions of a percentage and the last item is an count integer.
    """
    # Create table, give it a title a column names
    table = PrettyTable()

    if title is not None:
        table.title = title

    table.field_names = ['Label', 'Precision', 'Recall', 'F1', 'Support']

    # Column alignment
    table.align['Label'] = 'l'
    table.align['Precision'] = 'r'
    table.align['Recall'] = 'r'
    table.align['F1'] = 'r'
    table.align['Support'] = 'r'

    # Create and add the rows
    for label, scores in evaluation.items():
        row = [label]
        # convert scores to formatted percentage strings
        support = scores[-1]
        performance_metrics = [f'{x:.2%}' for x in scores[:-1]]
        row_scores = performance_metrics + [support]

        row.extend(row_scores)
        table.add_row(row)

    print(table)

    return table


def save_eval_to_disk(args, step, **kwargs):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    evaluation = {'train': {}, 'valid': {}, 'test': {}}
    for partition in evaluation:
        for task, results in kwargs.items():
            if partition in results:
                evaluation[partition][task] = results[partition]

    with open(os.path.join(args.output_dir, f'evaluation_{step}.json'), 'w') as f:
        json.dump(evaluation, f, indent=2)
