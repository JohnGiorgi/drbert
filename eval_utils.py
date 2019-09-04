from statistics import mean

from constants import OUTSIDE
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities


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
    assert len(y_true) == len(y_pred)
    num_correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            num_correct += 1
    return num_correct / len(y_true)



