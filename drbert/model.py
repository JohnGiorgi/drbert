import logging

from transformers import BertModel
from transformers.modeling_bert import BertPreTrainedModel

from .heads import DocumentClassificationHead
from .heads import SequenceLabellingHead

logger = logging.getLogger(__name__)


class DrBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(DrBERT, self).__init__(config)

        # TODO (John): This will eventually be decoupled from the deid/cohort tasks
        self.num_deid_labels = config.num_deid_labels
        # Document classification labels can be a tuple (for multi-label) or integer (multi-class).
        self.num_cohort_labels = \
            (config.num_cohort_labels if isinstance(config.num_cohort_labels, (list, tuple))
             else (1, config.num_cohort_labels))

        # Core BERT model
        self.bert = BertModel(config)
        self.init_weights()

        # TODO (John): In the future, heads will be defined within a configuration file
        # passed to the constructor of this object.
        # DeID head
        self.deid_head = SequenceLabellingHead(config)
        # Cohort head
        self.cohort_head = DocumentClassificationHead(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels=None, task=None):
        if task not in ['deid', 'cohort'] and task is not None:
            raise ValueError(('Task must be one of "deid", "cohort" or None (for inference).'
                              'Got {task}.'))

        inputs = {
            "bert": self.bert,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "labels": labels,
        }

        # TODO (John): In the future, a given forward pass will trigger forward passes for each
        # and every head, unless otherwise specified.
        # If labels are provided, compute a loss for ONE task
        if labels is not None:
            if task is None:
                raise ValueError('If labels is not None, task must be one of "deid", "cohort".')
            if task == 'deid':
                outputs = self.deid_head(**inputs)
            elif task == 'cohort':
                outputs = self.cohort_head(**inputs)
        # Otherwise, we are performing inference
        else:
            if task == 'deid':
                outputs = self.deid_head(**inputs)
            elif task == 'cohort':
                outputs = self.cohort_head(**inputs)
            else:
                deid_outputs = self.deid_head(**inputs)
                cohort_outputs = self.cohort_head(**inputs)

                deid_logits = deid_outputs[0]
                cohort_logits = cohort_outputs[0]

                # TODO (John): This looks wrong? Logits should come second.
                outputs = (deid_logits, cohort_logits) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
