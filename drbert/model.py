import logging

import torch
import torch.nn as nn
from pytorch_transformers import BertModel
from pytorch_transformers.modeling_bert import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from .modules.sequence_labelling_head import SequenceLabellingHead

logger = logging.getLogger(__name__)


class BertForJointDeIDAndCohortID(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForJointDeIDAndCohortID, self).__init__(config)

        # Parameters
        self.num_deid_labels = config.num_deid_labels
        self.num_cohort_disease = config.num_cohort_disease  # diabetes, ..., etc.
        self.num_cohort_classes = config.num_cohort_classes  # yes, no, unmentioned, questionable
        self.num_cohort_labels = self.num_cohort_disease * self.num_cohort_classes
        self.max_batch_size = config.max_batch_size
        self.deid_class_weights = torch.as_tensor(config.deid_class_weights)
        self.cohort_class_weights = torch.as_tensor([0.0029, 0.7232, 0.2664, 0.0075])

        # Core BERT model
        self.bert = BertModel(config)
        self.init_weights()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # DeID head
        self.deid_classifier = SequenceLabellingHead(config.hidden_size, config.num_deid_labels)

        # Cohort head
        self.cohort_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.cohort_ffnn_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.cohort_ffnn_size, config.cohort_ffnn_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.cohort_ffnn_size // 2, self.num_cohort_labels)
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, task=None):
        if task not in ['deid', 'cohort'] and task is not None:
            raise ValueError(('Task must be one of "deid", "cohort" or None (for inference).'
                              'Got {task}.'))

        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "head_mask": head_mask
        }

        # If labels are provided, compute a loss for ONE task
        if labels is not None:
            loss_fct_inputs = {'labels': labels}
            if task is None:
                raise ValueError('If labels is not None, task must be one of "deid", "cohort".')
            if task == 'deid':
                outputs = self._deid_forward(**inputs)
                num_labels = (-1, self.num_deid_labels)
                loss_fct_inputs.update({
                    'attention_mask': attention_mask,
                    # 'class_weights': self.deid_class_weights
                })
            elif task == 'cohort':
                outputs = self._cohort_forward(**inputs)
                num_labels = (self.num_cohort_disease, self.num_cohort_classes)
                loss_fct_inputs.update({
                    'class_weights': self.cohort_class_weights
                })

            loss_fct_inputs.update({'logits': outputs[0], 'num_labels': num_labels})
            loss = self.loss_function(**loss_fct_inputs)

            outputs = (loss, outputs[0]) + outputs[2:]
        # Otherwise, we are performing inference
        else:
            if task == 'deid':
                outputs = self._deid_forward(**inputs)
            elif task == 'cohort':
                outputs = self._cohort_forward(**inputs)
            else:
                deid_outputs = self._deid_forward(**inputs)
                cohort_outputs = self._cohort_forward(**inputs)

                deid_logits = deid_outputs[0]
                cohort_logits = cohort_outputs[0]

                outputs = (deid_logits, cohort_logits) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def _deid_forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None,
                      head_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.deid_classifier(sequence_output)

        outputs = (logits, ) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)

    def _cohort_forward(self, input_ids, token_type_ids=None, attention_mask=None,
                        position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        cls_output = outputs[1]

        # Pool outputs
        cls_output = torch.sum(cls_output, dim=0)
        cls_output = self.dropout(cls_output)

        logits = self.cohort_classifier(cls_output)
        logits = logits.view(self.num_cohort_disease, self.num_cohort_classes)  # 16 x 4

        # Get hidden states or attentions or both (if they exist)
        outputs = (logits, ) + (outputs[2:], )

        return outputs  # logits, (hidden_states), (attentions)

    def loss_function(self, logits, labels, num_labels, attention_mask=None, class_weights=None):
        if class_weights is not None:
            class_weights = class_weights.to(logits.device)

        loss_fct = CrossEntropyLoss(weight=class_weights)

        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(*num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(*num_labels), labels.view(-1))

        return loss
