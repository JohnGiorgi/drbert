import logging
import math

import torch
import torch.nn as nn
from pytorch_transformers import BertModel
from pytorch_transformers.modeling_bert import BertPreTrainedModel
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)


class BertForJointDeIDAndCohortID(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForJointDeIDAndCohortID, self).__init__(config)

        # Parameters
        self.num_deid_labels = config.num_deid_labels
        self.num_cohort_disease = config.num_cohort_disease  # diabetes, ..., etc.
        self.num_cohort_classes = config.num_cohort_classes  # yes, no, unmentioned, questionable
        self.num_cohort_labels = self.num_cohort_classes * self.num_cohort_disease
        self.max_batch_size = config.max_batch_size

        # Core BERT model
        self.bert = BertModel(config)
        self.init_weights()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # DeID head
        self.deid_classifier = nn.Linear(config.hidden_size, self.num_deid_labels)

        # Cohort head
        self.cohort_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.cohort_ffnn_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.cohort_ffnn_size, config.cohort_ffnn_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.cohort_ffnn_size // 2, self.num_cohort_disease * self.num_cohort_classes)
        )

        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, task=None):
        if task not in ['deid', 'cohort'] and task is not None:
            raise ValueError(('Task must be one of "deid", "cohort" or None (for inference).'
                              'Got {task}.'))

        outputs = tuple()

        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "head_mask": head_mask
        }

        # Labels are provided, compute a loss for ONE task
        if labels is not None:
            if task is None:
                raise ValueError('If labels is not None, task must be one of "deid", "cohort".')
            if task == 'deid':
                outputs_ = self._deid_forward(**inputs)
                num_labels = (-1, self.num_deid_labels)
            elif task == 'cohort':
                outputs_ = self._cohort_forward(**inputs)
                num_labels = (-1, self.num_cohort_disease * self.num_cohort_classes)

            logits = outputs_[0]
            loss = self.loss_function(logits, labels, num_labels, attention_mask)
            outputs = (loss, logits) + outputs
        # Otherwise, we are performing inference
        else:
            if task == 'deid':
                outputs_ = self._deid_forward(**inputs)
                outputs = (outputs_[0], ) + outputs
            elif task == 'cohort':
                outputs_ = self._cohort_forward(**inputs)
                outputs = (outputs_[0], ) + outputs
            else:
                deid_outputs = self._deid_forward(**inputs)
                cohort_outputs = self._cohort_forward(**inputs)

                deid_logits = deid_outputs[0]
                cohort_logits = cohort_outputs[0]

                outputs = (deid_logits, cohort_logits) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def _deid_forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None,
                      head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.deid_classifier(sequence_output)

        outputs = (logits, ) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # scores, (hidden_states), (attentions)

    def _cohort_forward(self, input_ids, token_type_ids=None, attention_mask=None,
                        position_ids=None, head_mask=None):
        # At test time, batch size simply becomes the number of sentences in the input document
        max_batch_size = self.max_batch_size if self.train else input_ids.size(0)
        num_splits = math.ceil(input_ids.size(0) / max_batch_size)
        sentence_rep_sum = 0

        hidden_states_and_attentions = []

        for split in range(num_splits):
            cur_idx = split * max_batch_size
            next_idx = (split + 1) * max_batch_size
            output = self.bert(
                input_ids[cur_idx: next_idx],
                position_ids=position_ids[cur_idx: next_idx] if position_ids is not None else position_ids,
                token_type_ids=token_type_ids[cur_idx: next_idx] if token_type_ids is not None else token_type_ids,
                attention_mask=attention_mask[cur_idx: next_idx] if attention_mask is not None else attention_mask,
                head_mask=head_mask[cur_idx: next_idx] if head_mask is not None else head_mask
            )
            sentence_rep_sum += output[0][:, 0]

            hidden_states_and_attentions.append(output[2:])

        sentence_rep_sum = self.dropout(sentence_rep_sum)

        logits = self.cohort_classifier(sentence_rep_sum)
        outputs = (logits.view(-1, self.num_cohort_classes), )  # logits: batch size * 16 x 4

        # Get hidden states or attentions or both (if they exist)
        for i in range(len(hidden_states_and_attentions[0])):
            outputs = outputs + (torch.cat([x[i] for x in hidden_states_and_attentions]), )

        return outputs  # scores, (hidden_states), (attentions)

    def loss_function(self, logits, labels, num_labels, attention_mask=None):
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(*num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fct(active_logits, active_labels)
        else:
            loss = self.loss_fct(logits.view(*num_labels), labels.view(-1))

        return loss
