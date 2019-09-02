import torch.nn as nn
from torch.nn import CrossEntropyLoss

from pytorch_transformers import BertModel, BertPreTrainedModel


class BertForJointDeIDAndCohortID(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForJointDeIDAndCohortID, self).__init__(config)

        # Parameters
        self.num_deid_labels = config.num_deid_labels
        self.num_cohort_labels = config.num_cohort_disease * config.num_cohort_classes
        self.max_batch_size = config.max_batch_size

        # Core BERT model
        self.bert = BertModel(config)
        self.apply(self.init_weights)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # DeID Head
        self.deid_classifier = nn.Linear(config.hidden_size, self.num_deid_labels)

        # Cohort head
        self.cohort_ffnn = nn.Sequential(
            nn.Linear(config.hidden_size, config.cohort_ffnn_size),
            nn.LayerNorm(config.cohort_ffnn_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.cohort_ffnn_size, config.cohort_ffnn_size // 2),
            nn.LayerNorm(config.cohort_ffnn_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.cohort_classifier = nn.Linear(config.cohort_ffnn_size // 2, self.num_cohort_labels)

        self.loss_fct = CrossEntropyLoss()

    def split_batch_forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_splits = 1 + (input_ids.size(0) // self.max_match_size)
        bert_output_list = list()
        for split in range(num_splits):
            outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask, head_mask=head_mask)
            bert_output_list.append(outputs)
        return bert_output_list

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, task=None):

        bert_output_list = self.split_batch_forward(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                position_ids=position_ids, head_mask=head_mask)

        total_loss = 0
        deid_pred = None
        cohort_pred = None

        if task == 'deid':
            deid_loss, deid_pred = self._deid_forward(bert_output_list, labels)
            total_loss += deid_loss
        elif task == 'cohort':
            cohort_loss, cohort_pred = self._cohort_forward(bert_output_list, labels)
            total_loss += cohort_loss
        else:
            _, deid_pred = self._deid_forward(bert_output_list, labels)
            _, cohort_pred = self._cohort_forward(bert_output_list, labels)

        return total_loss, deid_pred, cohort_pred

    def _deid_forward(self, bert_output_list):
        sequence_output_list = list()
        for i in range(len(bert_output_list)):
            sequence_output_list.append(bert_output_list[i][0])
        sequence_output = torch.cat(sequence_output_list, dim=0)

        sequence_output = self.dropout(sequence_output)

        num_labels = (-1, self.num_deid_labels)
        logits = self.deid_classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(*num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(*num_labels), labels.view(-1))
            outputs = (loss,) + outputs

    def _cohort_forward(self, bert_outputs):
        sequence_output_list = list()
        for i in range(len(bert_output_list)):
            sequence_output_list.append(bert_output_list[i][0])
        sequence_output = torch.cat(sequence_output_list, dim=0)

        sentence_rep_sum = sequence_output.sum(0)

        sequence_output = self.dropout(sequence_output)

        num_labels = (-1, self.num_cohort_labels)
        sequence_output = self.cohort_ffnn(sequence_output)
        logits = self.cohort_classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(*num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(*num_labels), labels.view(-1))
            outputs = (loss,) + outputs
