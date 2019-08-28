import torch.nn as nn
from torch.nn import CrossEntropyLoss

from pytorch_transformers import BertModel, BertPreTrainedModel


class BertForJointDeIDAndCohortID(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForJointDeIDAndCohortID, self).__init__(config)

        # Parameters
        self.num_deid_labels = config.num_deid_labels
        self.num_cohort_labels = config.cohort_num_disease * config.cohort_num_classes

        # Core BERT model
        self.bert = BertModel(config)
        self.apply(self.init_weights)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # DeID Head
        self.deid_classifier = nn.Linear(config.hidden_size, config.max_seq_len * self.num_deid_labels)

        # Cohort head
        self.cohort_ffnn = nn.Sequential(
            nn.Linear(config.hidden_size, config.cohort_ffnn_size),
            nn.LayerNorm(config.cohort_ffnn_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.cohort_ffnn_size, config.cohort_ffnn_size // 2),
            nn.LayerNorm(config.cohort_ffnn_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        

        self.cohort_classifier = nn.Linear(config.cohort_ffnn_size // 2, self.num_cohort_labels)

        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, task='deid'):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        if task == 'deid':
            num_labels = (-1, self.num_deid_labels)
            logits = self.deid_classifier(sequence_output)
        elif task == 'cohort':
            num_labels = (-1, self.num_cohort_labels)
            sequence_output = self.cohort_ffnn(sequence_output)
            logits = self.cohort_classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(*num_labels)[active_loss]
                # TODO: Will this work for cohort task?

                [0.1, 0.1, 0.8] [2] 
                [0.1, 0.8, 0.1] [1] 1 x 16
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(*num_labels), labels.view(-1))
            outputs = (loss,) + outputs
