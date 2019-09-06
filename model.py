import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from pytorch_transformers import BertModel
from pytorch_transformers.modeling_bert import BertPreTrainedModel

from constants import *

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
        num_splits = math.ceil(input_ids.size(0) / self.max_batch_size)
        bert_output_list = list()
        for split in range(num_splits):
            cur_idx = split * self.max_batch_size
            next_idx = (split + 1) * self.max_batch_size
            outputs = self.bert(input_ids[cur_idx: next_idx], 
                position_ids=position_ids[cur_idx: next_idx] if position_ids is not None else position_ids, 
                token_type_ids=token_type_ids[cur_idx: next_idx] if token_type_ids is not None else token_type_ids,
                attention_mask=attention_mask[cur_idx: next_idx] if attention_mask is not None else attention_mask, 
                head_mask=head_mask[cur_idx: next_idx] if head_mask is not None else head_mask)
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
            deid_loss, deid_pred = self._deid_forward(bert_output_list, attention_mask, labels)
            total_loss += deid_loss
        elif task == 'cohort':
            cohort_loss, cohort_pred = self._cohort_forward(bert_output_list, attention_mask, labels)
            total_loss += cohort_loss
        else:
            _, deid_pred = self._deid_forward(bert_output_list, labels)
            _, cohort_pred = self._cohort_forward(bert_output_list, labels)

        return total_loss, deid_pred, cohort_pred

    def _deid_forward(self, bert_output_list, attention_mask = None, labels = None):
        sequence_output_list = list()
        for i in range(len(bert_output_list)):
            sequence_output_list.append(bert_output_list[i][0])
        sequence_output = torch.cat(sequence_output_list, dim=0)
        sequence_output = self.dropout(sequence_output)

        num_labels = (-1, self.num_deid_labels)
        logits = self.deid_classifier(sequence_output)

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss = 0
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(*num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(*num_labels), labels.view(-1))

        return loss, logits


    def _cohort_forward(self, bert_output_list, attention_mask, labels):
        sequence_output_list = list()
        for i in range(len(bert_output_list)):
            sequence_output_list.append(bert_output_list[i][0])
        sequence_output = torch.cat(sequence_output_list, dim=0)

        sentence_rep_sum = sequence_output[0].sum() # add the CLS tokens
        print(f"sentence_rep_sum: {sentence_rep_sum.shape}")
        # sentence_rep_sum: 1 x 768

        sentence_rep_sum = self.dropout(sentence_rep_sum)

        num_labels = (-1, self.num_cohort_labels)
        sentence_rep_sum = self.cohort_ffnn(sentence_rep_sum)
        logits = self.cohort_classifier(sentence_rep_sum)

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss = 0
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(*num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(*num_labels), labels.view(-1))

        return loss, logits


if __name__ == "__main__":
    from pytorch_transformers import BertTokenizer
    from pytorch_transformers import BertConfig
    import data_utils
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", default=None, type=str, required=True,
                        help="De-id and co-hort identification data directory")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="train batch size")
    args = parser.parse_args()

    print(args)
    
    bert_model = "bert-base-uncased"
    config = BertConfig.from_pretrained(bert_model)
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    # Process the data
    deid_dataset = data_utils.prepare_deid_dataset(args, tokenizer)
    cohort_dataset = data_utils.prepare_cohort_dataset(args, tokenizer)
    print('loaded datasets')

    # TODO: Come up with a much better scheme for this
    config.__dict__['num_deid_labels'] = len(DEID_LABELS)
    config.__dict__['num_cohort_disease'] = len(COHORT_LABEL_CONSTANTS)
    config.__dict__['num_cohort_classes'] = len(COHORT_DISEASE_LIST)
    config.__dict__['cohort_ffnn_size'] = 128
    config.__dict__['max_batch_size'] = args.train_batch_size

    model = BertForJointDeIDAndCohortID.from_pretrained(
        pretrained_model_name_or_path=bert_model,
        config=config
    )
    print('created model')

    deid_train = deid_dataset['train']
    cohort_train = cohort_dataset['train']

    print("starting deid")
    model.eval()
    model.cuda()

    for param in model.parameters():
        param.requires_grad = False

    for i, (input_ids, attn_mask, orig_tok_map, labels) in enumerate(deid_train):
        print(i)
        input_ids = input_ids.unsqueeze(0).cuda()
        attn_mask = attn_mask.unsqueeze(0).cuda()
        orig_tok_map = orig_tok_map.unsqueeze(0).cuda()
        labels = labels.unsqueeze(0).cuda()
        print(f"input_ids: {input_ids.shape}")
        print(f"attn_mask: {attn_mask.shape}")
        print(f"orig_tok_map: {orig_tok_map.shape}")
        print(f"labels: {labels.shape}")

        total_loss, deid_pred, cohort_pred = model(input_ids, attention_mask=attn_mask, labels=labels, task="deid")
        print(f"total_loss: {total_loss}")
        print(f"deid_pred: {deid_pred}")
        print(f"cohort_pred: {cohort_pred}")

        if i > 2:
            break

    print("starting cohort")
    for i, (input_ids, attn_mask, labels) in enumerate(cohort_train):
        print(i)
        input_ids = input_ids.cuda()
        attn_mask = attn_mask.cuda()
        labels = labels.cuda()
        print(f"input_ids: {input_ids.shape}")
        print(f"attn_mask: {attn_mask.shape}")
        print(f"labels: {labels.shape}")
        input_ids = input_ids[:1]
        attn_mask = attn_mask[:1]
        labels = labels[:1]

        total_loss, deid_pred, cohort_pred = model(input_ids, attention_mask=attn_mask, labels=labels, task="cohort")
        print(f"total_loss: {total_loss}")
        print(f"deid_pred: {deid_pred}")
        print(f"cohort_pred: {cohort_pred}")

        if i > 2:
            break