import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss


class SequenceLabellingHead(torch.nn.Module):
    """A head which can be placed at the output of a language model (such as BERT) to perform
    sequence labelling tasks (e.g. de-identification or NER).

    Args:
        config (BertConfig): `BertConfig` class instance with a configuration to build a new model.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        - loss: (optional, returned when `labels` is provided) `torch.FloatTensor` of shape `(1,)`:
            Classification loss.
        - scores: `torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`
            Classification scores (before SoftMax).
        - hidden_states: (`optional`, returned when `config.output_hidden_states=True`)
            list of `torch.FloatTensor` (one for the output of each layer + the output of the
            embeddings) of shape `(batch_size, sequence_length, hidden_size)`:
            Hidden-states of the model at the output of each layer plus the initial embedding
            outputs.
        - attentions: (`optional`, returned when `config.output_attentions=True`)
            list of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`: Attentions weights after the attention softmax, used
            to compute the weighted average in the self-attention heads.

    Examples:
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        >>> config = AutoConfig.from_pretrained('bert-base-uncased')
        >>> bert = AutoModel.from_pretrained('bert-base-uncased')
        >>> head = SequenceLabellingHead(config)
        >>> input_ids = tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)
        >>> input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1
        >>> labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)
        >>> outputs = head(bert, input_ids, labels=labels)
        >>> loss, scores = outputs[:2]
    """
    def __init__(self, config, num_labels):
        super(SequenceLabellingHead, self).__init__()
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, bert, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels=None):
        outputs = bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class SequenceClassificationHead(torch.nn.Module):
    """A head which can be placed at the output of a language model (such as BERT) to perform
    sequence classification tasks (e.g. natural language inference or relation extraction).

    Args:
        config (BertConfig): `BertConfig` class instance with a configuration to build a new model.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        - loss: (optional, returned when `labels` is provided) `torch.FloatTensor` of shape `(1,)`:
            Classification loss.
        - scores: `torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`
            Classification scores (before SoftMax).
        - hidden_states: (`optional`, returned when `config.output_hidden_states=True`)
            list of `torch.FloatTensor` (one for the output of each layer + the output of the
            embeddings) of shape `(batch_size, sequence_length, hidden_size)`: Hidden-states of the
            model at the output of each layer plus the initial embedding outputs.
        - attentions: (`optional`, returned when `config.output_attentions=True`)
            list of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`: Attentions weights after the attention softmax, used
            to compute the weighted average in the self-attention heads.

    Examples:
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        >>> config = AutoConfig.from_pretrained('bert-base-uncased')
        >>> bert = AutoModel.from_pretrained('bert-base-uncased')
        >>> head = SequenceClassificationHead(config)
        >>> input_ids = tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)
        >>> input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1
        >>> labels = torch.tensor([1]).unsqueeze(0)
        >>> outputs = head(bert, input_ids, labels=labels)
        >>> loss, logits = outputs[:2]
    """
    def __init__(self, config, num_labels):
        super(SequenceClassificationHead, self).__init__()
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, bert, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels=None):

        outputs = bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class DocumentClassificationHead(torch.nn.Module):
    """A head which can be placed at the output of a language model (such as BERT) to perform
    document classification tasks.

    Args:
        config (BertConfig): `BertConfig` class instance with a configuration to build a new model.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        - loss: (optional, returned when `labels` is provided) `torch.FloatTensor` of shape `(1,)`:
            Classification loss.
        - scores: `torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`
            Classification scores (before SoftMax).
        - hidden_states: (`optional`, returned when `config.output_hidden_states=True`)
            list of `torch.FloatTensor` (one for the output of each layer + the output of the
            embeddings) of shape `(batch_size, sequence_length, hidden_size)`: Hidden-states of the
            model at the output of each layer plus the initial embedding outputs.
        - attentions: (`optional`, returned when `config.output_attentions=True`)
            list of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`: Attentions weights after the attention softmax, used
            to compute the weighted average in the self-attention heads.

    Examples:
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        >>> config = AutoConfig.from_pretrained('bert-base-uncased')
        >>> bert = AutoModel.from_pretrained('bert-base-uncased')
        >>> head = DocumentClassificationHead(config)
        >>> input_ids = tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)
        >>> input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1
        >>> labels = torch.ones(0, 4, (16, 4))  # E.g. 16 diseases, with 4 classes each.
        >>> outputs = head(bert, input_ids, labels=labels)
        >>> loss, logits = outputs[:2]
    """
    def __init__(self, config, num_labels):
        super(DocumentClassificationHead, self).__init__()
        # For non-multi label datasets, num_classes[0] will be 1
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.cohort_ffnn_size,
            num_layers=2,
            batch_first=True,
            dropout=config.hidden_dropout_prob,
            bidirectional=True,
        )

        self.classifier = nn.Linear(config.cohort_ffnn_size * 2, num_labels[0] * num_labels[1])

    def forward(self, bert, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels=None):
        outputs = bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask)

        pooled_outputs = outputs[1]
        pooled_outputs = self.dropout(pooled_outputs)

        reccurent_outputs, _ = self.lstm(pooled_outputs.unsqueeze(0))  # batch dim of 1
        reccurent_outputs = reccurent_outputs[:, -1, :].squeeze(0)  # hidden state of final timestep

        logits = self.classifier(reccurent_outputs)
        logits = logits.view(self.num_labels)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels[0] == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
