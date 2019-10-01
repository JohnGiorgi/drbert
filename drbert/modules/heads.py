import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class SequenceLabellingHead(torch.nn.Module):
    """A head which can be placed at the output of a language model (such as BERT) to perform
    document classification tasks.

    # TODO: Finish this docstring.

    Args:
        config (TODO): TODO

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        - loss: (optional, returned when `labels` is provided) `torch.FloatTensor` of shape `(1,)`:
            Classification loss.
        - scores: `torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`
            Classification scores (before SoftMax).
        - hidden_states: (`optional`, returned when `config.output_hidden_states=True`)
            list of `torch.FloatTensor` (one for the output of each layer + the output of the embeddings)
            of shape `(batch_size, sequence_length, hidden_size)`:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        - attentions: (`optional`, returned when `config.output_attentions=True`)
            list of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples:
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> model = SequenceLabellingHead()
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids, labels=labels)
        >>> loss, scores = outputs[:2]
    """
    def __init__(self, config):
        super(SequenceLabellingHead, self).__init__()
        # TODO (John): This will eventually be decoupled from the deids task
        self.num_labels = config.num_deid_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_deid_labels)

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
                loss = loss_fct(logits.view(-1, self.self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class DocumentClassificationHead(torch.nn.Module):
    """A head which can be placed at the output of a language model (such as BERT) to perform
    document classification tasks.

    # TODO: Finish this docstring.

    Args:
        config (todo): todo
        out_features (int): The size of the feature dimension of the output.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        - loss: (optional, returned when `labels` is provided) `torch.FloatTensor` of shape `(1,)`:
            Classification loss.
        - scores: `torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`
            Classification scores (before SoftMax).
        - hidden_states: (`optional`, returned when `config.output_hidden_states=True`)
            list of `torch.FloatTensor` (one for the output of each layer + the output of the embeddings)
            of shape `(batch_size, sequence_length, hidden_size)`:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        - attentions: (`optional`, returned when `config.output_attentions=True`)
            list of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples:
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> labels = torch.tensor(torch.ones()).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids, labels=labels)
        >>> loss, scores = outputs[:2]
    """
    def __init__(self, config):
        super(DocumentClassificationHead, self).__init__()
        # TODO (John): This will eventually be decoupled from the cohort task
        # For non-multi label datasets, num_classes will be 1
        self.num_labels = config.num_cohort_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.cohort_ffnn_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.cohort_ffnn_size, config.cohort_ffnn_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.cohort_ffnn_size // 2, self.num_labels[0] * self.num_labels[1])
        )

    def forward(self, bert, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels=None):
        outputs = bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask)
        cls_output = outputs[1]

        # Pool outputs
        cls_output = torch.sum(cls_output, dim=0)
        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)
        logits = logits.view(self.num_labels[0] * self.num_labels[1])

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.view(*self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
