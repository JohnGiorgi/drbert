class SequencePairClassificationHead(torch.nn.Module):
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
        super(SequencePairClassificationHead, self).__init__()
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels)

    def forward(self, bert, sentence1, sentence2, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        # TODO (John): Need attention masks for each sentence
        attention_mask = torch.where(sentence1 == 0, torch.zeros_like(sentence1), torch.ones_like(sentence1))
        sentence1_outputs = bert(sentence1, attention_mask, token_type_ids, position_ids, head_mask)
        attention_mask = attention_mask.unsqueeze(-1).expand(sentence1_outputs[0].size()).float()
        sentence1_pooled_output = torch.mean(sentence1_outputs[0] * attention_mask, dim=1)

        attention_mask = torch.where(sentence2 == 0, torch.zeros_like(sentence2), torch.ones_like(sentence2))
        sentence2_outputs = bert(sentence2, attention_mask, token_type_ids, position_ids, head_mask)
        attention_mask = attention_mask.unsqueeze(-1).expand(sentence2_outputs[0].size()).float()
        sentence2_pooled_output = torch.mean(sentence2_outputs[0] * attention_mask, dim=1)

        elem_wise_diff = torch.abs(sentence1_pooled_output - sentence2_pooled_output)
        pooled_output = torch.cat((sentence1_pooled_output, sentence2_pooled_output, elem_wise_diff), dim=-1)

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Add hidden states and attention if they are here
        # TODO (John): Find a way to add this back in
        # outputs = (sentence_1_pooled_output, sentence_2_pooled_output) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)