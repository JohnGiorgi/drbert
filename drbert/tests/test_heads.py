import torch
from torch import nn
from pytorch_transformers import BertModel
from random import randint

VOCAB = 30000  # BERT has a vocab of ~30K

# TODO (John): These tests are very shallow.


class TestSequenceLabellingHead(object):
    """Collects all unit tests for `drbert.heads.SequenceLabellingHead`.
    """
    def test_attributes_after_initialization(self, bert_config, sequence_labelling_head):
        _, sequence_length, num_labels, head = sequence_labelling_head

        assert head.num_labels == num_labels
        assert isinstance(head.dropout, (nn.Module, nn.Dropout))
        assert isinstance(head.classifier, (nn.Module, nn.Linear))

    def test_forward_pass_input_ids_only(self, bert, sequence_labelling_head):
        batch_size, sequence_length, num_labels, head = sequence_labelling_head

        input_ids = torch.LongTensor(batch_size, sequence_length).random_(0, VOCAB)

        outputs = head(bert, input_ids)

        assert len(outputs) == 1
        assert outputs[0].size() == (batch_size, sequence_length, num_labels)

    def test_forward_pass_with_attn_masks(self, bert, sequence_labelling_head):
        batch_size, sequence_length, num_labels, head = sequence_labelling_head

        input_ids = torch.randint(VOCAB, (batch_size, sequence_length))
        attention_mask = torch.randint(2, (batch_size, sequence_length))

        outputs = head(bert, input_ids, attention_mask=attention_mask)

        assert len(outputs) == 1
        assert outputs[0].size() == (batch_size, sequence_length, num_labels)

    def test_forward_pass_with_token_type_ids(self, bert, sequence_labelling_head):
        batch_size, sequence_length, num_labels, head = sequence_labelling_head

        input_ids = torch.randint(VOCAB, (batch_size, sequence_length))
        token_type_ids = torch.ones_like(input_ids)

        outputs = head(bert, input_ids, token_type_ids=token_type_ids)

        assert len(outputs) == 1
        assert outputs[0].size() == (batch_size, sequence_length, num_labels)

    def test_forward_pass_with_position_ids(self, bert, sequence_labelling_head):
        batch_size, sequence_length, num_labels, head = sequence_labelling_head

        input_ids = torch.randint(VOCAB, (batch_size, sequence_length))
        position_ids = torch.arange(sequence_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        outputs = head(bert, input_ids, position_ids=position_ids)

        assert len(outputs) == 1
        assert outputs[0].size() == (batch_size, sequence_length, num_labels)

    def test_forward_pass_with_head_mask(self, bert_config, sequence_labelling_head):
        # We need all hidden states to confirm the head masking worked as expected
        bert_config.output_attentions = True
        bert = BertModel(bert_config)

        batch_size, sequence_length, num_labels, head = sequence_labelling_head

        input_ids = torch.randint(VOCAB, (batch_size, sequence_length))
        head_mask = torch.ones(bert_config.num_attention_heads)
        # Turn on one head off random
        masked_head = randint(0, head_mask.size(-1))
        head_mask[masked_head] = 0

        outputs = head(bert, input_ids, head_mask=head_mask)

        assert len(outputs) == 2
        assert outputs[0].size() == (batch_size, sequence_length, num_labels)

        # Asserts that attention weights are of the expected size, and that one of the attention
        # heads was masked.
        output_shape = \
            (batch_size, bert_config.num_attention_heads, sequence_length, sequence_length)
        for attention_weights in outputs[1]:
            assert attention_weights.size() == output_shape
            for head, weights in enumerate(attention_weights):
                if head == masked_head:
                    assert torch.equal(weights, torch.zeros_like(weights))
                else:
                    assert not torch.equal(weights, torch.zeros_like(weights))

    def test_forward_pass_with_labels(self, bert, sequence_labelling_head):
        batch_size, sequence_length, num_labels, head = sequence_labelling_head

        input_ids = torch.randint(VOCAB, (batch_size, sequence_length))
        labels = torch.randint(0, num_labels, (batch_size, sequence_length))

        outputs = head(bert, input_ids, labels=labels)

        assert len(outputs) == 2
        assert outputs[0].size() == torch.Size([])  # Loss is a single element tensor
        assert outputs[1].size() == (batch_size, sequence_length, num_labels)


class TestDocumentClassificationHead(object):
    """Collects all unit tests for `drbert.heads.DocumentClassificationHead`.
    """
    def test_attributes_after_initialization(self, bert_config, sequence_labelling_head):
        _, sequence_length, num_labels, head = sequence_labelling_head

        assert head.num_labels == num_labels
        assert isinstance(head.dropout, (nn.Module, nn.Dropout))
        assert isinstance(head.classifier, (nn.Module, nn.Linear))

    def test_forward_pass_input_ids_only(self, bert, bert_config, document_classification_head):
        batch_size, sequence_length, num_labels, head = document_classification_head

        input_ids = torch.LongTensor(batch_size, sequence_length).random_(0, VOCAB)

        outputs = head(bert, input_ids)

        assert len(outputs) == 1
        assert outputs[0].size() == num_labels

    def test_forward_pass_with_attn_masks(self, bert, document_classification_head):
        batch_size, sequence_length, num_labels, head = document_classification_head

        input_ids = torch.randint(VOCAB, (batch_size, sequence_length))
        attention_mask = torch.randint(2, (batch_size, sequence_length))

        outputs = head(bert, input_ids, attention_mask=attention_mask)

        assert len(outputs) == 1
        assert outputs[0].size() == num_labels

    def test_forward_pass_with_token_type_ids(self, bert, document_classification_head):
        batch_size, sequence_length, num_labels, head = document_classification_head

        input_ids = torch.randint(VOCAB, (batch_size, sequence_length))
        token_type_ids = torch.ones_like(input_ids)

        outputs = head(bert, input_ids, token_type_ids=token_type_ids)

        assert len(outputs) == 1
        assert outputs[0].size() == num_labels

    def test_forward_pass_with_position_ids(self, bert, document_classification_head):
        batch_size, sequence_length, num_labels, head = document_classification_head

        input_ids = torch.randint(VOCAB, (batch_size, sequence_length))
        position_ids = torch.arange(sequence_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        outputs = head(bert, input_ids, position_ids=position_ids)

        assert len(outputs) == 1
        assert outputs[0].size() == num_labels

    def test_forward_pass_with_head_mask(self, bert_config, document_classification_head):
        # We need all hidden states to confirm the head masking worked as expected
        bert_config.output_attentions = True
        bert = BertModel(bert_config)

        batch_size, sequence_length, num_labels, head = document_classification_head

        input_ids = torch.randint(VOCAB, (batch_size, sequence_length))
        head_mask = torch.ones(bert_config.num_attention_heads)
        # Turn on one head off random
        masked_head = randint(0, head_mask.size(-1))
        head_mask[masked_head] = 0

        outputs = head(bert, input_ids, head_mask=head_mask)

        assert len(outputs) == 2
        assert outputs[0].size() == num_labels

        # Asserts that attention weights are of the expected size, and that one of the attention
        # heads was masked.
        output_shape = \
            (batch_size, bert_config.num_attention_heads, sequence_length, sequence_length)
        for attention_weights in outputs[1]:
            assert attention_weights.size() == output_shape
            for head, weights in enumerate(attention_weights):
                if head == masked_head:
                    assert torch.equal(weights, torch.zeros_like(weights))
                else:
                    assert not torch.equal(weights, torch.zeros_like(weights))

    def test_forward_pass_with_labels(self, bert, document_classification_head):
        batch_size, sequence_length, num_labels, head = document_classification_head

        input_ids = torch.randint(VOCAB, (batch_size, sequence_length))
        labels = torch.randint(num_labels[-1], (num_labels[0],))

        outputs = head(bert, input_ids, labels=labels)

        assert len(outputs) == 2
        assert outputs[0].size() == torch.Size([])  # Loss is a single element tensor
        assert outputs[1].size() == num_labels
