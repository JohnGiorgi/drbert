import pytest
import torch

from ..training.util import generate_inputs


class TestTrainingUtil(object):
    """Collects all unit tests for `drbert.training.util`.
    """
    def test_generate_inputs_invalid_task_value_error(self, sequence_labelling_dataset_reader, bert_tokenizer):
        name = 'arbitrary'
        task = 'invalid'
        _, dataset_reader = sequence_labelling_dataset_reader
        iterator = dataset_reader.textual_to_iterator()['train']
        batch = next(iter(iterator))

        with pytest.raises(ValueError):
            _ = generate_inputs(name, task, batch, bert_tokenizer)

    def test_generate_inputs(self, sequence_labelling_dataset_reader, bert_tokenizer):
        name = 'arbitrary'
        task = 'sequence_labelling'
        _, dataset_reader = sequence_labelling_dataset_reader
        iterator = dataset_reader.textual_to_iterator()['train']
        batch = next(iter(iterator))

        expected = {
            'name': name,
            'input_ids': batch.text,
            'labels': batch.label,
            'attention_mask': torch.where(
                batch.text == bert_tokenizer.pad_token_id,
                torch.zeros_like(batch.text),
                torch.ones_like(batch.text)
            )
        }
        actual = generate_inputs(name, task, batch, bert_tokenizer)

        for key, value in expected.items():
            if key == 'name':
                assert value == actual[key]
            else:
                assert torch.equal(value, actual[key])
