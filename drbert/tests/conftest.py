import pytest
from pytorch_transformers import BertTokenizer

from ..modules.sequence_labelling_head import SequenceLabellingHead


@pytest.fixture
def bert_tokenizer():
    """Tokenizer for pre-trained BERT model.
    """
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return bert_tokenizer


@pytest.fixture
def sequence_labelling_head():
    batch_size, in_features, out_features = 32, 768, 10
    head = SequenceLabellingHead(in_features, out_features)

    return batch_size, in_features, out_features, head
