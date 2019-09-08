import pytest
from pytorch_transformers import BertTokenizer


@pytest.fixture
def bert_tokenizer():
    """Tokenizer for pre-trained BERT model.
    """
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return bert_tokenizer
