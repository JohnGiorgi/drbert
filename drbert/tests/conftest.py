import pytest
from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer

from ..heads import DocumentClassificationHead
from ..heads import SequenceClassificationHead
from ..heads import SequenceLabellingHead


@pytest.fixture
def bert_tokenizer():
    """Tokenizer for pre-trained BERT model.
    """
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return bert_tokenizer


@pytest.fixture
def bert_config():
    """Config for pre-trained BERT model.
    """
    bert_config = BertConfig.from_pretrained('bert-base-uncased')

    return bert_config


@pytest.fixture
def bert(bert_config):
    """Pre-trained BERT model.
    """
    bert = BertModel(bert_config)

    return bert


@pytest.fixture
def sequence_labelling_head(bert_config):
    """Initialized sequence labelling head.
    """
    batch_size, sequence_length, num_labels = 1, 8, 10
    # TODO (John): This will change when we decouple the model from these tasks.
    bert_config.__dict__['num_deid_labels'] = num_labels

    head = SequenceLabellingHead(bert_config)

    return batch_size, sequence_length, num_labels, head


@pytest.fixture
def sequence_classification_head(bert_config):
    """Initialized sequence classification head.
    """
    batch_size, sequence_length, num_labels = 1, 8, 2
    # TODO (John): This will change when we decouple the model from these tasks.
    bert_config.__dict__['num_labels'] = num_labels

    head = SequenceClassificationHead(bert_config)

    return batch_size, sequence_length, num_labels, head


@pytest.fixture
def document_classification_head(bert_config):
    """Initialized sequence labelling head.
    """
    batch_size, sequence_length, num_labels = 1, 8, (16, 4)
    cohort_ffnn_size = 64
    # TODO (John): This will change when we decouple the model from these tasks.
    bert_config.__dict__['num_cohort_labels'] = num_labels
    bert_config.__dict__['cohort_ffnn_size'] = cohort_ffnn_size

    head = DocumentClassificationHead(bert_config)

    return batch_size, sequence_length, num_labels, head
