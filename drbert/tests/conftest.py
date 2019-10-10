import pytest
from pkg_resources import resource_filename
from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer

from ..data.dataset_readers import DatasetReader
from ..data.dataset_readers import NLIDatasetReader
from ..heads import DocumentClassificationHead
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


@pytest.fixture
def dataset_reader(bert_tokenizer):
    """Initialized DatasetReader.
    """
    args = {
        'path':          'totally_arbitrary',
        'partitions':    {'train': 'totally_arbitrary'},
        'tokenizer':     bert_tokenizer,
        'format':        'tsv',
        'batch_sizes':   (16,),
        'maxlen':        512,
        'lower_case': False,
    }

    dataset_reader = DatasetReader(**args)

    return args, dataset_reader


@pytest.fixture
def nli_dataset_reader(bert_tokenizer):
    """Initialized NLIDatasetReader.
    """
    args = {
        'path':          resource_filename(__name__, 'resources/snli_1.0'),
        'partitions':    {'train':      'snli_1.0_train.jsonl',
                          'validation': 'snli_1.0_dev.jsonl',
                          'test':       'snli_1.0_test.jsonl'},
        'tokenizer':     bert_tokenizer,
        'format':        'json',
        'batch_sizes':   (16, 256, 256),
        'maxlen':        512,
        'lower_case': False,
    }

    dataset_reader = NLIDatasetReader(**args)

    return args, dataset_reader
