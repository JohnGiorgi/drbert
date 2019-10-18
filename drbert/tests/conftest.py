import pytest
from pkg_resources import resource_filename
from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer

from ..data.dataset_readers import DatasetReader
from ..data.dataset_readers import DocumentClassificationDatasetReader
from ..data.dataset_readers import NLIDatasetReader
from ..data.dataset_readers import RelationClassificationDatasetReader
from ..data.dataset_readers import SequenceLabellingDatasetReader
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


@pytest.fixture
def dataset_reader(bert_tokenizer):
    """Initialized DatasetReader.
    """
    args = {
        'path':          'totally_arbitrary',
        'partitions':    {'train': 'totally_arbitrary'},
        'tokenizer':     bert_tokenizer,
        'format':        'tsv',
        'skip_header':   False,
        'batch_sizes':   (16,),
        'lower':         False,
    }

    dataset_reader = DatasetReader(**args)

    return args, dataset_reader


@pytest.fixture
def sequence_labelling_dataset_reader(bert_tokenizer):
    """Initialized SequenceLabellingDatasetReader.
    """
    args = {
        'path':          resource_filename(__name__, 'resources/BC5CDR'),
        # TODO (John): Change these back to original data once we figure out the wordpiece problem
        'partitions':    {'train':      'train_wordpiece.tsv',
                          'validation': 'devel_wordpiece.tsv',
                          'test':       'test_wordpiece.tsv'},
        'tokenizer':     bert_tokenizer,
        'batch_sizes':   (16, 256, 256),
        'lower':         False,
    }

    dataset_reader = SequenceLabellingDatasetReader(**args)

    return args, dataset_reader


@pytest.fixture
def relation_classification_dataset_reader(bert_tokenizer):
    """Initialized RelationClassificationDatasetReader.
    """
    args = {
        'path':          resource_filename(__name__, 'resources/ChemProt'),
        'partitions':    {'train':      'train.tsv',
                          'validation': 'dev.tsv',
                          'test':       'test.tsv'},
        'tokenizer':     bert_tokenizer,
        'batch_sizes':   (16, 256, 256),
        'lower':         False,
    }

    dataset_reader = RelationClassificationDatasetReader(**args)

    return args, dataset_reader


@pytest.fixture
def document_classification_dataset_reader(bert_tokenizer):
    """Initialized DocumentClassificationDatasetReader.
    """
    args = {
        'path':          resource_filename(__name__, 'resources/n2c2_2006_smoking'),
        'partitions':    {'train': 'smokers_surrogate_train_all_version2.jsonl',
                          'test':  'smokers_surrogate_test_all_groundtruth_version2.jsonl'},
        'tokenizer':     bert_tokenizer,
        'batch_sizes':   (16, 256),
        'lower':         False,
    }

    dataset_reader = DocumentClassificationDatasetReader(**args)

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
        'batch_sizes':   (16, 256, 256),
        'lower':         False,
    }

    dataset_reader = NLIDatasetReader(**args)

    return args, dataset_reader
