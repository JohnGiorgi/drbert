import pytest
from torchtext import datasets
from ..data.dataset_readers import DatasetReader


class TestDatasetReader(object):
    """Collects all unit tests for `drbert.data.dataset_readers.DatasetReader`.
    """
    def test_attributes_after_initialization(self, dataset_reader):
        args, dataset_reader = dataset_reader

        assert dataset_reader.path == args['path']
        assert dataset_reader.partitions == args['partitions']
        assert dataset_reader.tokenizer is args['tokenizer']
        assert dataset_reader.format == args['format']
        assert dataset_reader.skip_header == args['skip_header']
        assert dataset_reader.batch_sizes == args['batch_sizes']
        assert dataset_reader.lower == args['lower']

    def test_value_error_not_tuple(self, dataset_reader):
        args, dataset_reader = dataset_reader
        args['batch_sizes'] = 16  # Change batch_size to be an int to trigger ValueError

        with pytest.raises(ValueError):
            DatasetReader(*args)

    def test_value_error_wrong_sizes(self, dataset_reader):
        args, dataset_reader = dataset_reader
        # Change batch_size to be longer than len(partitions) to trigger ValueError
        args['batch_sizes'] = (16, 256)

        with pytest.raises(ValueError):
            DatasetReader(*args)


class TestSequenceLabellingDatasetReader(object):
    def test_attributes_after_initialization(self, sequence_labelling_dataset_reader):
        args, dataset_reader = sequence_labelling_dataset_reader

        assert dataset_reader.path == args['path']
        assert dataset_reader.partitions == args['partitions']
        assert dataset_reader.tokenizer is args['tokenizer']
        assert dataset_reader.format.lower() == 'tsv'
        assert not dataset_reader.skip_header
        assert dataset_reader.batch_sizes == args['batch_sizes']
        assert dataset_reader.lower == args['lower']
        assert dataset_reader.sort_key == datasets.SequenceTaggingDataset.sort_key

    def test_textual_to_iterator(self, sequence_labelling_dataset_reader):
        args, dataset_reader = sequence_labelling_dataset_reader

        train_iter, valid_iter, test_iter = dataset_reader.textual_to_iterator()

        train_batch = next(iter(train_iter))
        # There are only 2 examples in the test data, so bs == 2
        assert train_batch.text.size(0) == 2
        assert len(train_batch.label) == 2
        assert len(train_batch.mask) == 2

        valid_batch = next(iter(valid_iter))
        assert valid_batch.text.size(0) == 2
        assert len(valid_batch.label) == 2
        assert len(valid_batch.mask) == 2

        test_batch = next(iter(test_iter))
        assert test_batch.text.size(0) == 2
        assert len(test_batch.label) == 2
        assert len(test_batch.mask) == 2


class TestRelationClassificationDatasetReader(object):
    def test_attributes_after_initialization(self, relation_classification_dataset_reader):
        args, dataset_reader = relation_classification_dataset_reader

        assert dataset_reader.path == args['path']
        assert dataset_reader.partitions == args['partitions']
        assert dataset_reader.tokenizer is args['tokenizer']
        assert dataset_reader.format.lower() == 'tsv'
        assert dataset_reader.skip_header
        assert dataset_reader.batch_sizes == args['batch_sizes']
        assert dataset_reader.lower == args['lower']

    def test_textual_to_iterator(self, relation_classification_dataset_reader):
        args, dataset_reader = relation_classification_dataset_reader

        train_iter, valid_iter, test_iter = dataset_reader.textual_to_iterator()

        train_batch = next(iter(train_iter))
        # There are only 5 examples in the test data, so bs == 5
        assert train_batch.text.size(0) == 5
        assert len(train_batch.label) == 5

        valid_batch = next(iter(valid_iter))
        assert valid_batch.text.size(0) == 5
        assert len(valid_batch.label) == 5

        test_batch = next(iter(test_iter))
        assert test_batch.text.size(0) == 5
        assert len(test_batch.label) == 5


class TestDocumentClassificationDatasetReader(object):
    def test_attributes_after_initialization(self, document_classification_dataset_reader):
        args, dataset_reader = document_classification_dataset_reader

        assert dataset_reader.path == args['path']
        assert dataset_reader.partitions == args['partitions']
        assert dataset_reader.tokenizer is args['tokenizer']
        assert dataset_reader.format.lower() == 'json'
        assert not dataset_reader.skip_header
        assert dataset_reader.batch_sizes == args['batch_sizes']
        assert dataset_reader.lower == args['lower']

    def test_textual_to_iterator(self, document_classification_dataset_reader):
        args, dataset_reader = document_classification_dataset_reader

        train_iter, test_iter = dataset_reader.textual_to_iterator()

        train_batch = next(iter(train_iter))
        # There are only 5 examples in the test data, so bs == 5
        assert train_batch.text.size(0) == 5
        assert len(train_batch.label) == 5

        test_batch = next(iter(test_iter))
        assert test_batch.text.size(0) == 5
        assert len(test_batch.label) == 5


class TestNLIDatasetReader(object):
    def test_attributes_after_initialization(self, nli_dataset_reader):
        args, dataset_reader = nli_dataset_reader

        assert dataset_reader.path == args['path']
        assert dataset_reader.partitions == args['partitions']
        assert dataset_reader.tokenizer is args['tokenizer']
        assert dataset_reader.format.lower() == 'json'
        assert not dataset_reader.skip_header
        assert dataset_reader.batch_sizes == args['batch_sizes']
        assert dataset_reader.lower == args['lower']
        assert dataset_reader.sort_key == datasets.nli.NLIDataset.sort_key

    def test_textual_to_iterator(self, nli_dataset_reader):
        args, dataset_reader = nli_dataset_reader

        train_iter, valid_iter, test_iter = dataset_reader.textual_to_iterator()

        train_batch = next(iter(train_iter))
        # There are only 5 examples in the test data, so bs == 5
        assert train_batch.premise.size(0) == 5
        assert train_batch.hypothesis.size(0) == 5
        assert len(train_batch.label) == 5

        valid_batch = next(iter(valid_iter))
        assert valid_batch.premise.size(0) == 5
        assert valid_batch.hypothesis.size(0) == 5
        assert len(valid_batch.label) == 5

        test_batch = next(iter(test_iter))
        assert test_batch.premise.size(0) == 5
        assert test_batch.hypothesis.size(0) == 5
        assert len(test_batch.label) == 5
