from torchtext import data
from torchtext import datasets


class DatasetReader(object):
    """The parent class for all dataset readers. A user is mean to interact with a subclass of this
    class (e.g. DocumentClassificationDatasetReader), not this class itself.

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'test', 'validation')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        format (str): The format of the data file. One of “CSV”, “TSV”, or “JSON” (case-insensitive).
        skip_header (bool) : Optional, whether to skip the first line of the input file. Defaults to
            False.
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        fix_length (int): Optional, a fixed length that all examples using this field will be padded
            to, for flexible sequence lengths. Defaults to 512.
        lower: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
        sort_key (function): A key to use for sorting examples in order to batch together examples
            with similar lengths and minimize padding. The sort_key provided to the Iterator
            constructor overrides the sort_key attribute of the Dataset, or defers to it if None.

    Raises:
        ValueError: If batch_sizes is not a tuple or None.
        ValueError: If batch_sizes is not None and `len(batch_sizes) != len(partitions)`.
    """
    def __init__(self, path, partitions, tokenizer, format, skip_header=False, batch_sizes=None,
                 fix_length=512, lower=False, sort_key=None):
        if batch_sizes is not None:
            if not isinstance(batch_sizes, tuple):
                raise ValueError(f"'batch_sizes' must be a tuple. Got: {batch_sizes}")
            if len(partitions) != len(batch_sizes):
                raise ValueError(f"(len(batch_sizes) ({len(batch_sizes)}) must equal the number of"
                                 f" partitions ({len(partitions)})")

        self.path = path
        self.partitions = partitions
        self.tokenizer = tokenizer
        self.format = format
        self.skip_header = skip_header
        self.batch_sizes = batch_sizes
        self.fix_length = fix_length
        self.lower = lower
        self.sort_key = sort_key

        # These are the default fields that can be updated by subclasses if necessary
        self.TEXT = data.Field(
            # To use the tokenizer from Transformers, we need to do three things:
            # 1. Set use_vocab=False
            # 2. Set tokenizer=tokenizer.encode
            # 3. Provide indices for all special tokens (init, eos, pad & unk)
            use_vocab=False,
            init_token=self.tokenizer.cls_token_id,
            eos_token=self.tokenizer.sep_token_id,
            fix_length=self.fix_length,
            tokenize=self.tokenizer.encode,
            batch_first=True,
            pad_token=self.tokenizer.pad_token_id,
            unk_token=self.tokenizer.unk_token_id
        )
        self.LABEL = data.LabelField()

    def textual_to_iterator(self, fields):
        # Define the splits. Need path to directory (self.path), and then name of each file.
        splits = data.TabularDataset.splits(
            path=self.path,
            **self.partitions,
            format=self.format,
            fields=fields,
            skip_header=self.skip_header
        )

        # Define the iterator. Batch sizes are defined per partition.
        # BucketIterator groups sentences of similar length together to minimize padding.
        iterators = data.BucketIterator.splits(
            splits, batch_sizes=self.batch_sizes, sort_key=self.sort_key
        )

        return splits, iterators


class SequenceLabellingDatasetReader(DatasetReader):
    """Reads instances from a 2 columned TSV file(s) where each line is in the following format:

    WORD    LABEL

    E.g.,

    Famotidine	B-CHEM
    -	O
    associated	O
    delirium	O
    .	O

    Example usage:
        >>> from transformers import AutoTokenizer
        >>> from dataset_readers import SequenceLabellingDatasetReader
        >>> partitions={
                'train': 'train_filename.jsonl',
                'validation': 'valid_filename.jsonl',
                'test': 'test_filename.jsonl',
            }
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        >>> train_iter, valid_iter, test_iter = SequenceLabellingDatasetReader(
                path='path/to/nli/dataset', partitions=partitions, tokenizer=tokenizer,
                batch_sizes=(16, 256, 256)
            ).textual_to_iterator()

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'test', 'validation')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        fix_length (int): Optional, a fixed length that all examples using this field will be padded
            to, for flexible sequence lengths. Default: 512.
        lower: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
    """
    def __init__(self, path, partitions, tokenizer, batch_sizes=None, fix_length=512, lower=False):
        super(SequenceLabellingDatasetReader, self).__init__(
            path=path, partitions=partitions, tokenizer=tokenizer, format='TSV', skip_header=False,
            batch_sizes=batch_sizes, fix_length=fix_length, lower=lower,
            # Sort examples according to length of sentences
            sort_key=datasets.SequenceTaggingDataset.sort_key
        )

    def textual_to_iterator(self):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a sequence labelling dataset.

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        # Overwrite the parents LABEL field
        self.LABEL = data.Field(
            init_token=self.tokenizer.cls_token_id,
            eos_token=self.tokenizer.sep_token_id,
            fix_length=self.fix_length,
            batch_first=True,
            pad_token=self.tokenizer.pad_token_id,
        )

        fields = [('text', self.TEXT), ('label', self.LABEL)]

        # Define the splits. Need path to directory (self.path), and then name of each file.
        splits = datasets.SequenceTaggingDataset.splits(
            path=self.path,
            **self.partitions,
            fields=fields,
        )

        # Define the iterator. Batch sizes are defined per partition.
        # BucketIterator groups sentences of similar length together to minimize padding.
        iterators = data.BucketIterator.splits(
            splits, batch_sizes=self.batch_sizes, sort_key=self.sort_key
        )

        # Finally, build the vocab. This "numericalizes" the field.
        self.LABEL.build_vocab(*(split.label for split in splits))

        return iterators


class RelationClassificationDatasetReader(DatasetReader):
    """Reads instances from a 3 columned TSV file(s) where each line is in the following format:

    INDEX   SENTENCE    LABEL

    Additionally expects a header in each file. E.g.,

    index   sentence    label
    10064839.T49.T56    Binding of dimemorfan to @CHEMICAL$ and its...  false

    Example usage:
        >>> from transformers import AutoTokenizer
        >>> from dataset_readers import RelationClassificationDatasetReader
        >>> partitions={
                'train': 'train_filename.jsonl',
                'validation': 'valid_filename.jsonl',
                'test': 'test_filename.jsonl',
            }
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        >>> train_iter, valid_iter, test_iter = RelationClassificationDatasetReader(
                path='path/to/nli/dataset', partitions=partitions, tokenizer=tokenizer,
                batch_sizes=(16, 256, 256)
            ).textual_to_iterator()

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'test', 'validation')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        fix_length (int): A fixed length that all examples using this field will be padded to, or
            None for flexible sequence lengths. Default: None.
        lower: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
    """
    def __init__(self, path, partitions, tokenizer, batch_sizes=None, fix_length=512, lower=False):

        super(RelationClassificationDatasetReader, self).__init__(
            path=path, partitions=partitions, tokenizer=tokenizer, format='TSV', skip_header=True,
            batch_sizes=batch_sizes, fix_length=fix_length, lower=lower,
            # Sort examples according to length of sentences
            sort_key=lambda x: len(x.text)
        )

    def textual_to_iterator(self):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a relation classification dataset.

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        fields = [('index', None), ('text', self.TEXT), ('label', self.LABEL)]

        splits, iterators = super(RelationClassificationDatasetReader, self).textual_to_iterator(fields)

        # Finally, build the vocab. This "numericalizes" the field.
        self.LABEL.build_vocab(*(split.label for split in splits))

        return iterators


class DocumentClassificationDatasetReader(DatasetReader):
    """TODO

    Example usage:
        >>> from transformers import AutoTokenizer
        >>> from dataset_readers import DocumentClassificationDatasetReader
        >>> partitions={
                'train': 'train_filename.jsonl',
                'validation': 'valid_filename.jsonl',
                'test': 'test_filename.jsonl',
            }
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        >>> train_iter, valid_iter, test_iter = DocumentClassificationDatasetReader(
                path='path/to/nli/dataset', partitions=partitions, tokenizer=tokenizer,
                batch_sizes=(16, 256, 256)
            ).textual_to_iterator()

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'test', 'validation')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        fix_length (int): Optional, a fixed length that all examples using this field will be padded
            to, for flexible sequence lengths. Default: 512.
        lower: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
    """
    def __init__(self, path, partitions, tokenizer, batch_sizes=None, fix_length=512, lower=False):
        super(RelationClassificationDatasetReader, self).__init__(
            path=path, partitions=partitions, tokenizer=tokenizer, format='JSON', skip_header=False,
            batch_sizes=batch_sizes, fix_length=fix_length, lower=lower,
            # Sort examples according to length of documents
            sort_key=lambda x: len(x.text)
        )

    def textual_to_iterator(self):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a document classification dataset.

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        fields = {'text': ('text', self.TEXT), 'label': ('label', self.LABEL)}

        splits, iterators = super().textual_to_iterator(fields)

        # Finally, build the vocab. This "numericalizes" the field.
        self.LABEL.build_vocab(*(split.label for split in splits))

        return iterators


class NLIDatasetReader(DatasetReader):
    """Reads instances from JSON lines file(s) were each line is in the following format:

    {"gold_label": ENTIALMENT LABEL, "sentence1": PREMISE TEXT, "sentence2": HYPOTHESIS TEXT}

    E.g.,

    {"gold_label": "entailment", "sentence1": "Children smiling and waving at camera", "sentence2": "There are children present"}

    Example usage:
        >>> from transformers import AutoTokenizer
        >>> from dataset_readers import NLIDatasetReader
        >>> partitions={
                'train': 'train_filename.jsonl',
                'validation': 'valid_filename.jsonl',
                'test': 'test_filename.jsonl',
            }
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        >>> train_iter, valid_iter, test_iter = NLIDatasetReader(
                path='path/to/nli/dataset', partitions=partitions, tokenizer=tokenizer,
                batch_sizes=(16, 256, 256)
            ).textual_to_iterator()

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'test', 'validation')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        fix_length (int): A fixed length that all examples using this field will be padded to, or
            None for flexible sequence lengths. Default: None.
        lower: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
    """
    def __init__(self, path, partitions, tokenizer, batch_sizes=None, fix_length=512, lower=False):
        super(NLIDatasetReader, self).__init__(
            path=path, partitions=partitions, tokenizer=tokenizer, format='JSON', skip_header=False,
            batch_sizes=batch_sizes, fix_length=fix_length, lower=lower,
            # Sort examples according to length of premise and hypothesis
            sort_key=lambda x: data.interleave_keys(len(x.premise), len(x.hypothesis))
        )

    def textual_to_iterator(self):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a NLI dataset.

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        # Premise and hypothesis will be paired, so fix_length is set to 256
        self.TEXT.fix_length = self.fix_length // 2

        fields = {
            'sentence1':  ('premise', self.TEXT),
            'sentence2':  ('hypothesis', self.TEXT),
            'gold_label': ('label', self.LABEL)
        }

        splits, iterators = super(NLIDatasetReader, self).textual_to_iterator(fields)

        # Finally, build the vocab. This "numericalizes" the field.
        self.LABEL.build_vocab(*(split.label for split in splits))

        return iterators
