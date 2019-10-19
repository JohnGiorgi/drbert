import json
import os

from torchtext import data
from torchtext import datasets

from ..constants import PARTITIONS


class DatasetReader(object):
    """The parent class for all dataset readers. A user is mean to interact with a subclass of this
    class (e.g. `DocumentClassificationDatasetReader`), not this class itself.

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition(s) ('train', 'validation', 'test')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        format (str): Format of the data file. One of “CSV”, “TSV”, or “JSON” (case-insensitive).
        skip_header (bool, pptional): Whether to skip the first line of the input file. Defaults to
            False.
        batch_sizes (tuple, optional): Tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        lower: (bool, optional): True if text should be lower cased, False if not. Defaults to
            False.
        sort_key (function): A key to use for sorting examples in order to batch together examples
            with similar lengths and minimize padding. The sort_key provided to the Iterator
            constructor overrides the sort_key attribute of the Dataset, or defers to it if None.
        device (str or torch.device, optional): A string or instance of torch.device specifying
            which device the Tensors are going to be created on. If left as default, the tensors
            will be created on cpu. Default: None.

    Raises:
        ValueError: If batch_sizes is not a tuple or None.
        ValueError: If batch_sizes is not None and `len(batch_sizes) != len(partitions)`.
        ValueError: If 'train' not `partitions`.
        ValueError: If any key in `partitions` is not in `PARTITIONS`.
    """
    def __init__(self, path, partitions, tokenizer, format, skip_header=False,
                 batch_sizes=None, lower=False, sort_key=None, device='cpu'):
        if batch_sizes is not None:
            if not isinstance(batch_sizes, tuple):
                raise ValueError(f"'batch_sizes' must be a tuple. Got: {batch_sizes}")
            if len(partitions) != len(batch_sizes):
                raise ValueError(f"(len(batch_sizes) ({len(batch_sizes)}) must equal the number of"
                                 f" partitions ({len(partitions)})")
        if 'train' not in partitions:
            raise ValueError(f'"train" must be a key in "partitions". Got keys: {partitions.keys()}')
        for partition in partitions:
            if partition not in {'train', 'validation', 'test'}:
                raise ValueError((f"Found invalid key ({partition}) in partitions. All keys must be"
                                  f" one of {PARTITIONS}"))

        self.path = path
        self.partitions = partitions
        self.tokenizer = tokenizer
        self.format = format
        self.skip_header = skip_header
        self.batch_sizes = batch_sizes
        self.lower = lower
        self.sort_key = sort_key
        self.device = device

        # These are the default fields that can be updated by subclasses if necessary
        self.TEXT = data.Field(
            # To use the tokenizer from Transformers, we need to do three things:
            # 1. Set use_vocab=False
            # 2. Set tokenizer=tokenizer.encode
            # 3. Provide indices for all special tokens (init, eos, pad & unk)
            use_vocab=False,
            init_token=self.tokenizer.cls_token_id,
            eos_token=self.tokenizer.sep_token_id,
            tokenize=self.tokenizer.encode,
            batch_first=True,
            pad_token=self.tokenizer.pad_token_id,
            unk_token=self.tokenizer.unk_token_id
        )
        self.LABEL = data.LabelField()

    def textual_to_iterator(self, fields, datasets=None, **kwargs):
        """"

        Args:
            fields (list or dict): If using a list, the format must be CSV or TSV, and the values of
                the list should be tuples of (name, field). The fields should be in the same order
                as the columns in the CSV or TSV file, while tuples of (name, None) represent
                columns that will be ignored.
            datasets (tuple, optional): A tuple of torchtext Dataset objects. If None,
                these are loaded via `data.TabularDataset.splits()`. Defaults to None.
            kwargs: Remaining keyword arguments are passed to the `data.TabularDataset.splits` and
                `data.BucketIterator.splits` classes.

        Returns:
            tuple: A tuple of dictionaries keyed by partitions (`self.partitions.keys()`) containing
            the torchtext `Dataset` and torchtext `Iterator` objects respectively.
        """
        if datasets is None:
            # Define the splits. Need path to directory (self.path), and then name of each file.
            datasets_ = data.TabularDataset.splits(
                path=self.path,
                **self.partitions,
                format=self.format,
                fields=fields,
                skip_header=self.skip_header
            )
        else:
            datasets_ = datasets

        # Define the iterator. Batch sizes are defined per partition.
        # BucketIterator groups sentences of similar length together to minimize padding.
        iterators_ = data.BucketIterator.splits(
            datasets_, batch_sizes=self.batch_sizes, sort_key=self.sort_key, device=self.device,
            **kwargs
        )

        # Finally, build the vocab. This "numericalizes" the field.
        self.LABEL.build_vocab(*(split.label for split in datasets_))

        # Create dictionary of splits_ / iterators_ keyed by partition
        iterators, datasets = {}, {}

        # BucketIterator assumes the first dataset is the train partition
        datasets['train'] = datasets_[0]
        iterators['train'] = iterators_[0]

        # If a validation set was provided, it will be the second item in splits_ / iterators_
        if 'validation' in self.partitions:
            datasets['validation'] = datasets_[1]
            iterators['validation'] = iterators_[1]
            if 'test' in self.partitions:
                datasets['test'] = datasets_[2]
                iterators['test'] = iterators_[2]
        elif 'test' in self.partitions:
            iterators['test'] = iterators_[1]

        return datasets, iterators


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
                path='path/to/dataset', partitions=partitions, tokenizer=tokenizer,
                batch_sizes=(16, 256, 256)
            ).textual_to_iterator()

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'validation', 'test')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        lower: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
        kwargs: Remaining keyword arguments. Will be ignored.
    """
    def __init__(self, path, partitions, tokenizer, batch_sizes=None, lower=False, device='cpu',
                 **kwargs):
        super(SequenceLabellingDatasetReader, self).__init__(
            path=path, partitions=partitions, tokenizer=tokenizer, format='tsv', skip_header=False,
            batch_sizes=batch_sizes, lower=lower,
            # Sort examples according to length of sentences
            sort_key=datasets.SequenceTaggingDataset.sort_key,
            device=device
        )

    def textual_to_iterator(self, **kwargs):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a sequence labelling dataset.

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        def preprocessor(batch):
            return self.tokenizer.encode(batch)

        self.TEXT.preprocessing = preprocessor

        # Overwrite the parents LABEL field
        self.LABEL = data.Field(
            init_token=self.tokenizer.cls_token,
            eos_token=self.tokenizer.sep_token,
            batch_first=True,
            pad_token=self.tokenizer.pad_token,
            unk_token=None,
            is_target=True
        )

        # HACK (John): Hopefully, this is temporary. This is a mask of the same length as TEXT and
        # LABEL fields. It is 1 where an original token is found, and 0 otherwise (e.g. for special
        # tokens, pads, and wordpiece tokens)
        self.MASK = data.Field(
            use_vocab=False,
            init_token=0,
            eos_token=0,
            # Convert the mask to integers
            preprocessing=lambda x: list(map(int, x)),
            batch_first=True,
            pad_token=0,
            unk_token=None
        )

        fields = [('text', self.TEXT), ('label', self.LABEL), ('mask', self.MASK)]

        # Define the datasets. Need path to directory (self.path), and then name of each file.
        datasets_ = datasets.SequenceTaggingDataset.splits(
            path=self.path,
            **self.partitions,
            fields=fields,
            **kwargs
        )

        _, iterators = super(SequenceLabellingDatasetReader, self).textual_to_iterator(
            fields, datasets_, **kwargs
        )

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
                path='path/to/dataset', partitions=partitions, tokenizer=tokenizer,
                batch_sizes=(16, 256, 256)
            ).textual_to_iterator()

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'validation', 'test')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        lower: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
        kwargs: Remaining keyword arguments. Will be ignored.
    """
    def __init__(self, path, partitions, tokenizer, batch_sizes=None, lower=False,
                 device='cpu', **kwargs):

        super(RelationClassificationDatasetReader, self).__init__(
            path=path, partitions=partitions, tokenizer=tokenizer, format='tsv', skip_header=True,
            batch_sizes=batch_sizes, lower=lower,
            # Sort examples according to length of sentences
            sort_key=lambda x: len(x.text),
            device=device
        )

    def textual_to_iterator(self, **kwargs):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a relation classification dataset.

        Args:
            kwargs: Passed to the `data.TabularDataset.splits` and
            `data.BucketIterator.splits` class in the parent classes method `textual_to_iterator`.

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        fields = [('index', None), ('text', self.TEXT), ('label', self.LABEL)]

        _, iterators = \
            super(RelationClassificationDatasetReader, self).textual_to_iterator(fields, **kwargs)

        return iterators


class DocumentClassificationDatasetReader(DatasetReader):
    """Reads instances from JSON lines file(s) were each line is in the following format:

    For multi-class datasets:

        {"text": DOCUMENT TEXT, "label": LABEL}

    For multi-label datasets

        {"text": DOCUMENT TEXT, "label_1": LABEL_2: "label_2": LABEL_2, ..., "label_n": LABEL_N}

    where any field in the JSON that is not in ["text", "id"] will be considered a label field. This
    allows for loading for multi-label datasets.

    E.g.,

    {"text": "The patient is a 58 year old...", label": "PAST SMOKER", "id": "660"}

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
                path='path/to/dataset', partitions=partitions, tokenizer=tokenizer,
                batch_sizes=(16, 256, 256)
            ).textual_to_iterator()

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'validation', 'test')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        lower: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
        kwargs: Remaining keyword arguments. Will be ignored.
    """
    def __init__(self, path, partitions, tokenizer, batch_sizes=None, lower=False,
                 device='cpu', **kwargs):
        super(DocumentClassificationDatasetReader, self).__init__(
            path=path, partitions=partitions, tokenizer=tokenizer, format='json', skip_header=False,
            batch_sizes=batch_sizes, lower=lower,
            # Sort examples according to length of documents
            sort_key=lambda x: len(x.text),
            device=device
        )

    def textual_to_iterator(self, **kwargs):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a document classification dataset.

        Args:
            kwargs: Passed to the `data.TabularDataset.splits` and
            `data.BucketIterator.splits` class in the parent classes method `textual_to_iterator`.

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        fields = self.get_fields()

        _, iterators = super().textual_to_iterator(fields, **kwargs)

        return iterators

    def get_fields(self):
        """Returns a dictionary of tuples containing torchtext fields.

        If "label" is in the JSON line file(s) at `self.path`, this is considered to be a
        single-label dataset. The returned dictionary will contain two torchtext fields at keys
        "text" and "label"

        Otherwise, this is considered to be a multi-label dataset. The returned dictionary will
        contain a torchtext field at key "text", as well as a field for every key in the JSON
        lines(s) not in ["text", "id"].
        """
        # These fields will NOT be considered label fields
        blacklist = {'text', 'id'}
        # Assume 'text' field is present
        fields = {'text': ('text', self.TEXT)}

        # Add label fields based on JSON lines file
        filepath = os.path.join(self.path, list(self.partitions.values())[0])
        with open(filepath, 'r') as f:
            jsonl_keys = [key for key in json.loads(f.readline()).keys() if key not in blacklist]
            # Multi-class datasets
            if 'label' in jsonl_keys:
                fields['label'] = ('label', self.LABEL)
            # Multi-label datasets
            else:
                for key in jsonl_keys:
                    fields[key] = (key, self.label)

        return fields


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
                path='path/to/dataset', partitions=partitions, tokenizer=tokenizer,
                batch_sizes=(16, 256, 256)
            ).textual_to_iterator()

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'validation', 'test')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        lower: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
        kwargs: Remaining keyword arguments. Will be ignored.
    """
    def __init__(self, path, partitions, tokenizer, batch_sizes=None, lower=False,
                 device='cpu', **kwargs):
        super(NLIDatasetReader, self).__init__(
            path=path, partitions=partitions, tokenizer=tokenizer, format='json', skip_header=False,
            batch_sizes=batch_sizes, lower=lower,
            # Sort examples according to length of premise and hypothesis
            sort_key=datasets.nli.NLIDataset.sort_key,
            device=device
        )

    def textual_to_iterator(self, **kwargs):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a NLI dataset.

        Args:
            kwargs: Passed to the `data.TabularDataset.splits` and
            `data.BucketIterator.splits` class in the parent classes method `textual_to_iterator`.

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        # Premise and hypothesis will be paired, so max_len is set to 256
        self.tokenizer.max_len = 256

        fields = {
            'sentence1':  ('premise', self.TEXT),
            'sentence2':  ('hypothesis', self.TEXT),
            'gold_label': ('label', self.LABEL)
        }

        _, iterators = super(NLIDatasetReader, self).textual_to_iterator(fields, **kwargs)

        return iterators
