from torchtext import data


class DatasetReader(object):
    """The parent class for all dataset readers.

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'test', 'validation')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        format (str): The format of the data file. One of “CSV”, “TSV”, or “JSON” (case-insensitive).
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        maxlen (int): Optional, a fixed length that all examples using this field will be padded to,
            for flexible sequence lengths. Default: 512.
        lower_case: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.

    Raises:
        ValueError: If batch_sizes is not a tuple or None.
        ValueError: If batch_sizes is not None and `len(batch_sizes) != len(partitions)`.
    """
    def __init__(self, path, partitions, tokenizer, format, batch_sizes=None, maxlen=512,
                 lower_case=False, sort_key=None):
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
        self.batch_sizes = batch_sizes
        self.maxlen = maxlen
        self.lower_case = lower_case
        self.sort_key = sort_key

    def textual_to_iterator(self, fields):
        # Define the splits. Need path to directory (self.path), and then name of each file.
        splits = data.TabularDataset.splits(
            path=self.path,
            **self.partitions,
            format=self.format,
            fields=fields,
        )

        # Define the iterator. Batch sizes are defined per partition.
        # BucketIterator groups sentences of similar length together to minimize padding.
        iterators = data.BucketIterator.splits(
            splits, batch_sizes=self.batch_sizes, sort_key=self.sort_key
        )

        return splits, iterators


class DocumentClassificationDatasetReader(DatasetReader):
    """The parent class for all dataset readers.

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'test', 'validation')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        format (str): The format of the data file. One of “CSV”, “TSV”, or “JSON” (case-insensitive).
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        maxlen (int): Optional, a fixed length that all examples using this field will be padded to,
            for flexible sequence lengths. Default: 512.
        lower_case: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
    """
    def __init__(self, path, partitions, tokenizer, format='json', batch_sizes=None, maxlen=512,
                 lower_case=False):
        super(DocumentClassificationDatasetReader, self).__init__(
            path, partitions, tokenizer, batch_sizes, maxlen, lower_case,
            # Sort examples according to length of documents
            sort_key=lambda x: len(x.text)
        )

    def textual_to_iterator(self):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a document classification dataset. The data
        must be in JSON lines format (http://jsonlines.org/).

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        # To use the tokenizer from Transformers, we need to do three things:
        # 1. Set use_vocab=False
        # 2. Set tokenizer=tokenizer.encode
        # 3. Provide indices for all special tokens (init, eos, pad & unk)
        TEXT = data.Field(
            use_vocab=False,
            init_token=self.tokenizer.cls_token_id,
            eos_token=self.tokenizer.sep_token_id,
            fix_length=self.maxlen,
            tokenize=self.tokenizer.encode,
            batch_first=True,
            pad_token=self.tokenizer.pad_token_id,
            unk_token=self.tokenizer.unk_token_id
        )
        LABEL = data.LabelField(fix_length=self.maxlen)

        fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}

        splits, iterators = super().textual_to_iterator(fields)

        # Finally, build the vocab. This "numericalizes" the field.
        # Assume the first split is 'train'.
        LABEL.build_vocab(splits[0])

        return iterators


class SequenceLabellingDatasetReader(DatasetReader):
    """The parent class for all dataset readers.

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'test', 'validation')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        format (str): The format of the data file. One of “CSV”, “TSV”, or “JSON” (case-insensitive).
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        maxlen (int): Optional, a fixed length that all examples using this field will be padded to,
            for flexible sequence lengths. Default: 512.
        lower_case: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
    """
    def __init__(self, path, partitions, tokenizer, format='csv', batch_sizes=None, maxlen=512,
                 lower_case=False):
        super(SequenceLabellingDatasetReader, self).__init__(
            path, partitions, tokenizer, format, batch_sizes, maxlen, lower_case,
            # Sort examples according to length of sentences
            sort_key=lambda x: len(x.text)
        )

    def textual_to_iterator(self):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a document classification dataset. The data
        must be in JSON lines format (http://jsonlines.org/).

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        # Define the "fields". Think of these like columns in your dataset.
        # To use the tokenizer from Transformers, we need to do three things:
        # 1. Set use_vocab=False
        # 2. Set tokenizer=tokenizer.encode
        # 3. Provide indices for all special tokens (init, eos, pad & unk)
        TEXT = data.Field(
            use_vocab=False,
            init_token=self.tokenizer.cls_token_id,
            eos_token=self.tokenizer.sep_token_id,
            fix_length=self.maxlen,
            tokenize=self.tokenizer.encode,
            batch_first=True,
            pad_token=self.tokenizer.pad_token_id,
            unk_token=self.tokenizer.unk_token_id
        )
        LABEL = data.LabelField()

        fields = [('text', TEXT), ('label', LABEL)]

        splits, iterators = super(NLIDatasetReader, self).textual_to_iterator(fields)

        # Finally, build the vocab. This "numericalizes" the field.
        # Assume the first split is 'train'.
        LABEL.build_vocab(splits[0])

        return iterators


class NLIDatasetReader(DatasetReader):
    """The parent class for all dataset readers.

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'test', 'validation')
            containing the suffix to add to `path` for that partition.
        tokenizer (transformers.PreTrainedTokenizer): The function used to tokenize strings into
            sequential examples.
        format (str): The format of the data file. One of “CSV”, “TSV”, or “JSON” (case-insensitive).
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        maxlen (int): Optional, a fixed length that all examples using this field will be padded to,
            for flexible sequence lengths. Default: 512.
        lower_case: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
    """
    def __init__(self, path, partitions, tokenizer, format='json', batch_sizes=None, maxlen=512,
                 lower_case=False):
        super(NLIDatasetReader, self).__init__(
            path, partitions, tokenizer, format, batch_sizes, maxlen, lower_case,
            # Sort examples according to length of premise and hypothesis
            sort_key=lambda x: data.interleave_keys(len(x.premise), len(x.hypothesis))
        )

    def textual_to_iterator(self):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a NLI dataset. The data must be in JSON lines
        format (http://jsonlines.org/).

        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition
            is `self.partitions`.
        """
        TEXT = data.Field(
            use_vocab=False,
            init_token=self.tokenizer.cls_token_id,
            eos_token=self.tokenizer.sep_token_id,
            fix_length=self.maxlen // 2,  # We will be pairing sentences, so divide by 2.
            tokenize=self.tokenizer.encode,
            batch_first=True,
            pad_token=self.tokenizer.pad_token_id,
            unk_token=self.tokenizer.unk_token_id
        )
        LABEL = data.LabelField()

        fields = {
            'sentence1':  ('premise', TEXT),
            'sentence2':  ('hypothesis', TEXT),
            'gold_label': ('label', LABEL)
        }

        splits, iterators = super(NLIDatasetReader, self).textual_to_iterator(fields)

        # Finally, build the vocab. This "numericalizes" the field.
        # Assume the first split is 'train'.
        LABEL.build_vocab(splits[0])

        return iterators
