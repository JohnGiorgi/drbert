from torchtext import data
from torchtext import datasets
from transformers import AutoTokenizer

from ..constants import COHORT_DATA_DIR, COHORT_DISEASE_CONSTANTS


class DatasetReader(object):
    """The parent class for all dataset readers.

    Args:
        path (str): Common prefix of the splits’ file paths.
        partitions (dict): A dictionary keyed by partition ('train', 'test', 'validation')
            containing the suffix to add to `path` for that partition.
        tokenizer (TODO): The function used to tokenize strings into sequential examples.
        batch_sizes (tuple): Optional, tuple of batch sizes to use for the different splits, or None
            to use the same batch_size for all splits. Defaults to None.
        maxlen (int): Optional, a fixed length that all examples using this field will be padded to, or None
            for flexible sequence lengths. Default: 512.
        lower_case: (bool): Optional, True if text should be lower cased, False if not. Defaults to
            False.
    """
    def __init__(self, path, partitions, tokenizer, label_vocab, batch_sizes=None, maxlen=512, lower_case=False):
        if batch_sizes is not None:
            if not isinstance(batch_sizes, tuple):
                raise ValueError(f"'batch_sizes' must be a tuple. Got: {batch_sizes}")
            if len(partitions) != len(batch_sizes):
                raise ValueError(f"(len(batch_sizes) ({len(batch_sizes)}) must equal the number of"
                                 f" partitions ({len(partitions)})")

        self.path = path
        self.partitions = partitions
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab
        self.batch_sizes = batch_sizes
        self.maxlen = maxlen
        self.lower_case = lower_case


class DocumentClassificationDatasetReader(DatasetReader):
    def __init__(self, path, partitions, tokenizer, label_vocab, batch_sizes=None, maxlen=512, lower_case=False):
        super(DocumentClassificationDatasetReader, self).__init__(
            path, partitions, tokenizer, label_vocab, batch_sizes, maxlen, lower_case
        )

    def text_to_iterator(self):
        """Does whatever tokenization or processing is necessary to go from textual input to
        iterators for training and evaluation for a document classification dataset. Assumes multi-label multi-class framing.
        
        Args:
            path: path to data splits. Assume labels are numericalized. Can also tokenize text. In that case call the AutoTokenizer's `convert_tokens_to_ids` method in a torchtext Pipeline and add to TEXT field below
        
        Returns:
            A tuple of `torchtext.data.iterator.BucketIterator` objects, one for each partition.
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
        LABELS = data.LabelField(use_vocab=False)

        fields = {'TEXT': ('text', TEXT)}

        # Add new entry in fields for every label in multi-label output
        for label in self.label_vocab:
            fields[label] = (f'label_{label}', LABELS)

        # Define the splits. Need path to a top level directory (self.path), and then the name of
        # each file.
        splits = data.TabularDataset.splits(
            path=self.path,
            **self.partitions,
            format='json',
            fields=fields
        )

        # Define the iterator. Batch sizes are defined per partition.
        # BucketIterator groups sentences of similar length together to minimize padding. Use sort_key to sort examples by length and group similar sized entries into batches
        iterators = data.BucketIterator.splits(
            splits, sort_key=lambda x: len(x.text), batch_sizes=self.batch_sizes
        )

        # Finally, build the vocab. This "numericalizes" the field.
        # Assume the first split is 'train'.
        # LABEL.build_vocab(splits[0]) # easier to numericalize labels before calling torchtext pipeline

        return iterators


class SequenceLabellingDatasetReader(DatasetReader):
    pass


class NLIDatasetReader(DatasetReader):
    pass

if __name__ == "__main__":
   
    # Params for Cohort Task
    path = None
    partitions = None
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    label_vocab = COHORT_DISEASE_CONSTANTS
    batch_sizes = None

    ICD_iters = DocumentClassificationDatasetReader(path, partitions, tokenizer, label_vocab, batch_sizes=batch_sizes, lower_case=True)

    print(ICD_iters.text_to_iterator())
