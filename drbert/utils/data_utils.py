import json
import os
import threading
import time
from glob import glob
from multiprocessing import Pool

import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus.reader.conll import ConllCorpusReader
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset

from ..constants import (BERT_MAX_SENT_LEN, CLS, DEID_LABELS, PAD, SEP,
                         WORDPIECE)


class CohortDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.file_list = os.listdir(data_path)
        self.file_list = [os.path.join(data_path, file_name) for file_name in self.file_list]
        self.cached = [None] * len(self.file_list)

        self.lock = threading.Lock()
        pool = Pool(5)
        pool.map_async(self._cache_items, range(len(self.file_list)))

    def _cache_items(self, idx):
        file_name = self.file_list[idx]
        with open(file_name) as open_file:
            read_file = json.loads(open_file.read())
        input_ids = torch.LongTensor(read_file['input_ids'])
        attn_mask = torch.LongTensor(read_file['attn_mask'])
        labels = torch.LongTensor(read_file['labels'])
        self.lock.acquire()
        self.cached[idx] = (input_ids, attn_mask, labels)
        self.lock.release()
        return True

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        self.lock.acquire()
        if self.cached[index] is not None:
            input_ids, attn_mask, labels = self.cached[index]
            self.lock.release()
            return input_ids, attn_mask, labels
        self.lock.release()

        file_name = self.file_list[index]
        with open(file_name) as open_file:
            read_file = json.loads(open_file.read())

        input_ids = torch.LongTensor(read_file['input_ids'])
        attn_mask = torch.LongTensor(read_file['attn_mask'])
        labels = torch.LongTensor(read_file['labels'])

        return input_ids, attn_mask, labels


def prepare_cohort_dataset(args, tokenizer):
    train_data_path = os.path.join(args.dataset_folder, "cohort", "preprocessed", "train")
    valid_data_path = os.path.join(args.dataset_folder, "cohort", "preprocessed", "valid")
    test_data_path = os.path.join(args.dataset_folder, "cohort", "preprocessed", "test")

    train_cohort_dataset = CohortDataset(train_data_path)
    valid_cohort_dataset = CohortDataset(valid_data_path)
    test_cohort_dataset = CohortDataset(test_data_path)

    all_dataset = {
        'train': train_cohort_dataset,
        'valid': valid_cohort_dataset,
        'test': test_cohort_dataset
    }

    return all_dataset


def prepare_deid_dataset(args, tokenizer):
    """Prepares the DeID tasks data for training / evaluation.

    Args:
        args (ArgumentParser): ArgumentParser object containing arguments parsed from the command
            line.
        tokenizer (BertTokenizer): An object with methods for tokenizing text for input to BERT.

    Returns:
        dict: A dictionary containing the preprocessed data for each partition 'train', 'valid' and
            'test'.
    """
    deid_folder = os.path.join(args.dataset_folder, 'deid')
    conll_parser = ConllCorpusReader(deid_folder, '.conll', ('words', 'pos'))
    dataset = {'train': None, 'valid': None, 'test': None}

    partitions = glob(os.path.join(deid_folder, '*.tsv'))

    for partition_filepath in partitions:
        start = time.time()
        partition_filename = os.path.basename(partition_filepath)
        partition = os.path.splitext(partition_filename)[0]
        print(f"Processing DeID data partition: '{partition}'...", end=' ', flush=True)

        tokens = list(conll_parser.sents(partition_filename))
        tags = [[t[-1] for t in s] for s in list(conll_parser.tagged_sents(partition_filename))]

        maxlen = args.max_seq_length if partition == "train" else BERT_MAX_SENT_LEN

        bert_tokens, orig_tok_mask, bert_labels = wordpiece_tokenize_sents(tokens, tokenizer, tags)

        indexed_tokens, attention_mask, orig_tok_mask, indexed_labels = \
            index_pad_mask_bert_tokens(
                bert_tokens, tokenizer, maxlen=maxlen, labels=bert_labels,
                orig_tok_mask=orig_tok_mask, tag_to_idx=DEID_LABELS
            )

        # Accumulate all tags in the dataset
        if partition == 'train':
            class_weights = compute_class_weight(
                class_weight='balanced', classes=np.unique(indexed_labels.view(-1).tolist()),
                y=indexed_labels.view(-1).tolist()
            )
            class_weights = class_weights.tolist()

        dataset[partition] = \
            TensorDataset(indexed_tokens, attention_mask, indexed_labels, orig_tok_mask)

        print(f'Done ({time.time() - start:.2f} seconds).')

    return dataset, class_weights


def wordpiece_tokenize_sents(tokens, tokenizer, labels=None):
    """Tokenizes pre-tokenized text for use with a BERT-based model.

    Given some pre-tokenized text, represented as a list (sentences) of lists (tokens), tokenizies
    the text for use with a BERT-based model while deterministically maintaining an
    original-to-tokenized alignment. This is a near direct copy of the example given in the BERT
    GitHub repo (https://github.com/google-research/bert#tokenization) with additional code for
    mapping token-level labels.

    Args:
        tokens (list): A list of lists containing tokenized sentences.
        tokenizer (BertTokenizer): An object with methods for tokenizing text for input to BERT.
        labels (list): Optional, a list of lists containing token-level labels for a collection of
            sentences. Defaults to None.

    Returns:
        If `labels` is not `None`:
            A tuple of `bert_tokens`, `orig_tok_mask`, `bert_labels`, representing tokens and
            labels that can be used to train a BERT model and a deterministc mapping of the elements
            in `bert_tokens` to `tokens`.
        If `labels` is `None`:
            A tuple of `bert_tokens`, and `orig_tok_mask`, representing tokens that can be used to
            train a BERT model and a deterministc mapping of `bert_tokens` to `sents`.

    References:
     - https://github.com/google-research/bert#tokenization
    """
    bert_tokens = []
    orig_tok_mask = []

    for sent in tokens:
        bert_tokens.append([CLS])
        orig_tok_mask.append([0])
        for orig_token in sent:
            wordpiece_tokens = tokenizer.wordpiece_tokenizer.tokenize(orig_token)
            bert_tokens[-1].extend(wordpiece_tokens)
            orig_tok_mask[-1].extend([1] + [0] * (len(wordpiece_tokens) - 1))
        bert_tokens[-1].append(SEP)
        orig_tok_mask[-1].append(0)

        outputs = (bert_tokens, orig_tok_mask)

    # If labels are provided, project them onto bert_tokens
    if labels is not None:
        bert_labels = []
        # Idea is to take the next item in the iterator if tok_map == 1 else take WORDPIECE
        for labs, tok_mask in zip(labels, orig_tok_mask):
            bert_labels.append([])
            lab_iter = iter(labs)
            for i in tok_mask:
                if i:
                    bert_labels[-1].append(next(lab_iter))
                else:
                    bert_labels[-1].append(WORDPIECE)

        outputs = outputs + (bert_labels, )

    return outputs  # bert_tokens, orig_tok_mask, (bert_labels)


def index_pad_mask_bert_tokens(tokens,
                               tokenizer,
                               maxlen=512,
                               labels=None,
                               orig_tok_mask=None,
                               tag_to_idx=None):
    """Convert `tokens` to indices, pads them, and generates the corresponding attention masks.

    Args:
        tokens (list): A list of lists containing tokenized sentences.
        tokenizer (BertTokenizer): An object with methods for tokenizing text for input to BERT.
        maxlen (int): The maximum length of a sentence. Any sentence longer than this length
            with be truncated, any sentence shorter than this length will be right-padded.
        labels (list): A list of lists containing token-level labels for a collection of sentences.
        orig_tok_mask (list). A list of list mapping token indices of pre-bert-tokenized text to
            token indices in post-bert-tokenized text.
        tag_to_idx (dictionary): A dictionary mapping token-level tags/labels to unique integers.

    Returns:
        If `labels` is not `None`:
            A tuple of `torch.Tensor`'s: `indexed_tokens`, `attention_mask`, and `indexed_labels`
            that can be used as input to to train a BERT model. Note that if `labels` is not `None`,
            `tag_to_idx` must also be provided.
        If `labels` is `None`:
            A tuple of `torch.Tensor`'s: `indexed_tokens`, and `attention_mask`, representing
            tokens mapped to indices and corresponding attention masks that can be used as input to
            a BERT model.
    """
    # Convert sequences to indices and pad
    indexed_tokens = pad_sequences(
        sequences=[tokenizer.convert_tokens_to_ids(sent) for sent in tokens],
        maxlen=maxlen,
        dtype='long',
        padding='post',
        truncating='post',
        value=tokenizer.convert_tokens_to_ids(PAD)
    )
    indexed_tokens = torch.as_tensor(indexed_tokens)

    # Generate attention masks for pad values
    attention_mask = torch.where(
        indexed_tokens == tokenizer.convert_tokens_to_ids(PAD),
        torch.zeros_like(indexed_tokens),
        torch.ones_like(indexed_tokens)
    )

    outputs = (indexed_tokens, attention_mask)

    if orig_tok_mask:
        orig_tok_mask = pad_sequences(
            sequences=orig_tok_mask,
            maxlen=maxlen,
            dtype='bool',
            padding='post',
            truncating='post',
            value=tokenizer.convert_tokens_to_ids(PAD)
        )
        orig_tok_mask = torch.as_tensor(orig_tok_mask)
        outputs = outputs + (orig_tok_mask, )

    indexed_labels = None
    if labels:
        indexed_labels = pad_sequences(
            sequences=[[tag_to_idx[lab] for lab in sent] for sent in labels],
            maxlen=maxlen,
            dtype='long',
            padding="post",
            truncating="post",
            value=tokenizer.convert_tokens_to_ids(PAD)
        )
        indexed_labels = torch.as_tensor(indexed_labels)
        outputs = outputs + (indexed_labels, )

    return outputs  # indexed_tokens, attention_mask, (orig_tok_mask), (indexed_labels)
