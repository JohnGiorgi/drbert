import json
import os

import torch
from keras_preprocessing.sequence import pad_sequences
from pytorch_transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler

from constants import *


class CohortDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.file_list = os.listdir(data_path)
        self.file_list = [os.path.join(data_path, file_name) for file_name in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]

        with open(file_name) as open_file:
            read_file = json.loads(open_file.read())

        input_ids = torch.LongTensor(read_file['input_ids'])
        attn_mask = torch.LongTensor(read_file['attn_mask'])
        labels = torch.LongTensor(read_file['labels'])

        return input_ids, attn_mask, labels


def prepare_cohort_dataset(args, tokenizer):
    train_data_path = os.path.join(args.dataset_folder, "diabetes_data", "preprocessed", "train")
    valid_data_path = os.path.join(args.dataset_folder, "diabetes_data", "preprocessed", "valid")
    test_data_path = os.path.join(args.dataset_folder, "diabetes_data", "preprocessed", "test")

    train_cohort_dataset = CohortDataset(train_data_path)
    valid_cohort_dataset = CohortDataset(valid_data_path)
    test_cohort_dataset = CohortDataset(test_data_path)

    all_dataset = {
        'train': train_cohort_dataset,
        'valid': valid_cohort_dataset,
        'test': test_cohort_dataset
    }

    return all_dataset


class DeidDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        with open(self.data_file + ".json") as open_file:
            read_file = json.loads(open_file.read())

        self.indexed_tokens = torch.LongTensor(read_file["indexed_tokens"])
        self.attention_mask = torch.LongTensor(read_file["attention_mask"])
        self.orig_to_tok_map = torch.LongTensor(read_file["orig_to_tok_map"])
        self.indexed_labels = torch.LongTensor(read_file["indexed_labels"])

    def __len__(self):
        return self.indexed_tokens.size(0)

    def __getitem__(self, index):
        token_ids = self.indexed_tokens[index]
        attn_mask = self.attention_mask[index]
        orig_tok_map = self.orig_to_tok_map[index]
        labels_idx = self.indexed_labels[index]
        return token_ids, attn_mask, orig_tok_map, labels_idx


def prepare_deid_dataset(args, tokenizer):
    train_data_path = os.path.join(args.dataset_folder, "deid_data", "preprocessed", "train")
    valid_data_path = os.path.join(args.dataset_folder, "deid_data", "preprocessed", "valid")
    test_data_path = os.path.join(args.dataset_folder, "deid_data", "preprocessed", "test")

    train_deid_dataset = DeidDataset(train_data_path)
    valid_deid_dataset = DeidDataset(valid_data_path)
    test_deid_dataset = DeidDataset(test_data_path)

    all_dataset = {
        'train': train_deid_dataset,
        'valid': valid_deid_dataset,
        'test': test_deid_dataset
    }

    return all_dataset


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
            A tuple of `bert_tokens`, `orig_to_tok_map`, `bert_labels`, representing tokens and
            labels that can be used to train a BERT model and a deterministc mapping of the elements
            in `bert_tokens` to `tokens`.
        If `labels` is `None`:
            A tuple of `bert_tokens`, and `orig_to_tok_map`, representing tokens that can be used to
            train a BERT model and a deterministc mapping of `bert_tokens` to `sents`.

    References:
     - https://github.com/google-research/bert#tokenization
    """
    bert_tokens = []
    orig_to_tok_map = []

    for sent in tokens:
        bert_tokens.append([CLS])
        orig_to_tok_map.append([])
        for orig_token in sent:
            orig_to_tok_map[-1].append(len(bert_tokens[-1]))
            bert_tokens[-1].extend(tokenizer.wordpiece_tokenizer.tokenize(orig_token))
        bert_tokens[-1].append(SEP)

    # If labels are provided, project them onto bert_tokens
    if labels is not None:
        bert_labels = []
        for bert_toks, labs, tok_map in zip(bert_tokens, labels, orig_to_tok_map):
            labs_iter = iter(labs)
            bert_labels.append([])
            for i, _ in enumerate(bert_toks):
                bert_labels[-1].extend([WORDPIECE if i not in tok_map else next(labs_iter)])

        return bert_tokens, orig_to_tok_map, bert_labels

    return bert_tokens, orig_to_tok_map


def index_pad_mask_bert_tokens(tokens,
                               tokenizer,
                               maxlen=512,
                               labels=None,
                               orig_to_tok_map=None,
                               tag_to_idx=None):
    """Convert `tokens` to indices, pads them, and generates the corresponding attention masks.

    Args:
        tokens (list): A list of lists containing tokenized sentences.
        tokenizer (BertTokenizer): An object with methods for tokenizing text for input to BERT.
        maxlen (int): The maximum length of a sentence. Any sentence longer than this length
            with be truncated, any sentence shorter than this length will be right-padded.
        labels (list): A list of lists containing token-level labels for a collection of sentences.
        orig_to_tok_map (list). A list of list mapping token indices of pre-bert-tokenized text to
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
    attention_mask = torch.as_tensor([[float(idx > 0) for idx in sent] for sent in indexed_tokens])

    if orig_to_tok_map:
        orig_to_tok_map = pad_sequences(
            sequences=orig_to_tok_map,
            maxlen=maxlen,
            dtype='long',
            padding='post',
            truncating='post',
            value=TOK_MAP_PAD
        )
        orig_to_tok_map = torch.as_tensor(orig_to_tok_map)

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

    return indexed_tokens, attention_mask, orig_to_tok_map, indexed_labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", default=None, type=str, required=True,
                        help="De-id and co-hort identification data directory")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="train batch size")
    args = parser.parse_args()

    bert_type = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_type)

    deid_dataset = prepare_deid_dataset(tokenizer, args)
    deid_train_dataset = deid_dataset['train']
    train_deid_sampler = RandomSampler(deid_train_dataset)
    train_deid_dataloader = DataLoader(deid_train_dataset, sampler=train_deid_sampler, batch_size=args.train_batch_size)
    for step, (indexed_tokens, attention_mask, indexed_labels, orig_to_tok_map) in enumerate(train_deid_dataloader):
        print(f"step: {step}")
        print(f"indexed_tokens: {indexed_tokens}")
        print(f"attention_mask: {attention_mask}")
        print(f"indexed_labels: {indexed_labels}")
        print(f"orig_to_tok_map: {orig_to_tok_map}")
        break

    cohort_train_datasets = prepare_cohort_dataset(tokenizer, args)
    train_datasets = cohort_train_datasets['train']
    train_cohort_sampler = RandomSampler(train_datasets)
    train_cohort_dataloader = DataLoader(train_datasets, sampler=train_cohort_sampler, batch_size=1)
    for i, (input_ids, attn_mask, labels) in enumerate(train_cohort_dataloader):
        print(i)
        print(input_ids)
        print(attn_mask)
        print(labels)
        quit()







