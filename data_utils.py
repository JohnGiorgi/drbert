import os
import json
import spacy
import torch

from tqdm import tqdm

from keras_preprocessing.sequence import pad_sequences
from nltk.corpus.reader.conll import ConllCorpusReader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from preprocess_cohort import read_charts, read_labels
from constants import *
from pytorch_transformers import BertTokenizer


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

def prepare_cohort_dataset(tokenizer, args):
    nlp = spacy.load("en_core_sci_md")
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
        return len(self.indexed_tokens)

    def __getitem__(self, index):
        token_ids = self.indexed_tokens[index]
        attn_mask = self.attention_mask[index]
        orig_tok_map = self.orig_to_tok_map[index]
        labels_idx = self.indexed_labels[index]
        return token_ids, attn_mask, orig_tok_map, labels_idx

def prepare_deid_dataset(tokenizer, args):
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
    
    # cohort_train_datasets = prepare_cohort_dataset(tokenizer, args)
    # train_datasets = cohort_train_datasets['train']
    # train_cohort_sampler = RandomSampler(train_datasets)
    # train_cohort_dataloader = DataLoader(train_datasets, sampler=train_cohort_sampler, batch_size=1)
    # for i, data_batch in enumerate(train_cohort_dataloader):
    #     print(i)
    #     print(data_batch)
    #     quit()



















