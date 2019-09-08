import json
import os

import torch
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus.reader.conll import ConllCorpusReader
from pytorch_transformers import BertTokenizer
from torch.data.utils import TensorDataset
from tqdm import tqdm

from constants import *
from data_utils import index_pad_mask_bert_tokens, wordpiece_tokenize_sents


def prepare_deid_dataset(args, tokenizer):
    conll_parser = ConllCorpusReader(args.dataset_folder, '.conll', ('words', 'pos'))

    type_list = ["train", "valid", "test"]
    all_dataset = dict.fromkeys(type_list)
    for data_type in tqdm(type_list):
        data_file = os.path.join("deid_data", data_type + ".tsv")
        sents = list(conll_parser.sents(data_file))
        tagged_sents = list(conll_parser.tagged_sents(data_file))

        maxlen = args.max_seq_len if data_type == "train" else BERT_MAX_SENT_LEN

        bert_tokens, orig_to_tok_map, bert_labels = \
            wordpiece_tokenize_sents(sents, tokenizer, tagged_sents)

        indexed_tokens, attention_mask, orig_to_tok_map, indexed_labels = \
            index_pad_mask_bert_tokens(
                bert_tokens, tokenizer, maxlen=maxlen, labels=bert_labels,
                orig_to_tok_map=orig_to_tok_map, tag_to_idx=DEID_LABELS
            )

        dataset_dict = {
            "indexed_tokens": indexed_tokens.tolist(),
            "attention_mask": attention_mask.tolist(),
            "orig_to_tok_map": orig_to_tok_map.tolist(),
            "indexed_labels": indexed_labels.tolist()
        }
        output_file = os.path.join(args.dataset_folder, "deid_data", "preprocessed")

        with open(output_file + f"/{data_type}.json", 'w') as open_file:
            open_file.write(json.dumps(dataset_dict))
    
        # all_dataset[data_type] = TensorDataset(indexed_tokens, attention_mask, indexed_labels, orig_to_tok_map)

    return all_dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", default=None, type=str, required=True,
                        help="De-id and co-hort identification data directory")
    args = parser.parse_args()

    bert_type = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_type)

    deid_dataset = prepare_deid_dataset(args, tokenizer)
