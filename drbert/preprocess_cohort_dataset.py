import json
import os

import spacy
import torch
from pytorch_transformers import BertTokenizer
from tqdm import tqdm

from .constants import (BERT_MAX_SENT_LEN, COHORT_DISEASE_CONSTANTS,
                        COHORT_LABEL_CONSTANTS, MAX_COHORT_NUM_SENTS)
from .preprocess_cohort import read_charts, read_labels
from .utils.data_utils import (index_pad_mask_bert_tokens,
                               wordpiece_tokenize_sents)


def prepare_cohort_dataset(args, tokenizer):
    nlp = spacy.load("en_core_sci_md")
    data_path = os.path.join(args.dataset_folder, "cohort")

    # charts format: test_charts[chart_id] = text # format
    inputs_preprocessed = read_charts(data_path)

    # labels format: test_labels[chart_id][disease_name] = judgement # format
    labels_preprocessed = read_labels(data_path)

    maxlen = args.max_seq_length if args.type == "train" else BERT_MAX_SENT_LEN

    if args.type == "train" or args.type == "valid":
        inputs = inputs_preprocessed[0]
        labels = labels_preprocessed[0]

    else:
        inputs = inputs_preprocessed[1]
        labels = labels_preprocessed[1]

    chart_ids = list(labels.keys())

    split = int(len(chart_ids) * 0.9)
    print(f"total data {len(chart_ids)}")
    if args.type == "train":
        chart_ids = chart_ids[:split]
        print(f"train split {len(chart_ids)}")
    elif args.type == "valid":
        chart_ids = chart_ids[split:]
        print(f"valid split {len(chart_ids)}")

    for chart_id in tqdm(chart_ids):
        chart = inputs[chart_id]
        label = labels[chart_id]

        doc = nlp(chart)

        sentences = list(doc.sents)[:MAX_COHORT_NUM_SENTS]  # clip
        token_list = [[token.text for token in sentence] for sentence in sentences]
        bert_tokens = wordpiece_tokenize_sents(token_list, tokenizer)[0]

        indexed_tokens, attention_mask = index_pad_mask_bert_tokens(
            tokens=bert_tokens,
            tokenizer=tokenizer,
            maxlen=maxlen,
            tag_to_idx=COHORT_DISEASE_CONSTANTS
        )

        labels_array = torch.zeros(16)

        for disease, judgement in label.items():
            judgement = COHORT_LABEL_CONSTANTS[judgement]
            labels_array[COHORT_DISEASE_CONSTANTS[disease]] = judgement

        chart_data = {
            "input_ids": indexed_tokens.tolist(),
            "attn_mask": attention_mask.tolist(),
            "labels": labels_array.tolist()
        }

        filepath = os.path.join(data_path, "preprocessed", args.type, f"{chart_id}.json")
        with open(filepath, 'w') as open_file:
            open_file.write(json.dumps(chart_data))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", default=None, type=str, required=True,
                        help="De-id and co-hort identification data directory")
    parser.add_argument("--type", default='train', type=str,
                        help="train valid test")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help=("The maximum total input sequence length after WordPiece"
                              " tokenization. Sequences longer than this will be truncated, and"
                              " sequences shorter than this will be padded."))
    args = parser.parse_args()

    bert_type = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_type)

    cohort_train_dataset = prepare_cohort_dataset(args, tokenizer)
