import json
import os

import spacy
import torch
from pytorch_transformers import BertTokenizer
from tqdm import tqdm

from .constants import (BERT_MAX_SENT_LEN, COHORT_DISEASE_CONSTANTS,
                        COHORT_INTUITIVE_LABEL_CONSTANTS,
                        COHORT_TEXTUAL_LABEL_CONSTANTS, MAX_COHORT_NUM_SENTS)
from .preprocess_cohort import (read_charts_intuitive, read_charts_textual,
                                read_labels_intuitive, read_labels_textual)
from .utils.data_utils import (index_pad_mask_bert_tokens,
                               wordpiece_tokenize_sents)


def prepare_cohort_dataset(args, tokenizer):
    nlp = spacy.load("en_core_sci_md")
    data_path = os.path.join(args.dataset_folder, "cohort")

    # charts format: test_charts[chart_id] = text # format
    if args.cohort_type == "textual":
        inputs_preprocessed = read_charts_textual(data_path)
        labels_preprocessed = read_labels_textual(data_path)
    else:  # intuitive
        inputs_preprocessed = read_charts_intuitive(data_path)
        labels_preprocessed = read_labels_intuitive(data_path)
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
            maxlen=args.max_seq_length,
            tag_to_idx=COHORT_DISEASE_CONSTANTS
        )

        labels_array = torch.zeros(16)

        for disease, judgement in label.items():
            if args.cohort_type == "textual":
                judgement = COHORT_TEXTUAL_LABEL_CONSTANTS[judgement]
            elif args.cohort_type == "intuitive":
                judgement = COHORT_INTUITIVE_LABEL_CONSTANTS[judgement]

            labels_array[COHORT_DISEASE_CONSTANTS[disease]] = judgement

        chart_data = {
            "input_ids": indexed_tokens.tolist(),
            "attn_mask": attention_mask.tolist(),
            "labels": labels_array.tolist()
        }

        preproc_folder = "preprocessed"
        '''
        if args.cohort_type == "textual":
            preproc_folder += "_textual"
        elif args.cohort_type == "intuitive":
            preproc_folder += "_intuitive"
        '''

        filepath = os.path.join(data_path, preproc_folder, args.type, f"{chart_id}.json")
        with open(filepath, 'w') as open_file:
            open_file.write(json.dumps(chart_data))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", default=None, type=str, required=True,
                        help="De-id and co-hort identification data directory")
    parser.add_argument("--type", default='train', type=str,
                        help="train valid test")
    parser.add_argument("--cohort_type", default='textual', choices=['textual', 'intuitive'], 
                        type=str, help="what do you think this is")
    parser.add_argument("--max_seq_length", default=60, type=int,
                        help=("The maximum total input sequence length after WordPiece"
                              " tokenization. Sequences longer than this will be truncated, and"
                              " sequences shorter than this will be padded."))
    args = parser.parse_args()

    bert_type = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_type, do_lower_case=True)

    cohort_train_dataset = prepare_cohort_dataset(args, tokenizer)
