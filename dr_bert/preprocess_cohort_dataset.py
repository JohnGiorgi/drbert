import json
import os

import spacy
import torch
from pytorch_transformers import BertTokenizer
from tqdm import tqdm

from .constants import (COHORT_DISEASE_CONSTANTS, COHORT_LABEL_CONSTANTS,
                        MAX_COHORT_NUM_SENTS)
from .preprocess_cohort import read_charts, read_labels
from .utils.data_utils import (index_pad_mask_bert_tokens,
                               wordpiece_tokenize_sents)


def prepare_cohort_dataset(tokenizer, args):
    nlp = spacy.load("en_core_sci_md")
    data_path = os.path.join(args.dataset_folder, "diabetes_data")

    # charts format: test_charts[chart_id] = text # format
    inputs_preprocessed = read_charts(data_path)

    # labels format: test_labels[chart_id][disease_name] = judgement # format
    labels_preprocessed = read_labels(data_path)

    if args.type == "train" or args.type == "valid":
        inputs = inputs_preprocessed[0]
        labels = labels_preprocessed[0]
    else:
        inputs = inputs_preprocessed[1]
        labels = labels_preprocessed[1]

    chart_ids = list(labels.keys())

    # TODO (Gary, Nick): This is different than what was provided to train.py.
    max_sent_len = args.max_seq_len

    split = int(len(chart_ids) * 0.8)
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

        sentence_list = [sentence for sentence in list(doc.sents)]
        sentence_list = sentence_list[:MAX_COHORT_NUM_SENTS]  # clip
        token_list = [[str(token) for token in sentence] for sentence in sentence_list]
        token_list, _, _, _ = wordpiece_tokenize_sents(token_list, tokenizer)

        # no more sentence padding
        # num_extra_sentences = MAX_COHORT_NUM_SENTS - len(token_list)
        # for i in range(num_extra_sentences):
        #     sentence_padding = [CONSTANTS['PAD']] * max_sent_len
        #     token_list.append(sentence_padding)
        # TODO (Gary, Nick): Maxlen is hardcoded. It should be whatever was provided to train.py
        token_ids, attention_mask, _, indexed_labels = \
            index_pad_mask_bert_tokens(token_list, tokenizer, maxlen=256, tag_to_idx=COHORT_DISEASE_CONSTANTS)

        labels_array = torch.zeros(16)

        for disease, judgement in label.items():
            judgement = COHORT_LABEL_CONSTANTS[judgement]
            labels_array[COHORT_DISEASE_CONSTANTS[disease]] = judgement

        chart_data = {
            "input_ids": token_ids.tolist(),
            "attn_mask": attention_mask.tolist(),
            "labels": labels_array.tolist()
        }

        with open(os.path.join(data_path, "preprocessed", args.type) + f"/{chart_id}.json", 'w') as open_file:
            open_file.write(json.dumps(chart_data))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", default=None, type=str, required=True,
                        help="De-id and co-hort identification data directory")
    parser.add_argument("--type", default='train', type=str,
                        help="train valid test")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="max seq len")
    args = parser.parse_args()

    bert_type = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_type)

    cohort_train_dataset = prepare_cohort_dataset(tokenizer, args)
