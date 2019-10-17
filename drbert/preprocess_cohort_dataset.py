import json
import os

import spacy
import torch
from transformers import BertTokenizer
from tqdm import tqdm

from .constants import (COHORT_DISEASE_CONSTANTS,
                        COHORT_INTUITIVE_LABEL_CONSTANTS,
                        COHORT_TEXTUAL_LABEL_CONSTANTS, MAX_COHORT_NUM_SENTS)
from .preprocess_cohort import (read_charts_intuitive, read_charts_textual,
                                read_labels_intuitive, read_labels_textual)
from .utils.data_utils import (index_pad_mask_bert_tokens,
                               wordpiece_tokenize_sents)

def encode_documents(documents: list, tokenizer: BertTokenizer, max_input_length=512):
    """
    Returns a len(documents) * max_sequences_per_document * 3 * 512 tensor where len(documents) is the batch
    dimension and the others encode bert input.
    This is the input to any of the document bert architectures.
    :param documents: a list of text documents
    :param tokenizer: the sentence piece bert tokenizer
    :return:
    """
    tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    max_sequences_per_document = math.ceil(max(len(x)/(max_input_length-2) for x in tokenized_documents))
    assert max_sequences_per_document <= 20, "Your document is to large, arbitrary size when writing"

    output = torch.zeros(size=(max_sequences_per_document, 3, 512), dtype=torch.long)
    document_seq_lengths = [] #number of sequence generated per document
    #Need to use 510 to account for 2 padding tokens
    for doc_index, tokenized_document in enumerate(tokenized_documents):
        max_seq_index = 0
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length-2))):
            raw_tokens = tokenized_document[i:i+(max_input_length-2)]
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)

            assert len(input_ids) == 512 and len(attention_masks) == 512 and len(input_type_ids) == 512

            #we are ready to rumble
            output[seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                           torch.LongTensor(input_type_ids).unsqueeze(0),
                                                           torch.LongTensor(attention_masks).unsqueeze(0)),
                                                          dim=0)
            max_seq_index = seq_index
        document_seq_lengths.append(max_seq_index+1)
    return output, torch.LongTensor(document_seq_lengths)

def prepare_cohort_dataset_long(args, tokenizer):
    nlp = spacy.load("en_core_sci_md")
    data_path = os.path.join(args.dataset_folder, "cohort")

    # Prepares the full datasets
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

    # Train Test Val Split
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

        doc = nlp(chart) # Not sure if this line is needed

        chart_representation , chart_sequence_length  = encode_documents(doc, tokenizer)
        # chart_representations is a concatenated tensor of tokens, token_type_ids, and attention_masks

        labels_array = torch.zeros(16)

        for disease, judgement in label.items():
            if args.cohort_type == "textual":
                judgement = COHORT_TEXTUAL_LABEL_CONSTANTS[judgement]
            elif args.cohort_type == "intuitive":
                judgement = COHORT_INTUITIVE_LABEL_CONSTANTS[judgement]

            labels_array[COHORT_DISEASE_CONSTANTS[disease]] = judgement

        chart_representation_chunks = torch.chunk(chart_representation, 3)
        chart_input_ids = chart_representation_chunks[0]
        chart_input_type_ids = chart_representation_chunks[1]
        chart_attention_mask = chart_representation_chunks[2]

        chart_data = {
            "chart_input_ids": chart_input_ids.tolist(),
            "chart_input_type_ids": chart_input_type_ids.tolist(),
            "chart_attention_mask": chart_attention_mask.tolist(),
            "labels": labels_array.tolist()
        }

        # Save into preprocessed folder (this is a bit of an issue since I already have input ids etc and not raw tokens.)
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

    bert_type = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_type, do_lower_case=False)

    cohort_train_dataset = prepare_cohort_dataset(args, tokenizer)
#    cohort_long_train_dataset = prepare_cohort_dataset_long(args, tokenizer)
