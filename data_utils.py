import os

import spacy
import torch
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus.reader.conll import ConllCorpusReader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from preprocess_cohort import read_charts, read_labels
from pytorch_transformers import BertTokenizer

CONSTANTS = {
    'SEP': '[SEP]',
    'CLS': '[CLS]',
    'UNK': '[UNK]',
    'PAD': '[PAD]',
    'TOK_MAP_PAD': -100,
    'WORDPIECE': 'X'
}

COHORT_LABEL_CONSTANTS = {
    'U': 0,
    'Q': 1,
    'N': 2,
    'Y': 3,
}

COHORT_DISEASE_CONSTANTS = {
    'Obesity': 0,
    'Diabetes': 1,
    'Hypercholesterolemia': 2,
    'Hypertriglyceridemia': 3,
    'Hypertension': 4,
    'CAD': 5, 
    'CHF': 6,
    'PVD': 7,
    'Venous Insufficiency': 8,
    'OA': 9,
    'OSA': 10,
    'Asthma': 11,
    'GERD': 12,
    'Gallstones': 13,
    'Depression': 14,
    'Gout': 15
}

MAX_COHORT_NUM_SENTS = 256 # number of sentences in chart
MAX_COHORT_NUM_TOKENS = 64 # sentence length

def prepare_cohort_dataset(tokenizer, args, is_train=True):
    nlp = spacy.load("en_core_sci_md")
    data_path = os.path.join(args.dataset_folder, "diabetes_data")
    
    #charts format: test_charts[chart_id] = text # format
    inputs_preprocessed = read_charts(data_path)
    
    #labels format: test_labels[chart_id][disease_name] = judgement # format
    labels_preprocessed = read_labels(data_path)
    
    inputs_list = []
    inputs_ids = []
    inputs_padded = []
    attention_masks = []
    
    labels_list = []
    labels_ids = []
    labels_tensorized = []
    
    if is_train: 
        inputs = inputs_preprocessed[0]
        labels = labels_preprocessed[0]
    else: 
        inputs = inputs_preprocessed[1]
        labels = labels_preprocessed[1] 

    # process inputs
    for chart_id, chart in inputs.items(): # chart_id, text
        
        doc = nlp(chart)

        sentence_list = [sentence for sentence in list(doc.sents)]
        sentence_list = sentence_list[:MAX_COHORT_NUM_SENTS]

        token_list = [[str(token) for token in sentence] for sentence in sentence_list]
        # truncate sentence list at max len
        for i, tokens in enumerate(token_list):
            padding_len = MAX_COHORT_NUM_TOKENS - len(tokens)
            if padding_len <= 0:
                token_list[i] = tokens[:MAX_COHORT_NUM_TOKENS]
            else:
                padding = [CONSTANTS['PAD']] * padding_len
                token_list[i].extend(padding)
        
        token_ids, attention_mask, _, indexed_labels = index_pad_mask_bert_tokens(token_list, tokenizer, tag_to_idx=COHORT_DISEASE_CONSTANTS)
        print(f"token_ids: {token_ids.shape}")
        print(f"attention_mask: {attention_mask.shape}")
        quit()
        inputs_padded.append(token_ids.unsqueeze(0))
        attention_masks.append(attention_mask.unsqueeze(0))

    #process labels
    for i in range(len(labels_list)):
        labels_array = torch.zeros(16)

        for disease in COHORT_DISEASE_CONSTANTS:
            if disease in labels_list[i]:
                score = COHORT_LABEL_CONSTANTS[labels_list[i][disease]]
                labels_array[COHORT_DISEASE_CONSTANTS[disease]] = score

        labels_tensorized.append(labels_array.unsqueeze(0))
        
    inputs_padded = torch.cat(inputs_padded, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_tensorized = torch.cat(labels_tensorized, dim=0)

    return TensorDataset(inputs_padded, attention_masks, labels_tensorized)
    
def prepare_deid_dataset(tokenizer, args, is_train=True):
    conll_parser = ConllCorpusReader(args.dataset_folder, '.conll', ('words', 'pos'))
    if is_train:
        data_file = "train.tsv"
    else:
        data_file = "test.tsv"
    sents = list(conll_parser.sents(data_file))
    tagged_sents = list(conll_parser.tagged_sents(data_file))
    max_sent_len = 512 if is_train else None

    assert len(sents) == len(tagged_sents)

    bert_tokens, orig_to_tok_map, bert_labels, tag_to_idx = wordpiece_tokenize_sents(sents, tokenizer, tagged_sents)

    indexed_tokens, attention_mask, orig_to_tok_map, indexed_labels = \
        index_pad_mask_bert_tokens(bert_tokens, tokenizer, maxlen=max_sent_len, labels=bert_labels, orig_to_tok_map=orig_to_tok_map, tag_to_idx=tag_to_idx)

    return TensorDataset(indexed_tokens, attention_mask, indexed_labels, orig_to_tok_map)


def wordpiece_tokenize_sents(tokens, tokenizer, sentence_labels=None):
    """Tokenizes pre-tokenized text for use with a BERT-based model.

    Given some pre-tokenized text, represented as a list (sentences) of lists (tokens), tokenizies
    the text for use with a BERT-based model while deterministically maintaining an
    original-to-tokenized alignment. This is a near direct copy of the example given in the BERT
    GitHub repo (https://github.com/google-research/bert#tokenization) with additional code for
    mapping token-level labels.

    Args:
        tokens (list): A list of lists containing tokenized sentences.
        tokenizer (BertTokenizer): An object with methods for tokenizing text for input to BERT.
        sentence_labels (list): Optional, a list of lists containing token-level labels for a collection of
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
    bert_labels = []
    tag_to_idx = dict()
    for sent in tokens:
        bert_tokens.append([CONSTANTS['CLS']])
        orig_to_tok_map.append([])
        for orig_token in sent:
            orig_to_tok_map[-1].append(len(bert_tokens[-1]))
            bert_tokens[-1].extend(tokenizer.wordpiece_tokenizer.tokenize(orig_token))
        bert_tokens[-1].append(CONSTANTS['SEP'])

    # If sentence_labels are provided, project them onto bert_tokens
    if sentence_labels is not None:
        for bert_toks, labs, tok_map in zip(bert_tokens, sentence_labels, orig_to_tok_map):
            labs_iter = iter(labs)
            bert_labels.append([])
            for i, _ in enumerate(bert_toks):
                bert_labels[-1].extend([CONSTANTS['WORDPIECE'] if i not in tok_map
                                        else next(labs_iter)])

        for labels in sentence_labels:
            for label in labels:
                if label not in tag_to_idx:
                    tag_to_idx[label] = len(tag_to_idx) + 1
        tag_to_idx[CONSTANTS['WORDPIECE']] = len(tag_to_idx) + 1
    return bert_tokens, orig_to_tok_map, bert_labels, tag_to_idx


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
    CONSTANTS['MAX_SENT_LEN'] = maxlen
    # Convert sequences to indices and pad
    indexed_tokens = pad_sequences(
        sequences=[tokenizer.convert_tokens_to_ids(sent) for sent in tokens],
        maxlen=CONSTANTS['MAX_SENT_LEN'],
        dtype='long',
        padding='post',
        truncating='post',
        value=tokenizer.convert_tokens_to_ids([CONSTANTS['PAD']])
    )
    indexed_tokens = torch.as_tensor(indexed_tokens)

    # Generate attention masks for pad values
    attention_mask = torch.as_tensor([[float(idx > 0) for idx in sent] for sent in indexed_tokens])

    if orig_to_tok_map:
        orig_to_tok_map = pad_sequences(
            sequences=orig_to_tok_map,
            maxlen=CONSTANTS['MAX_SENT_LEN'],
            dtype='long',
            padding='post',
            truncating='post',
            value=tokenizer.convert_tokens_to_ids([CONSTANTS['TOK_MAP_PAD']])
        )
        orig_to_tok_map = torch.as_tensor(orig_to_tok_map)

    indexed_labels = None
    if labels:
        indexed_labels = pad_sequences(
            sequences=[[tag_to_idx[lab] for lab in sent] for sent in labels],
            maxlen=CONSTANTS['MAX_SENT_LEN'],
            dtype='long',
            padding="post",
            truncating="post",
            value=tokenizer.convert_tokens_to_ids([CONSTANTS['PAD']])
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

    # deid_train_dataset = prepare_deid_dataset(tokenizer, args, is_train=True)
    # train_deid_sampler = RandomSampler(deid_train_dataset)
    # train_deid_dataloader = DataLoader(deid_train_dataset, sampler=train_deid_sampler, batch_size=args.train_batch_size)
    # for step, (indexed_tokens, attention_mask, indexed_labels, orig_to_tok_map) in enumerate(train_deid_dataloader):
    #     print(f"step: {step}")
    #     print(f"indexed_tokens: {indexed_tokens}")
    #     print(f"attention_mask: {attention_mask}")
    #     print(f"indexed_labels: {indexed_labels}")
    #     print(f"orig_to_tok_map: {orig_to_tok_map}")
    #     break

    cohort_train_dataset = prepare_cohort_dataset(tokenizer, args)
    train_cohort_sampler = RandomSampler(cohort_train_dataset)
    train_cohort_dataloader = DataLoader(cohort_train_dataset, sampler=train_cohort_sampler, batch_size=1)



















