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
    attention_mask = torch.where(
        indexed_tokens == tokenizer.convert_tokens_to_ids(PAD),
        torch.zeros_like(indexed_tokens),
        torch.ones_like(indexed_tokens)
    )

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
        # The map cant contain an index outside the maximum sequence length
        orig_to_tok_map[orig_to_tok_map >= maxlen] = TOK_MAP_PAD

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