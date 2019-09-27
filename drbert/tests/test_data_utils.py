"""Test suite for the `data_utils` module (drbert.utils.data_utils).
"""
import torch

from ..constants import BERT_MAX_SENT_LEN, CLS, PAD, SEP, WORDPIECE
from ..utils import data_utils

# Value reused across multiple tests
tokens = [["john", "johanson", "'s",  "house"], ["who", "was", "jim", "henson",  "?"]]
bert_tokens = [
    [CLS, "john", "johan", "##son", "'", "##s",  "house", SEP],
    [CLS, "who", "was", "jim", "henson", "?", SEP]
]

labels = [["B-PER", "I-PER", "I-PER",  "O"], ["O", "O", "B-PER", "I-PER",  "O"]]
bert_labels = [
    [WORDPIECE, "B-PER", "I-PER", WORDPIECE, "I-PER", WORDPIECE, "O", WORDPIECE],
    [WORDPIECE, "O", "O", "B-PER", "I-PER", "O", WORDPIECE]
]

orig_tok_mask = [[0, 1, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0]]

tag_to_idx = {
    PAD: 0,
    'O': 1,
    'B-PER': 2,
    'I-PER': 3,
    WORDPIECE: 4
}

attention_mask = torch.as_tensor([
    [1.] * len(bert_tokens[0]) + [0.] * (BERT_MAX_SENT_LEN - len(bert_tokens[0])),
    [1.] * len(bert_tokens[1]) + [0.] * (BERT_MAX_SENT_LEN - len(bert_tokens[1])),
], dtype=torch.long)


class TestBertUtils(object):
    """Collects all unit tests for `saber.utils.data_utils`.
    """
    def test_wordpiece_tokenize_sents(self, bert_tokenizer):
        """Asserts that `data_utils.wordpiece_tokenize_sents()` returns the expected values for a
        simple input when input argument `labels` is None.
        """
        expected = (bert_tokens, orig_tok_mask)

        actual = data_utils.wordpiece_tokenize_sents(tokens, tokenizer=bert_tokenizer)

        assert expected == actual

    def test_wordpiece_tokenize_sents_labels(self, bert_tokenizer):
        """Asserts that `data_utils.wordpiece_tokenize_sents()` returns the expected values for a
        simple input when input argument `labels` is not None.
        """
        expected = (bert_tokens, orig_tok_mask, bert_labels)

        actual = \
            data_utils.wordpiece_tokenize_sents(tokens, tokenizer=bert_tokenizer, labels=labels)

        assert expected == actual

    def test_index_pad_mask_bert_tokens(self, bert_tokenizer):
        """Asserts that `data_utils.index_pad_mask_bert_tokens()` returns the expected values for a
        simple input when input argument `labels` is None.
        """
        actual_indexed_tokens, actual_attention_mask, actual_orig_tok_mask = \
            data_utils.index_pad_mask_bert_tokens(bert_tokens,
                                                  orig_tok_mask=orig_tok_mask,
                                                  tokenizer=bert_tokenizer)

        expected_orig_tok_mask = torch.as_tensor(
            [tm + [0] * (BERT_MAX_SENT_LEN - len(tm)) for tm in orig_tok_mask],
            dtype=torch.bool
        )

        print(actual_orig_tok_mask, expected_orig_tok_mask)

        # Just check for shape, as token indicies will depend on specific BERT model used
        assert actual_indexed_tokens.shape == (2, BERT_MAX_SENT_LEN)
        assert torch.equal(expected_orig_tok_mask, actual_orig_tok_mask)
        assert torch.equal(attention_mask, actual_attention_mask)

    def test_index_pad_mask_bert_tokens_labels(self, bert_tokenizer):
        """Asserts that `data_utils.index_pad_mask_bert_tokens()` returns the expected values for a
        simple input when input argument `labels` is not None.
        """
        (actual_indexed_tokens, actual_attention_mask, actual_orig_tok_mask,
         actual_indexed_labels) = \
            data_utils.index_pad_mask_bert_tokens(tokens=bert_tokens,
                                                  orig_tok_mask=orig_tok_mask,
                                                  tokenizer=bert_tokenizer,
                                                  labels=bert_labels,
                                                  tag_to_idx=tag_to_idx)

        expected_orig_tok_mask = torch.as_tensor(
            [tm + [tag_to_idx[PAD]] * (BERT_MAX_SENT_LEN - len(tm)) for tm in orig_tok_mask],
            dtype=torch.bool
        )
        expected_indexed_labels = torch.tensor(
            [[tag_to_idx[lab] for lab in sent] + [0] * (BERT_MAX_SENT_LEN - len(sent))
             for sent in bert_labels]
        )

        # Just check for shape, as token indicies will depend on specific BERT model used
        assert actual_indexed_tokens.shape == (2, BERT_MAX_SENT_LEN)
        assert torch.equal(expected_orig_tok_mask, actual_orig_tok_mask)
        assert torch.equal(attention_mask, actual_attention_mask)
        assert torch.equal(expected_indexed_labels, actual_indexed_labels)
