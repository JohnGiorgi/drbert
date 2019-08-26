import os

import spacy
import torch
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus.reader.conll import ConllCorpusReader
from torch.utils.data import TensorDataset

from preprocess_cohort import read_charts, read_labels

# TODO: Don't define this globally. Create it once and pass it to any function that needs it or
# have the function that need it init it within their scope.
nlp = spacy.load("en_core_sci_sm")

CONSTANTS = {
    'SEP': '[SEP]',
    'CLS': '[CLS]',
    'UNK': '[UNK]',
    'PAD': '[PAD]',
    'TOK_MAP_PAD': -100
}


def prepare_cohort_dataset(tokenizer):
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "diabetes_data")

    # charts format: test_charts[chart_id] = text # format
    input = read_charts(data_path)

    # labels format: test_labels[chart_id][disease_name] = judgement # format
    # example label: {'Asthma': 'N', 'CHF': 'Y', 'Depression': 'N', 'Diabetes': 'Y', 'Gallstones': 'N', 'Gout': 'N', 'Hypercholesterolemia': 'N', 'Hypertriglyceridemia': 'N', 'OA': 'Y', 'OSA': 'N'}
    # TODO: This is assigned to be never used?
    labels = read_labels(data_path)

    documents = []
    documents_padded = []
    attention_masks = []
    doc_ids = []

    for chart in input[0]:
        documents.append(input[0][chart])
        doc_ids.append(chart)

    for doc in documents:
        max_sent_len = 250
        doc = nlp(doc)

        sentence_list = [sentence for sentence in list(doc.sents)]
        token_list = [[str(token) for token in sentence] for sentence in sentence_list]

        padding_len = max_sent_len - len(sentence_list)
        pad = [CONSTANTS['PAD'] for i in range(512)]

        token_list += [pad for i in range(padding_len)]

        token_ids, attention_mask = index_pad_mask_bert_tokens(token_list, tokenizer)
        documents_padded.append(token_ids.unsqueeze(0))
        attention_masks.append(attention_mask.unsqueeze(0))

    documents_padded = torch.cat(documents_padded, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return TensorDataset(documents_padded, attention_masks)


def prepare_deid_dataset(tokenizer, args, is_train=True):
    conll_parser = ConllCorpusReader(args.dataset_folder, '.conll', ('words', 'pos'))

    sents = list(conll_parser.sents(args.data_file))
    tagged_sents = list(conll_parser.tagged_sents(args.data_file))
    max_sent_len = 512 if is_train else None

    assert len(sents) == len(tagged_sents)

    bert_tokens, orig_to_tok_map, bert_labels, tag_to_idx = wordpiece_tokenize_sents(sents, tokenizer, tagged_sents)

    indexed_tokens, orig_to_tok_map, attention_mask, indexed_labels = index_pad_mask_bert_tokens(bert_tokens, orig_to_tok_map, tokenizer, bert_labels, tag_to_idx, max_sent_len)

    return TensorDataset(indexed_tokens, attention_mask, indexed_labels, orig_to_tok_map)


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
    bert_labels = []
    tag_to_idx = dict()
    for sent in tokens:
        bert_tokens.append([CONSTANTS['CLS']])
        orig_to_tok_map.append([])
        for orig_token in sent:
            orig_to_tok_map[-1].append(len(bert_tokens[-1]))
            bert_tokens[-1].extend(tokenizer.wordpiece_tokenizer.tokenize(orig_token))
        bert_tokens[-1].append(CONSTANTS['SEP'])

    # If labels are provided, project them onto bert_tokens
    if labels is not None:
        for bert_toks, labs, tok_map in zip(bert_tokens, labels, orig_to_tok_map):
            labs_iter = iter(labs)
            bert_labels.append([])
            for i, _ in enumerate(bert_toks):
                bert_labels[-1].extend([CONSTANTS['WORDPIECE'] if i not in tok_map
                                        else next(labs_iter)])
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

    outputs = indexed_tokens, attention_mask

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

        outputs = outputs + orig_to_tok_map

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

        outputs = outputs + labels

    return outputs

# TODO: Delete when done with.
# document = "398811546 | TH | 30493757 | | 594476 | 9/15/2001 12:00:00 AM | CONGESTIVE HEART FAILURE | Signed | DIS | Admission Date: 9/15/2001 Report Status: Signed\n\nDischarge Date: 2/5/2001\nHISTORY OF PRESENT ILLNESS: Mr. Raffo is a 59-year-old male\nwith a history of coronary artery\ndisease status post small non-ST elevation myocardial infarction in\nOctober of 2000 and also status post cardiac catheterization with 2\nvessel disease , small left PICA cerebrovascular accident ,\ncongestive heart failure with an echocardiogram in October\nrevealing an ejection fraction of 30%. Also diabetes mellitus type\nII complicated by retinopathy , nephropathy and question neuropathy.\nHistory of hypertension , hypercholesterolemia as well who\npresented to Dr. Bouie clinic on May , 2001 with\nprogressive increasing dyspnea , increasing abdominal girth and\nweight gain over the past few months.\nPAST MEDICAL HISTORY: Diabetes mellitus x 20 years , chronic renal\ninsufficiency with the last creatinine in\nJanuary of 3.5 , congestive heart failure with an ejection fraction of\n30% in October of 2000 , coronary artery disease status post\nmyocardial infarction , October 2000 question f silent myocardial\ninfarction x 2 , history of hyperlipidemia.\nMEDICATIONS ON ADMISSION: Aspirin daily , Lasix 80 mg p.o. q day ,\nZaroxolyn 2.5 mg p.o. q day , toprol XL\n50 mg p.o. q day , insulin 70/30 65/45 , Actos 45 q p.m , Avapro 300\nmg p.o. q day , Lipitor 10 mg p.o. q.h.s. , sublingual nitroglycerin\np.r.n..\nPHYSICAL EXAM: Temperature 98.1 , blood pressure 116/70 , pulse 88 ,\nprimary care physician of 18. The patient appeared\nto be in no acute distress. Jugular venous pressure approximately\n13 cm water. LUNGS: Faint bibasilar rales. CARDIAC:\nTachycardia , regular rhythm without murmurs. ABDOMEN: Positive ,\nnormal active bowel sounds. Obese , distended and nontender.\nEXTREMITIES: 2 plus dorsalis pedis pulse , 2-3 plus edema\nbilaterally to the knees.\nHOSPITAL COURSE: 1. Cardiovascular: It was felt that given notes\nof thyroid dysfunction , his history was\nconsistent with fluid overload. His jugular venous pressure was\nelevated and he had 3plus pitting edema to the knees without\nsignificant rales on chest exam. He was aggressively diuresed with\ndoses of Lasix 200 mg b.i.d. IV as well as Zaroxolyn. Weight on\nadmission was 135 kilograms and on discharge he was down to 132\nkilograms. A repeat echocardiogram at Ethool Hospital\nshowed an ejection fraction of 30-35 , left ventricular dimensions\nof 47 mm , 1 plus mitral regurgitation and global hypokinesis as\nwell as moderate right ventricular dysfunction. Abdominal\nultrasound showed no ascites despite extremely distended abdomen\nand right renal cyst x 2. Follow up abdominal CT also showed there\nwas no mass or ascites and his distended abdomen was likely due to\nadipose tissue. Diuresis was limited by acute and chronic renal\nfailure and oliguria. Diuresis was held from May , 2001.\nRight heart catheterization on February , 2001 showed mildly\nelevated pulmonary capillary wedge pressure and moderately elevated\nright ventricular pressures. Dopamine was started on June ,\n2001 to aid with renal perfusion and diuresis and he was then\nweaned off of that on July , 2001. His Lasix was stopped on\nJuly , 2001 and he auto-diuresed until the time of discharge\nwith stable blood pressure. He remained stable with no dyspnea at\nrest and is symptom free at the time of discharge.\n2. Renal: His basic chronic renal insufficiency is likely\nsecondary to poor diabetic control with a creatinine of 2.5 on October , 2001 , 3.3 at the time of admission. Acute renal failure with\nincreasing creatinine of 6 after aggressive diuresis with a mean of\n0.8 percent. Renal function improved with creatinine of 4.1 on\nJuly , 2001 with autodiuresis. The renal service was\nconsulted to comment on the nature of the patient's renal\ninsufficiency and acute renal failure as well as to assist in the\nuse of vasodilators.\n3. Endocrine: He has a long standing history of type II diabetes\ncomplicated by retinopathy and neuropathy and nephropathy. He was\nmaintained on his outpatient endocrine regimen during his stay.\nThe patient was discharged home with services. He is to follow up\nwith Dr. Marro on November , 2001 at 3:30 p.m. , with Dr. Loatman on\nAugust , 2002 at 3:30 p.m. , Dr. Carcano January , 2001 at 10:00\na.m.\nMEDICATIONS ON DISCHARGE: Aspirin 325 mg p.o. q day , Lasix 80 mg\np.o. q day , Zocor 20 mg p.o. q.h.s. ,\ninsulin 70/30 65 units q a.m. , insulin 70/30 45 units q p.m. ,\nToprol XL 50 mg p.o. q day , Levaquin 250 mg p.o. q day for a\nduration of 7 days , Actos 45 mg p.o. q p.m.\nCONDITION ON DISCHARGE: Stable.\nDictated By: BENJAMIN DOUYON , M.D. OF45\nAttending: RUDOLF D. FEDERKEIL , M.D. YI70  OL023/102273\nBatch: 54454 Index No. D0NUNVNAP2 D: 11/2/02\nT: 11/2/02"
# print(prepare_cohort_dataset())
