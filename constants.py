
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

COHORT_DISEASE_LIST = [
    'Obesity',
    'Diabetes',
    'Hypercholesterolemia',
    'Hypertriglyceridemia',
    'Hypertension',
    'CAD',
    'CHF',
    'PVD',
    'Venous Insufficiency',
    'OA',
    'OSA',
    'Asthma',
    'GERD',
    'Gallstones',
    'Depression',
    'Gout'
]


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