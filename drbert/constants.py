OUTSIDE = 'O'
SEP = '[SEP]'
CLS = '[CLS]'
UNK = '[UNK]'
PAD = '[PAD]'
WORDPIECE = 'X'
BERT_MAX_SENT_LEN = 512

TASKS = {'sequence_labelling', 'sequence_classification', 'document_classification'}

DEID_TYPES = [
    PAD,
    'E-AGE',
    'S-ORGANIZATION',
    'I-AGE',
    'E-PATIENT',
    'O',
    'I-STATE',
    'S-PATIENT',
    'E-DEVICE',
    'B-AGE',
    'I-PATIENT',
    'E-PHONE',
    'E-HEALTHPLAN',
    'I-STREET',
    'I-ZIP',
    'B-STATE',
    'B-MEDICALRECORD',
    'S-LOCATION-OTHER',
    'E-COUNTRY',
    'S-MEDICALRECORD',
    'B-DOCTOR',
    'E-PROFESSION',
    'B-HOSPITAL',
    'B-LOCATION-OTHER',
    'E-DOCTOR',
    'I-FAX',
    'S-DOCTOR',
    'B-PATIENT',
    'E-MEDICALRECORD',
    'I-DATE',
    'S-CITY',
    'B-EMAIL',
    'I-LOCATION-OTHER',
    'B-HEALTHPLAN',
    'S-COUNTRY',
    'B-CITY',
    'I-IDNUM',
    'E-URL',
    'E-FAX',
    'I-MEDICALRECORD',
    'I-URL',
    'B-URL',
    'S-DEVICE',
    'B-STREET',
    'E-HOSPITAL',
    'I-DOCTOR',
    'I-CITY',
    'E-LOCATION-OTHER',
    'I-PROFESSION',
    'S-HOSPITAL',
    'E-CITY',
    'S-PROFESSION',
    'I-PHONE',
    'S-AGE',
    'B-DATE',
    'S-STREET',
    'B-IDNUM',
    'S-USERNAME',
    'S-BIOID',
    'B-PHONE',
    'E-DATE',
    'I-HOSPITAL',
    'S-IDNUM',
    'B-ORGANIZATION',
    'S-ZIP',
    'I-EMAIL',
    'E-STATE',
    'B-ZIP',
    'E-STREET',
    'I-ORGANIZATION',
    'E-IDNUM',
    'B-DEVICE',
    'E-ZIP',
    'S-DATE',
    'E-ORGANIZATION',
    'S-PHONE',
    'S-STATE',
    'B-PROFESSION',
    'I-COUNTRY',
    'B-COUNTRY',
    'E-EMAIL',
    'I-DEVICE',
    'B-FAX',
    PAD,
    CLS,
    SEP,
    WORDPIECE
]

DEID_LABELS = {
    PAD: 0,
    'E-AGE': 1,
    'S-ORGANIZATION': 2,
    'I-AGE': 3,
    'E-PATIENT': 4,
    'O': 5,
    'I-STATE': 6,
    'S-PATIENT': 7,
    'E-DEVICE': 8,
    'B-AGE': 9,
    'I-PATIENT': 10,
    'E-PHONE': 11,
    'E-HEALTHPLAN': 12,
    'I-STREET': 13,
    'I-ZIP': 14,
    'B-STATE': 15,
    'B-MEDICALRECORD': 16,
    'S-LOCATION-OTHER': 17,
    'E-COUNTRY': 18,
    'S-MEDICALRECORD': 19,
    'B-DOCTOR': 20,
    'E-PROFESSION': 21,
    'B-HOSPITAL': 22,
    'B-LOCATION-OTHER': 23,
    'E-DOCTOR': 24,
    'I-FAX': 25,
    'S-DOCTOR': 26,
    'B-PATIENT': 27,
    'E-MEDICALRECORD': 28,
    'I-DATE': 29,
    'S-CITY': 30,
    'B-EMAIL': 31,
    'I-LOCATION-OTHER': 32,
    'B-HEALTHPLAN': 33,
    'S-COUNTRY': 34,
    'B-CITY': 35,
    'I-IDNUM': 36,
    'E-URL': 37,
    'E-FAX': 38,
    'I-MEDICALRECORD': 39,
    'I-URL': 40,
    'B-URL': 41,
    'S-DEVICE': 42,
    'B-STREET': 43,
    'E-HOSPITAL': 44,
    'I-DOCTOR': 45,
    'I-CITY': 46,
    'E-LOCATION-OTHER': 47,
    'I-PROFESSION': 48,
    'S-HOSPITAL': 49,
    'E-CITY': 50,
    'S-PROFESSION': 51,
    'I-PHONE': 52,
    'S-AGE': 53,
    'B-DATE': 54,
    'S-STREET': 55,
    'B-IDNUM': 56,
    'S-USERNAME': 57,
    'S-BIOID': 58,
    'B-PHONE': 59,
    'E-DATE': 60,
    'I-HOSPITAL': 61,
    'S-IDNUM': 62,
    'B-ORGANIZATION': 63,
    'S-ZIP': 64,
    'I-EMAIL': 65,
    'E-STATE': 66,
    'B-ZIP': 67,
    'E-STREET': 68,
    'I-ORGANIZATION': 69,
    'E-IDNUM': 70,
    'B-DEVICE': 71,
    'E-ZIP': 72,
    'S-DATE': 73,
    'E-ORGANIZATION': 74,
    'S-PHONE': 75,
    'S-STATE': 76,
    'B-PROFESSION': 77,
    'I-COUNTRY': 78,
    'B-COUNTRY': 79,
    'E-EMAIL': 80,
    'I-DEVICE': 81,
    'B-FAX': 82,
    UNK: 83,
    CLS: 84,
    SEP: 85,
    WORDPIECE: 86
}

COHORT_TEXTUAL_LABEL_CONSTANTS = {
    'U': 0,
    'Q': 1,
    'N': 2,
    'Y': 3
}

COHORT_INTUITIVE_LABEL_CONSTANTS = {
    'Q': 0,
    'N': 1,
    'Y': 2
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

MAX_COHORT_NUM_SENTS = 150  # number of sentences in chart
