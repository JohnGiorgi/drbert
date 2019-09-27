import os

import xmltodict

from .constants import COHORT_DISEASE_LIST


def read_charts_textual(data_path="data/cohort"):

    train_charts = dict()
    train_file_list = [
        os.path.join(data_path, "obesity_patient_records_training.xml"),
        os.path.join(data_path, "obesity_patient_records_training2.xml")
    ]
    for train_file in train_file_list:
        with open(train_file) as open_file:
            read_file = open_file.read()
            train_data1 = dict(xmltodict.parse(read_file))
            for i in range(len(train_data1['root']['docs']['doc'])):
                chart_id = train_data1['root']['docs']['doc'][i]['@id']
                text = train_data1['root']['docs']['doc'][i]['text']
                train_charts[chart_id] = text  # format

    test_charts = dict()
    test_file = os.path.join(data_path, "obesity_patient_records_test.xml")
    with open(test_file) as open_file:
        read_file = open_file.read()
        train_data1 = dict(xmltodict.parse(read_file))
        for i in range(len(train_data1['root']['docs']['doc'])):
            chart_id = train_data1['root']['docs']['doc'][i]['@id']
            text = train_data1['root']['docs']['doc'][i]['text']
            test_charts[chart_id] = text  # format

    return train_charts, test_charts


def read_labels_textual(data_path="data/cohort"):
    train_labels = dict()
    addendum_file = os.path.join(data_path, "obesity_standoff_annotations_training.xml")
    with open(addendum_file) as open_file:
        read_file = open_file.read()
        label_data = dict(xmltodict.parse(read_file))
        for i in range(len(label_data['diseaseset']['diseases'])):
            if label_data['diseaseset']['diseases'][i]['@source'] != "textual":
                continue
            for j in range(len(label_data['diseaseset']['diseases'][i]['disease'])):
                disease_name = label_data['diseaseset']['diseases'][i]['disease'][j]['@name']
                for k in range(len(label_data['diseaseset']['diseases'][i]['disease'][j]['doc'])):
                    chart_id = label_data['diseaseset']['diseases'][i]['disease'][j]['doc'][k]['@id']
                    judgement = label_data['diseaseset']['diseases'][i]['disease'][j]['doc'][k]['@judgment']
                    if chart_id not in train_labels:
                        train_labels[chart_id] = dict.fromkeys(COHORT_DISEASE_LIST, 'N')
                    train_labels[chart_id][disease_name] = judgement  # format

    addendum_file = os.path.join(data_path, "obesity_standoff_annotations_training_addendum2.xml")
    with open(addendum_file) as open_file:
        read_file = open_file.read()
        label_data = dict(xmltodict.parse(read_file))
        disease_label_list = label_data['diseaseset']['diseases']['disease']
        for disease_label in disease_label_list:
            disease_name = disease_label['@name']
            disease_docs = disease_label['doc']
            if isinstance(disease_docs, list):
                for doc in disease_docs:
                    chart_id = doc['@id']
                    chart_label = doc['@judgment']
                    if chart_id not in train_labels:
                        train_labels[chart_id] = dict.fromkeys(COHORT_DISEASE_LIST, 'N')
                    train_labels[chart_id][disease_name] = chart_label  # format
            else:
                doc = disease_docs
                chart_id = doc['@id']
                chart_label = doc['@judgment']
                if chart_id not in train_labels:
                    train_labels[chart_id] = dict.fromkeys(COHORT_DISEASE_LIST, 'N')
                train_labels[chart_id][disease_name] = chart_label  # format

    addendum_file = os.path.join(data_path, "obesity_standoff_annotations_training_addendum3.xml")
    with open(addendum_file) as open_file:
        read_file = open_file.read()
        label_data = dict(xmltodict.parse(read_file))
        for i in range(len(label_data['diseaseset']['diseases'])):
            if label_data['diseaseset']['diseases'][i]['@source'] != "textual":
                continue

            for j in range(len(label_data['diseaseset']['diseases'][i]['disease'])):
                disease_name = label_data['diseaseset']['diseases'][i]['disease'][j]['@name']
                docs = label_data['diseaseset']['diseases'][i]['disease'][j]['doc']
                for doc in docs:
                    chart_id = doc['@id']
                    judgement = doc['@judgment']
                    if chart_id not in train_labels:
                        train_labels[chart_id] = dict.fromkeys(COHORT_DISEASE_LIST, 'N')
                    train_labels[chart_id][disease_name] = judgement  # format

    test_labels = dict()
    test_file = os.path.join(data_path, "obesity_standoff_annotations_test_textual.xml")
    with open(test_file) as open_file:
        read_file = open_file.read()
        label_data = dict(xmltodict.parse(read_file))
        for i in range(len(label_data['diseaseset']['diseases']['disease'])):
            disease_name = label_data['diseaseset']['diseases']['disease'][i]['@name']
            for j in range(len(label_data['diseaseset']['diseases']['disease'][i]['doc'])):
                chart_id = label_data['diseaseset']['diseases']['disease'][i]['doc'][j]['@id']
                judgement = label_data['diseaseset']['diseases']['disease'][i]['doc'][j]['@judgment']
                if chart_id not in test_labels:
                    test_labels[chart_id] = dict.fromkeys(COHORT_DISEASE_LIST, 'N')
                test_labels[chart_id][disease_name] = judgement  # format
    return train_labels, test_labels


def read_charts_intuitive(data_path="diabetes_data"):
    train_charts = dict()
    train_file_list = [
        os.path.join(data_path, "obesity_patient_records_training.xml"),
        os.path.join(data_path, "obesity_patient_records_training2.xml")
    ]
    for train_file in train_file_list:
        with open(train_file) as open_file:
            read_file = open_file.read()
            train_data1 = dict(xmltodict.parse(read_file))
            for i in range(len(train_data1['root']['docs']['doc'])):
                chart_id = train_data1['root']['docs']['doc'][i]['@id']
                text = train_data1['root']['docs']['doc'][i]['text']
                train_charts[chart_id] = text  # format

    test_charts = dict()
    test_file = os.path.join(data_path, "obesity_patient_records_test.xml")
    with open(test_file) as open_file:
        read_file = open_file.read()
        train_data1 = dict(xmltodict.parse(read_file))
        for i in range(len(train_data1['root']['docs']['doc'])):
            chart_id = train_data1['root']['docs']['doc'][i]['@id']
            text = train_data1['root']['docs']['doc'][i]['text']
            test_charts[chart_id] = text  # format

    return train_charts, test_charts


def read_labels_intuitive(data_path="diabetes_data"):
    train_labels = dict()
    addendum_file = os.path.join(data_path, "obesity_standoff_annotations_training.xml")
    with open(addendum_file) as open_file:
        read_file = open_file.read()
        label_data = dict(xmltodict.parse(read_file))
        for i in range(len(label_data['diseaseset']['diseases'])):
            if label_data['diseaseset']['diseases'][i]['@source'] != "intuitive":
                continue
            # print(f"{addendum_file}: {label_data['diseaseset']['diseases'][i]['@source']}")
            for j in range(len(label_data['diseaseset']['diseases'][i]['disease'])):
                disease_name = label_data['diseaseset']['diseases'][i]['disease'][j]['@name']
                for k in range(len(label_data['diseaseset']['diseases'][i]['disease'][j]['doc'])):
                    chart_id = label_data['diseaseset']['diseases'][i]['disease'][j]['doc'][k]['@id']
                    judgement = label_data['diseaseset']['diseases'][i]['disease'][j]['doc'][k]['@judgment']
                    if chart_id not in train_labels:
                        train_labels[chart_id] = dict()

                    # print(f"{chart_id}, {disease_name}: {judgement}")
                    train_labels[chart_id][disease_name] = judgement  # format

    addendum_file = os.path.join(data_path, "obesity_standoff_annotations_training_addendum.xml")
    with open(addendum_file) as open_file:
        read_file = open_file.read()
        label_data = dict(xmltodict.parse(read_file))
        # print(f"{addendum_file}: {label_data['diseaseset']['diseases']['@source']}")
        disease_name = label_data['diseaseset']['diseases']['disease']['@name']
        for i in range(len(label_data['diseaseset']['diseases']['disease']['doc'])):
            chart_id = label_data['diseaseset']['diseases']['disease']['doc'][i]['@id']
            judgement = label_data['diseaseset']['diseases']['disease']['doc'][i]['@judgment']
            if chart_id not in train_labels:
                train_labels[chart_id] = dict()
            if disease_name in train_labels[chart_id]:
                print("uh oh.")
            train_labels[chart_id][disease_name] = judgement  # format

    addendum_list = [
        # os.path.join(data_path, "obesity_standoff_annotations_training_addendum2.xml"),
        os.path.join(data_path, "obesity_standoff_annotations_training_addendum3.xml")
    ]

    for addendum_file in addendum_list:
        with open(addendum_file) as open_file:
            read_file = open_file.read()
            label_data = dict(xmltodict.parse(read_file))
            for i in range(len(label_data['diseaseset']['diseases'])):
                if label_data['diseaseset']['diseases'][i]['@source'] != "intuitive":
                    continue
                # print(f"{addendum_file}: {label_data['diseaseset']['diseases'][i]['@source']}")

                for j in range(len(label_data['diseaseset']['diseases'][i]['disease'])):
                    disease_name = label_data['diseaseset']['diseases'][i]['disease'][j]['@name']
                    docs = label_data['diseaseset']['diseases'][i]['disease'][j]['doc']
                    for doc in docs:
                        chart_id = doc['@id']
                        judgement = doc['@judgment']
                        if chart_id not in train_labels:
                            train_labels[chart_id] = dict()
                        if disease_name in train_labels[chart_id]:
                            print("uh oh.")
                        train_labels[chart_id][disease_name] = judgement  # format

    test_labels = dict()
    test_file = os.path.join(data_path, "obesity_standoff_annotations_test_intuitive.xml")
    with open(test_file) as open_file:
        read_file = open_file.read()
        label_data = dict(xmltodict.parse(read_file))
        # print(f"{test_file}: {label_data['diseaseset']['diseases']['@source']}")
        for i in range(len(label_data['diseaseset']['diseases']['disease'])):
            disease_name = label_data['diseaseset']['diseases']['disease'][i]['@name']
            for j in range(len(label_data['diseaseset']['diseases']['disease'][i]['doc'])):
                chart_id = label_data['diseaseset']['diseases']['disease'][i]['doc'][j]['@id']
                judgement = label_data['diseaseset']['diseases']['disease'][i]['doc'][j]['@judgment']
                if chart_id not in test_labels:
                    test_labels[chart_id] = dict()

                # print(f"{chart_id}, {disease_name}: {judgement}")
                test_labels[chart_id][disease_name] = judgement  # format
    return train_labels, test_labels
