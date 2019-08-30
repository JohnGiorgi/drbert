import xmltodict
import json
import os
from constants import *


def read_charts(data_path="data/diabetes_data"):

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
                train_charts[chart_id] = text # format

    test_charts = dict()
    test_file = os.path.join(data_path, "obesity_patient_records_test.xml")
    with open(test_file) as open_file:
        read_file = open_file.read()
        train_data1 = dict(xmltodict.parse(read_file))
        for i in range(len(train_data1['root']['docs']['doc'])):
            chart_id = train_data1['root']['docs']['doc'][i]['@id']
            text = train_data1['root']['docs']['doc'][i]['text']
            test_charts[chart_id] = text # format

    return train_charts, test_charts

def read_labels(data_path="data/diabetes_data"):
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
                    train_labels[chart_id][disease_name] = judgement # format

    addendum_file = os.path.join(data_path, "obesity_standoff_annotations_training_addendum2.xml")
    with open(addendum_file) as open_file:
        read_file = open_file.read()
        label_data = dict(xmltodict.parse(read_file))
        disease_label_list = label_data['diseaseset']['diseases']['disease']
        for disease_label in disease_label_list:
            disease_name = disease_label['@name']
            disease_docs = disease_label['doc']
            if type(disease_docs) == list:
                for doc in disease_docs:
                    chart_id = doc['@id']
                    chart_label = doc['@judgment']
                    if chart_id not in train_labels:
                        train_labels[chart_id] = dict.fromkeys(COHORT_DISEASE_LIST, 'N')
                    train_labels[chart_id][disease_name] = chart_label # format
            else:
                doc = disease_docs
                chart_id = doc['@id']
                chart_label = doc['@judgment']
                if chart_id not in train_labels:
                    train_labels[chart_id] = dict.fromkeys(COHORT_DISEASE_LIST, 'N')
                train_labels[chart_id][disease_name] = chart_label # format

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
                    train_labels[chart_id][disease_name] = judgement # format

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
                test_labels[chart_id][disease_name] = judgement # format
    return train_labels, test_labels



if __name__ == "__main__":
    train_charts, test_charts = read_charts("data/diabetes_data")
    train_labels, test_labels = read_labels("data/diabetes_data")
    for chart_id, _ in train_charts.items():
        if chart_id not in train_labels:
            print(f"{chart_id} not in ")
    for chart_id, _ in test_charts.items():
        if chart_id not in test_labels:
            print(f"{chart_id} not in ")

    for chart_id, chart_labels in train_labels.items():
        if len(chart_labels) != 16:
            print(chart_labels)

    for chart_id, chart_labels in test_labels.items():
        if len(chart_labels) != 16:
            print(chart_labels)
