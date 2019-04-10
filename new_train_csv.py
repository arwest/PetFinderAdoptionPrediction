import csv
import json
import os
from os.path import isfile, join

desc_index = 20
pet_id_index = 21

def create_new_train_csv():
    sent = dict()

    json_folder_dir = join(os.curdir, 'petfinder-adoption-prediction', 'train_sentiment')

    for file in os.listdir(json_folder_dir):
        name = file.split('.')[0]
        if isfile(join(json_folder_dir, file)):
            with open(join(json_folder_dir, file)) as json_file:
                data = json.load(json_file)
                doc_sent = data['documentSentiment']
                sent[name] = (doc_sent['magnitude'], doc_sent['score'])

    data = []
    with open(join(os.curdir, 'petfinder-adoption-prediction', 'train.csv')) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        line = 0
        for row in csvReader:
            if line == 0:
                headers = row[:19] + ['DescMagnitude', 'DescScore'] + row[21:]
                line += 1
                continue
            try:
                (magnitude, score) = sent[row[pet_id_index]]
            except KeyError:
                magnitude = score = 'NaN'
            new_row_data = row[:19] + [magnitude, score] + row[21:]
            data.append(new_row_data)

    with open(join(os.curdir, 'petfinder-adoption-prediction', 'train_wo_desc.csv'), 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',')
        csvWriter.writerow(headers)
        for data_line in data:
            csvWriter.writerow(data_line)

    return

create_new_train_csv()