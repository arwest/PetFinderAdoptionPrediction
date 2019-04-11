import csv
import os
import numpy as np
from os.path import isfile, join

type_index = 0
adoption_speed_index = -1
pet_id_index = -3

def create_type_speed_array():
    folder_dir = join(os.curdir,'petfinder-adoption-prediction', 'train_images_npy')

    ids = []
    for file in os.listdir(folder_dir):
        name = file.split('-')[0]
        if isfile(join(folder_dir, file)) and not ids.__contains__(name):
            ids.append(name)
    ids.sort()
    dic = dict()
    with open(join(os.curdir,'petfinder-adoption-prediction','train.csv')) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        line = 0
        for row in csvReader:
            if line == 0:
                line += 1
                continue
            dic[row[pet_id_index]] = (row[type_index], row[adoption_speed_index])

    results_dir = join(os.curdir, 'petfinder-adoption-prediction','train_images_results')
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    data= []
    with open(join(results_dir,'ids.txt'), 'w') as id_file:
        for id in ids:
            (animal_type, adoption_speed) = dic[id]
            data.append([animal_type, adoption_speed])
            id_file.write(str(id) + '\n')
        data_matrix = np.array(data, dtype=np.int32)

    np.save(join(os.curdir, 'petfinder-adoption-prediction','train_images_results','matrix'), data_matrix)

    return data_matrix

create_type_speed_array()