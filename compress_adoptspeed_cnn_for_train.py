import numpy as np
import os
from os.path import isfile, join
import csv

pet_id_index = -3

adopt_cnn_array = np.load(join(os.curdir, 'petfinder-adoption-prediction', 'adoption_speed_matrix.npy'))
folder_dir = join(os.curdir, 'petfinder-adoption-prediction', 'train_images_npy')

ids = {}
index = 0
for file in os.listdir(folder_dir):
    name = file.split('-')[0]
    # we want to write duplicates as well
    if isfile(join(folder_dir, file)) and not ids.__contains__(name):
        ids[name] = []
        ids[name].append(adopt_cnn_array[index,:])
    else:
        ids[name].append(adopt_cnn_array[index, :])
    index += 1

for id in ids.keys():
    avg_preds = [0,0,0,0,0]
    for array in ids[id]:
        for i in range(5):
            avg_preds[i] += array[i]
    ids[id] = list(map(lambda x: x/len(ids[id]), avg_preds))

final_array = []

with open(join(os.curdir, 'petfinder-adoption-prediction', 'train.csv')) as csvFile:
    csvReader = csv.reader(csvFile, delimiter=',')
    line = 0
    for row in csvReader:
        if line == 0:
            line += 1
            continue
        try:
            final_array.append(ids[row[-3]])
            line += 1

        except KeyError:
            final_array.append([0, 0, 0, 0, 0])
            line += 1

result = np.array(final_array)
np.save(join(os.curdir, 'petfinder-adoption-prediction', 'compressed_adoption_speed_matrix.npy'), result)
