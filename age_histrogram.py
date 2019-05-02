import csv
import json
import os
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt


adoption_speed_index = -1
age_index = 2

adoption_speed_category = 1

data = []

with open(join(os.curdir, 'petfinder-adoption-prediction', 'train_wo_desc.csv')) as csvFile:
    csvReader = csv.reader(csvFile, delimiter=',')
    line = 0
    for row in csvReader:
        if line == 0:
            line += 1
            continue
        age = int(row[age_index])
        adopt_speed = int(row[adoption_speed_index])
        if adopt_speed == adoption_speed_category:
            data.append(age)

data = np.array(data)

plt.hist(data)
plt.show()


