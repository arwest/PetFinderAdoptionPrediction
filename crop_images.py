import json
import os
from os.path import join
from PIL import Image

train_img_path = join(os.curdir, 'petfinder-adoption-prediction','train_images')
train_meta_path = join(os.curdir, 'petfinder-adoption-prediction','train_metadata')

results_dir = join(os.curdir, 'petfinder-adoption-prediction', 'train_crop_images')
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

index = 0
train_meta_list = os.listdir(train_meta_path)
train_meta_list.sort()
for file in train_meta_list:
    file_name = file.split('.')[0]
    vertex_pairs = []
    with open(join(train_meta_path, file_name+'.json')) as json_file:
        data = json.load(json_file)
        crop_vertices = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices']
        for vertex in crop_vertices:
            try:
                x = int(vertex['x'])
            except KeyError:
                x = 0
            try:
                y = int(vertex['y'])
            except KeyError:
                y = 0
            vertex_pairs.append((x,y))
    with Image.open(join(train_img_path, file_name+'.jpg')) as img:
        crop_img = img.crop([vertex_pairs[0][0], vertex_pairs[0][1],
                             vertex_pairs[2][0] - 1, vertex_pairs[2][1] - 1])
        crop_img.save(join(results_dir, file), 'JPEG')

    index += 1
    if (index+1) % 5000 == 0:
        print(index+1, 'images cropped')
print('All images cropped')
#
# index = 0
# train_img_list = os.listdir(train_img_path)
# train_img_list.sort()
# for file in train_img_list:
#     name = file.split('.')[0]
#     if isfile(join(train_img_path, file)):
#         img = Image.open(join(train_img_path, file))
#         vertex_list = vertices[name]
#         crop_img = img.crop([vertex_list[0][0], vertex_list[0][1],
#                              vertex_list[2][0] - 1, vertex_list[2][1] - 1])
#         crop_img.save(join(results_dir, file), 'JPEG')
#         index += 1
#         if (index+1) % 5000 == 0:
#             print('Images saved for', index + 1, 'images')
# print('Cropped images saved for all images')
