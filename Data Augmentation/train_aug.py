# -*- coding: utf-8 -*-
# import packages
import json
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
ft=open('train.json','r')
train=json.load(ft)
num=0
for tr in train:
    num=num+1
    print(num)
    img = load_img(os.path.join('../ai_challenger_scene_train_20170904/train',tr['image_id']))  # this is a PIL image, please replace to your own file path
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x,batch_size=1,save_to_dir='train_aug',save_prefix=tr['label_id']+'#',save_format='jpg'):
        i += 1
        if i > 5:
            break  # otherwise the generator would loop indefinitely