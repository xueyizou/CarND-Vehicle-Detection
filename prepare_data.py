# -*- coding: utf-8 -*-
import cv2
import glob
import pickle
import os

# Read in cars and notcars
if not os.path.exists("./data_pickle.p"):
    car_images=[]
    notcar_images=[]

    car_files = glob.glob('./data/vehicles/*/*.png')
    for car_file in car_files:
        # Read in each one by one
        image = cv2.imread(car_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        car_images.append(image)

    notcar_files = glob.glob('./data/non-vehicles/*/*.png')
    for notcar_file in notcar_files:
        # Read in each one by one
        image = cv2.imread(notcar_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        notcar_images.append(image)
    img_shape = image.shape

    dist_pickle = {}
    dist_pickle["car_images"] = car_images
    dist_pickle["notcar_images"] = notcar_images
    dist_pickle["img_shape"] = img_shape
    pickle.dump( dist_pickle, open( "./data_pickle.p", "wb" ))


