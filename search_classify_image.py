# -*- coding: utf-8 -*-
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from helper_functions import search_windows, draw_boxes, add_heat,apply_threshold, draw_labeled_bboxes,generate_windows
import pickle
from scipy.ndimage.measurements import label

try_all_test_images = False

with open("./classifier_pickle.p", mode='rb') as f:
        data = pickle.load(f)
        svc=data['svc']
        X_scaler=data['X_scaler']
        svc = data["svc"]
        X_scaler = data["X_scaler"]
        color_space = data["color_space"]
        hog_channel = data["hog_channel"]
        orient = data["orient"]
        pix_per_cell = data["pix_per_cell"]
        cell_per_block = data["cell_per_block"]
        spatial_size = data["spatial_size"]
        hist_bins = data["hist_bins"]
        spatial_feat = data["spatial_feat"]
        hist_feat = data["hist_feat"]
        hog_feat = data["hog_feat"]

if try_all_test_images:
    loop_num=6
else:
    loop_num=1


for i in range(loop_num):
    image = cv2.imread('./test_images/test'+str(i+1)+'.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_shape = image.shape

    windows_all=generate_windows(img_shape, overlap=0.75)
    hot_windows = search_windows(image,windows_all, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)



    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    heatmap =add_heat(heatmap, hot_windows)
    heatmap_thresholded = apply_threshold(heatmap,2)
    labels = label(heatmap_thresholded)
    # Draw bounding boxes on a copy of the image
    cars_found_img = draw_labeled_bboxes(np.copy(image), labels)
    mpimg.imsave('output_images/cars_found_test'+str(i+1)+'.jpg',cars_found_img)

    # Display the image
    if not try_all_test_images:
        f, ax = plt.subplots(3, 2, figsize=(10,10))
        ax[0][0].imshow(image)
        ax[0][0].set_title('original', fontsize=10)

        window_img = draw_boxes(image, windows_all, color=(0, 0, 255), thick=6)
        ax[0][1].imshow(window_img)
        ax[0][1].set_title('sliding windows', fontsize=10)
        mpimg.imsave('output_images/sliding_windows.jpg',window_img)

        hot_window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)
        ax[1][0].imshow(hot_window_img)
        ax[1][0].set_title('hot windows', fontsize=10)
        mpimg.imsave('output_images/hot_windows.jpg',hot_window_img)

        ax[1][1].imshow(heatmap_thresholded*255/heatmap_thresholded.max())
        ax[1][1].set_title('heatmap_thresholded', fontsize=10)
        mpimg.imsave('output_images/heatmap_thresholded.jpg',heatmap_thresholded*255/heatmap_thresholded.max())

        ax[2][0].imshow(cars_found_img)
        ax[2][0].set_title('cars found', fontsize=10)

        ax[2][1].imshow(labels[0])
        ax[2][1].set_title('labels', fontsize=10)
        mpimg.imsave('output_images/labels',labels[0])

