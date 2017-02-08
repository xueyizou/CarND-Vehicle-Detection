# -*- coding: utf-8 -*-
import numpy as np
from helper_functions import search_windows, add_heat,apply_threshold, draw_labeled_bboxes,generate_windows
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

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

def car_detection_pipeline():
    windows=[]
    pre_heatmaps=[]
    def process(image):
        # NOTE: The output you return should be a color image (3 channel) for processing video below
        nonlocal windows
        if not windows:
            windows=generate_windows(image.shape, overlap=0.75)

        hot_windows = search_windows(image,windows, svc, X_scaler, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)

        cur_heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        cur_heatmap =add_heat(cur_heatmap, hot_windows)

        sum_heatmap=np.copy(cur_heatmap)
#        weights=[1,1,1,1,1]
        weights=[0.4,0.5,0.6,0.8,0.9]
        for idx,pre_heatmap in enumerate(pre_heatmaps):
            sum_heatmap += weights[idx]*pre_heatmap

        ave_heatmap= sum_heatmap/(1+sum(weights[0:len(pre_heatmaps)]))
        heatmap_thresholded = apply_threshold(ave_heatmap,1.5 )

        pre_heatmaps.append(cur_heatmap)
        if len(pre_heatmaps)>5:
            del pre_heatmaps[0]
        assert len(pre_heatmaps)<=5


        labels = label(heatmap_thresholded)
        # Draw bounding boxes on a copy of the image
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        return draw_img

    return process

#video_output = 'output_images/test_video_output.mp4' # test_video_output.mp4  project_video_output.mp4
#clip1 = VideoFileClip("test_video.mp4") # test_video.mp4  project_video.mp4

video_output = 'output_images/project_video_output.mp4' # test_video_output.mp4  project_video_output.mp4
clip1 = VideoFileClip("project_video.mp4") # test_video.mp4  project_video.mp4

pipeline=car_detection_pipeline()
video_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
video_clip.write_videofile(video_output, audio=False)


