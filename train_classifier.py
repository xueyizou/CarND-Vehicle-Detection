# -*- coding: utf-8 -*-
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from helper_functions import extract_features
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from helper_functions import get_hog_features

#%% code cell 1: define train function
with open("./data_pickle.p", mode='rb') as f:
    data = pickle.load(f)
    car_images=data['car_images']
    notcar_images=data['notcar_images']
    img_shape =data['img_shape']
    n_cars= len(car_images)
    n_notcars= len(notcar_images)


def train(color_space,hog_channel, orient,pix_per_cell,cell_per_block = 2,
          spatial_size = (16, 16), hist_bins = 16,
          spatial_feat = True, hist_feat = True, hog_feat = True):

    t=time.time()
    car_features = extract_features(car_images, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcar_images, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',color_space, 'color space')
    print('hog_channel: ',hog_channel)
    print('with:', orient,'orientations',pix_per_cell,'*',pix_per_cell,
        'pixels per cell and', cell_per_block,'*',cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    test_accur=round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', test_accur)

    return (X_scaler, svc, test_accur)
#%% code cell 2: Tune parameters
##### run this cell only if you want to tune parameters, otherwise, skip this section
parameters=[]
test_accurs=[]

color_spaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
hog_channels = [0,1,2,'ALL'] # Can be 0, 1, 2, or "ALL"
orients = [6,7,8,9,10,11,12]  # HOG orientations
pix_per_cell_values = [8,16] # HOG pixels per cell

cell_per_block = 2 # HOG cells per block
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

t=time.time()
for color_space in color_spaces:
    for hog_channel in hog_channels:
        for orient in orients:
            for pix_per_cell in pix_per_cell_values:
                parameters.append((color_space,hog_channel, orient, pix_per_cell))
                _, _, test_accur=train(color_space,hog_channel, orient,pix_per_cell,cell_per_block = cell_per_block,
                                  spatial_size = spatial_size, hist_bins = hist_bins,
                                  spatial_feat = spatial_feat, hist_feat = hist_feat, hog_feat = hog_feat)
                test_accurs.append(test_accur)
                print('-------------------'*5)

t2=time.time()
print(round(t2-t, 2), 'Seconds to tune parameters...')
#15931.63 Seconds

dist_pickle = {}
dist_pickle["parameters"] = parameters
dist_pickle["test_accurs"] = test_accurs
pickle.dump( dist_pickle, open( "./parameter_tuning_pickle.p", "wb" ) )



#%% code cell 3: train a classifier

with open("./parameter_tuning_pickle.p", mode='rb') as f:
    data = pickle.load(f)
    parameters=data['parameters']
    test_accurs=data['test_accurs']

combo_list = [(test_accurs[i], i) for i in range(len(test_accurs)) ]
sorted_combo_list = sorted(combo_list)
sorted_combo_list.reverse()

print('Here are some good choices for (color_space,hog_channel,orient,pix_per_cell):')
for i in range(20):
    idx = sorted_combo_list[i][1]
    parameter = parameters[idx]
    print('accuracy: ', sorted_combo_list[i][0], '  parameter: ',parameter)


# The following are some good choices for (color_space,hog_channel,orient,pix_per_cell):
#accuracy:  0.9955   parameter:  ('LUV', 'ALL', 12, 8)
#accuracy:  0.9952   parameter:  ('YCrCb', 'ALL', 12, 16)
#accuracy:  0.9952   parameter:  ('YCrCb', 'ALL', 6, 8)
#accuracy:  0.9952   parameter:  ('LUV', 'ALL', 10, 8)
#accuracy:  0.9949   parameter:  ('YUV', 'ALL', 10, 8)
#accuracy:  0.9949   parameter:  ('LUV', 'ALL', 8, 8)
#accuracy:  0.9949   parameter:  ('HSV', 'ALL', 10, 8)
#accuracy:  0.9947   parameter:  ('HSV', 'ALL', 9, 16)
#accuracy:  0.9944   parameter:  ('YCrCb', 'ALL', 10, 8)
#accuracy:  0.9944   parameter:  ('LUV', 'ALL', 12, 16)
#accuracy:  0.9941   parameter:  ('YCrCb', 'ALL', 10, 16)
#accuracy:  0.9941   parameter:  ('HLS', 'ALL', 12, 8)
#accuracy:  0.9941   parameter:  ('LUV', 'ALL', 7, 8)
#accuracy:  0.9938   parameter:  ('YUV', 'ALL', 12, 8)
#accuracy:  0.9935   parameter:  ('HLS', 'ALL', 8, 8)
#accuracy:  0.9935   parameter:  ('HSV', 'ALL', 10, 16)
#accuracy:  0.9932   parameter:  ('YCrCb', 'ALL', 11, 16)
#accuracy:  0.9932   parameter:  ('HLS', 'ALL', 12, 16)
#accuracy:  0.9932   parameter:  ('HLS', 'ALL', 10, 8)
#accuracy:  0.9932   parameter:  ('HSV', 'ALL', 9, 8)
color_space ='YCrCb' # ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
orient = 6 # HOG orientations [6,7,8,9,10,11,12]
pix_per_cell= 8 # HOG pixels per cell [8,16]
cell_per_block = 2 # HOG cells per block
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# show hog images
f, ax = plt.subplots(4, 4, figsize=(10,40))
for i in range(2):
    car_img = car_images[np.random.randint(0, n_cars)]
    ax[0][i].imshow(car_img)
    ax[0][i].set_title('original car '+str(i), fontsize=10)

    for j in range(3):
        _, hog_image = get_hog_features(car_img[:,:,i], orient, pix_per_cell, cell_per_block,
                                vis=True, feature_vec=True)
        ax[j+1][i].imshow(hog_image, cmap='gray')
        ax[j+1][i].set_title('car '+str(i)+'  hog image channel '+str(j), fontsize=10)

for i in range(2):
    notcar_img = notcar_images[np.random.randint(0, n_cars)]
    ax[0][i+2].imshow(notcar_img)
    ax[0][i+2].set_title('original non-car '+str(i), fontsize=10)

    for j in range(3):
        _, hog_image = get_hog_features(notcar_img[:,:,i], orient, pix_per_cell, cell_per_block,
                                vis=True, feature_vec=True)
        ax[j+1][i+2].imshow(hog_image,cmap='gray')
        ax[j+1][i+2].set_title('non-car '+str(i)+'  hog image channel '+str(j), fontsize=10)

# train a classifier
X_scaler, svc, test_accur=train(color_space,hog_channel, orient,pix_per_cell,cell_per_block = cell_per_block,
                                  spatial_size = spatial_size, hist_bins = hist_bins,
                                  spatial_feat = spatial_feat, hist_feat = hist_feat, hog_feat = hog_feat)

# save the classifier
dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["X_scaler"] = X_scaler
dist_pickle["color_space"] = color_space
dist_pickle["hog_channel"] = hog_channel
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
dist_pickle["spatial_feat"] = spatial_feat
dist_pickle["hist_feat"] = hist_feat
dist_pickle["hog_feat"] = hog_feat
pickle.dump( dist_pickle, open( "./classifier_pickle.p", "wb" ) )