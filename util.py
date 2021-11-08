import os
from parameters import *
import numpy as np
import cv2 as cv
from ML import *
from util import *
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage.measurements import label
from sklearn.metrics import mean_absolute_error
import torch


def save_load(func, savename):
    """
    tries to load the given function from npy files, if it is not there
    run the function
    """
    try:
        with open(savename, 'rb') as f:
            result = np.load(f)
    except:
        result = func()
        with open(savename, 'wb') as f:
            np.save(f, result)

    return result


def extract_parameters(filename: str) -> list:
    """
    extracts all the parameters from a filename
    return: steady_state, [mu, ka, k, D]
    """

    temp = filename.split('-mu')
    steady_state = float(temp[0])

    p = temp[1].split('.png')[0].split('-')
    parameters = np.array([p[1], p[3], p[5], p[7]], dtype=np.float64)

    return steady_state, parameters


def load_images(colorspace='RGB'):
    """
    reads all images in ./images and returns a list of 3-dimensional 
    arrays and the corresponding constants 
    """

    images, parameters, steady_states = [], [], []

    # load all turing pattern images
    for filename in os.listdir(params['img_path']):
        if filename.endswith('.png'):

            steady_state, p = extract_parameters(filename)
            if steady_state < 0.001:
                steady_states.append(steady_state)
                # parameters.append(torch.tensor(p).float())
                parameters.append(p)
                            
                # read the image array as RGB
                img = cv.imread(params['img_path'] + filename, cv.IMREAD_GRAYSCALE)/255
                img = torch.tensor(img.reshape((1, *img.shape))).float()
                
                if colorspace == 'HSV':
                    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                if colorspace == 'Grayscale':
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # append the image array to the images list
                images.append(img)

    return images, parameters, steady_states


def extract_features(images):
    """
    function that extracts all the features for each available image
    """

    features = []

    for img in images:

        # select the hue of the image: HSV
        img = img[:, :, 0]

        feature = []

        # create a 1 dimensional array with all the hues
        colors = img.flatten()

        # cluster the hues in n clusters
        kmeans = KMeans(n_clusters=params['clusters'], random_state=0).fit(colors.reshape(-1,1))
        colors = kmeans.labels_.reshape((200,200))

        # find patches in the image with the cluster hue and label each patch
        labels = measure.label(colors, connectivity=2, background=-1)
        
        # determine the properties of each label
        props = measure.regionprops(labels)

        # the area of each label
        areas = list(map(lambda x: x.area, props))

        # determine the contour of each labeled patch
        contour = np.array(list(map(lambda x: x.perimeter, props)))

        # remove the largest area
        argmax = np.argmax(areas, axis=0)
        areas = np.delete(areas, argmax)
        argmax = np.argmax(contour, axis=0)
        contour = np.delete(contour, argmax)

        conAreaRatio = np.array(contour) / np.array(areas)

        area_hist = np.histogram(areas, bins=params['area_bins'])[0]
        conAreaRatio_hist = np.histogram(conAreaRatio, bins=params['ratio_bins'])[0]

        # normalize
        area_hist = area_hist / sum(area_hist)
        # contour_hist = contour_hist / sum(contour_hist)

        feature.extend(area_hist)
        feature.extend(conAreaRatio_hist)

        feature.append(np.mean(conAreaRatio))
        feature.append(np.std(conAreaRatio))
        feature.append(np.mean(area_hist))
        feature.append(np.std(area_hist))

        features.append(feature)

    return features


def resize_images(images, dimension):
    """
    resizes images to the given dimension
    """
    return [cv.resize(img, (28, 28), interpolation = cv.INTER_AREA) for img in images]