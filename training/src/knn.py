import os
import math
import random
import shutil
import urllib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exif import Image
from PIL import Image as PILImage

from sklearn import preprocessing, neighbors, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split



def knn(df):
    # Randomize to create distributed labels. (More accurate training).
    np.random.shuffle(df)
    np.random.seed(1)

    # Separate dataframes for building KNN prediction.
    dfSize = df.shape[0]
    coords = df[:, 2:4].astype(float)
    building_names = df[:, 1:2].ravel()

    # Perform stratified 10-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    # Variables that will be constantly updated to determine best K.
    bestscore = 0
    bestk = 1

    # Test K values 1-101.
    print()
    print("Utilizing 10-Fold Cross Validation...")
    for k in range(1, 101):
        fold_scores = []

        # Set up parsed training/testing sets for each fold.
        for train_index, test_index in kfold.split(coords, building_names):
            X_train, X_test = coords[train_index], coords[test_index]
            y_train, y_test = building_names[train_index], building_names[test_index]

            knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
            knn.fit(X_train, y_train)

            # Attain accuracy for each fold.
            y_pred = knn.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            fold_scores.append(acc)

        # Average accuracy from each fold.
        avg_score = np.mean(fold_scores)
        print(f".  k = {k}, Average accuracy = {avg_score * 100:.4f}%")

        # If new leader, update the variables.
        if avg_score > bestscore:
            bestscore = avg_score
            bestk = k

    # Print and apply the found best K value.
    print(f"     - Finished! Best k: {bestk} with accuracy {bestscore * 100:.4f}%")
    knn = neighbors.KNeighborsClassifier(n_neighbors=bestk, weights='distance', metric='euclidean')
    knn.fit(coords, building_names)

    # Extract coordinates from this image.
    try:
        img_lat, img_lon = extract_metadata(PREDICTED_IMAGE)
    except AttributeError:
        print(f"ERROR: {img_name} doesn't have coordinate metadata.")
        quit()

    # Transform lat/long into 2d array for KNN.
    img_coords = [img_lat, img_lon]
    img_coords = np.array(img_coords)

    # Use KNN to predict what building you are in.
    print(f"\nUtilizing KNN (k = {bestk}) to determine building name...")
    predictedLabel = knn.predict(img_coords.reshape(1, -1))
    predictedLabel = predictedLabel[0] # Originally returns as array, so convert back to string.
    predictedBuildingDirectory = PARENT_DIRECTORY +  "/" + predictedLabel

    # Determine every neighbor from KNN.
    distances, indices = knn.kneighbors(img_coords.reshape(1, -1))
    neighbors = {}

    # Iterate through every neighbor found.
    for i, idx in enumerate(indices[0]):
        # Translate numerical label back into viewable string.
        decodedLabel = oneHotDict[knn._y[idx]]

        # If an unseen neighbor is found, add to list.
        if decodedLabel not in neighbors:
            neighbors[decodedLabel] = 0

        # Update count for each beighbor appearance.
        neighbors[decodedLabel] += 1


def extract_metadata(path):
    # Open each image.
    with open(path, 'rb') as image_file:
        my_image = Image(image_file)

    # Convert coordinates from D/M/S notation into decimal.
    convertedLat = (my_image.gps_latitude[0] + (my_image.gps_latitude[1] / 60) + (my_image.gps_latitude[2] / 3600))
    if my_image.gps_latitude_ref == 'S':
        convertedLat = -convertedLat
    convertedLon = (my_image.gps_longitude[0] + (my_image.gps_longitude[1] / 60) + (my_image.gps_longitude[2] / 3600))
    if my_image.gps_longitude_ref == 'W':
        convertedLon = -convertedLon

    return convertedLat, convertedLon