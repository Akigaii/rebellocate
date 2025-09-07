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


# CHANGE THESE WHEN RUNNING PROGRAM
PARENT_DIRECTORY = ""
PREDICTED_IMAGE = ""

# Parse components from predicted image directory.
parts = PREDICTED_IMAGE.split("/")
img_name = parts[len(parts) - 1]
img_building = parts[len(parts) - 2]
img_label = parts[len(parts) - 3]

# Temporary 1D arrays to store info during each loop.
imgname = []
buildingname = []
latitudes = []
longitudes = []

# Loop through every building.
print(f"\nParsing through dataset from {PARENT_DIRECTORY}:")
for building in os.listdir(PARENT_DIRECTORY):
    if building == '.DS_Store':
        continue # Ignore hidden file for Mac systems.
    print(f".  Processing {building}...")
    building_path = os.path.join(PARENT_DIRECTORY, building)

    # Loop through every room label.
    for label in os.listdir(building_path):
        if label == '.DS_Store':
            continue # Ignore hidden file for Mac systems.
        label_path = os.path.join(building_path, label)

        tempimgname = []
        tempbuildingname = []
        templatitudes = []
        templongitudes = []
        imgCount = 0

        # Loop through every image.
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)
            imgCount += 1

            if image == '.DS_Store':
                continue

            try:
                # Extract coordinates from each image.
                convertedLat, convertedLon = extract_metadata(image_path)

                # Append information to temp 1D arrays.
                imgname.append(image)
                buildingname.append(building)
                latitudes.append(convertedLat)
                longitudes.append(convertedLon)

                tempimgname.append(image)
                tempbuildingname.append(building)
                templatitudes.append(convertedLat)
                templongitudes.append(convertedLon)
            except:
                print(f"Error: {image} doesn't have coordinate metadata.")

# Make 2D matrix holding all data.
df = np.column_stack((imgname, buildingname, latitudes, longitudes))
print(f"     - Finished! Total image count: {df.shape[0]}")

# Encode all building names (features) and store in accessible dictionary.
oneHot = LabelEncoder()
labelIDs = oneHot.fit_transform(df[:, 1:2].ravel())
oneHotDict = {}
for x, label in enumerate(oneHot.classes_):
    oneHotDict[x] = str(label)

# Scatter plot all points.
print("\nCreating scatter plot for all images parsed...")
plt.figure(figsize=(8, 6))
plt.scatter(longitudes, latitudes, marker='o', c=labelIDs, alpha=0.6)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Scatter Plot of Image GPS Coordinates")
plt.grid(True)
plt.show()
print(f"     - Finished! Check terminal for scatter plot.")


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

# Scan the existing building directory to see its possible room labels.
possibleAreas = []
for labels in os.listdir(PARENT_DIRECTORY + "/" + predictedLabel):
    if labels == '.DS_Store':
        continue
    possibleAreas.append(labels)
print("     - Finished!")

# Image transformations that will be applied to all.
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Required size for Places365.
    transforms.ToTensor(), # Convert every image into a tensor.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Known from Places365 pretraining.
])

# Use 80/20 training/testing split.
dataset = datasets.ImageFolder(root = PARENT_DIRECTORY + "/" + predictedLabel, transform = transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)


# Download pretrained weights if not already there.
model_file = 'resnet18_places365.pth.tar'
model_url = 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
if not os.access(model_file, os.W_OK):
    urllib.request.urlretrieve(model_url, model_file)

# Load ResNet18 and update with Places365 weights.
model = models.resnet18(num_classes=365)
checkpoint = torch.load(model_file, map_location=torch.device('cpu'))

# Remove 'module.' prefix from multi-GPU training.
orig_state_dict = checkpoint['state_dict']
state_dict = {}
for key, value in orig_state_dict.items():
    new_key = key.replace('module.', '')
    state_dict[new_key] = value

# Import the weights from Places365.
model.load_state_dict(state_dict)
model.eval()

# Freeze all layers.
for parameter in model.parameters():
    parameter.requires_grad = False

# Adjust last layer in pre-trained model.
model.fc = nn.Linear(model.fc.in_features, len(possibleAreas))

# Define loss and optimizer.
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Train the last layer of the model, hyperparameters being adjusted.
print("\nUtilizing CNN to determine room label(s)...")
model.train()
for epoch in range(10):
    print('Epoch', epoch + 1)

    # Go through every image in directory.
    for image_tensors, labels in train_loader:

        # Change these to 'cuda' when using the RebelX cluster.
        image_tensors, labels = image_tensors.to('cpu'), labels.to('cpu')

        # Adjust hyperparameters.
        optimizer.zero_grad()
        outputs = model(image_tensors)
        loss = loss_func(outputs, labels)
        print(f".  Loss: {loss}")

        # Backpropogate.
        loss.backward()
        optimizer.step()

classes = dataset.classes
all_preds = []
all_labels = []

with torch.no_grad():
    for image_tensors, labels in test_loader:
        # Change these to 'cuda' when using the RebelX cluster.
        image_tensors, labels = image_tensors.to('cpu'), labels.to('cpu')

        # Predict each image.
        outputs = model(image_tensors)
        logit_scores = model(image_tensors)
        scores = functional.softmax(logit_scores, dim = 1)
        predicted_class = torch.argmax(scores, dim = 1) # Chooses the highest probable room. class.

        # Append all predictinos.
        all_preds.extend(predicted_class.tolist())
        all_labels.extend(labels.tolist())

# Evaluate accuracies.
correct = 0
for index in range(len(all_labels)):
    if all_preds[index] == all_labels[index]:
        correct += 1
accuracy = correct / len(all_labels)
print(f"     - Finished! CNN on Testing Data Accuracy: {accuracy * 100:.2f}%")

# Load and transform image
img = PILImage.open(PREDICTED_IMAGE).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to('cpu')

# Use CNN to predict what type of room you are in.
model.eval()
with torch.no_grad():
    # Use a softmax to gauge predictions with multiple labels.
    logit = model(img_tensor)
    scores = functional.softmax(logit, dim = 1).data.squeeze()
    probs, index = scores.sort(0, True)

# Print final results.
print(f"\nFinal Results for {img_name}:")
print(f".   Coordinates are {img_lat}, {img_lon}.")
if len(classes) > 1:
    print(f".   You are predicted to be at {predictedLabel} at {classes[index[0]]}.")
else:
    print(f".   You are predicted to be at {predictedLabel} at {classes[index.item()]}.")
print(f".   The real location is at {img_label} at {img_building}.")

print()
print("KNN Percentage Breakdown (%):")
for building in neighbors:
    print(f".   {(neighbors[building] / bestk * 100):.2f}% -> {building}")

print()
print("CNN Percentage Breakdown (%):")
if len(classes) > 1:
    for i in range(0, len(classes)):
        print(f".   {probs[i] * 100:.2f}% -> {classes[index[i]]}")
else: # Account for single class, 0-sized tensor.
    print(f".   {probs.item() * 100:.2f}% -> {classes[index.item()]}")

