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

from sklearn import preprocessing, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split

def cnn(predictedLabel, possibleAreas, PARENT_DIRECTORY, PREDICTED_IMAGE):

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
    for epoch in range(1):
    # for epoch in range(10):
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

    return predictedLabel, classes, probs, index