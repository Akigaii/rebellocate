import os
import matplotlib.pyplot as plt
import numpy as np
from exif import Image
from sklearn.preprocessing import LabelEncoder


def exif(PARENT_DIRECTORY, PREDICTED_IMAGE):
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

    return df


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