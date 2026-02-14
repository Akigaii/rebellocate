import src.exif
import src.knn
import src.cnn

# CHANGE THESE WHEN RUNNING PROGRAM
# TODO: Make these .env
PARENT_DIRECTORY = "c:\\Users\\aigaui\\Desktop\\rebellocate"
PREDICTED_IMAGE = "c:\\Users\\aigaui\\Desktop\\rebellocate\\BEH\\corridor\\IMG_4759.JPG"

df, img_name, img_label, img_building, oneHotDict = src.exif.exif(PARENT_DIRECTORY, PREDICTED_IMAGE)
predictedBuildingDirectory, predictedLabel, possibleAreas, img_lat, img_lon, bestk, neighbors  = src.knn.knn(df, img_name, img_label, img_building, oneHotDict, PARENT_DIRECTORY, PREDICTED_IMAGE)
predictedLabel, classes, probs, index = src.cnn.cnn(predictedLabel, possibleAreas, PARENT_DIRECTORY, PREDICTED_IMAGE)




# Print final results.
print(f"\nFinal Results for {img_name}:")
print(f".   Coordinates are {img_lat}, {img_lon}.")
if len(classes) > 1:
    print(f".   You are predicted to be at {predictedLabel} at {classes[index[0]]}.")
else:
    print(f".   You are predicted to be at {predictedLabel} at {classes[index.item()]}.")
print(f".   The real location is at {img_label} at {img_building}.")

print("\nKNN Percentage Breakdown (%):")
for building in neighbors:
    print(f".   {(neighbors[building] / bestk * 100):.2f}% -> {building}")

print("\nCNN Percentage Breakdown (%):")
if len(classes) > 1:
    for i in range(0, len(classes)):
        print(f".   {probs[i] * 100:.2f}% -> {classes[index[i]]}")
else: # Account for single class, 0-sized tensor.
    print(f".   {probs.item() * 100:.2f}% -> {classes[index.item()]}")