import src.exif
import src.knn
import src.cnn

# CHANGE THESE WHEN RUNNING PROGRAM
# TODO: Make these .env
PARENT_DIRECTORY = ""
PREDICTED_IMAGE = ""

df = src.exif.exif(PARENT_DIRECTORY, PREDICTED_IMAGE)


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