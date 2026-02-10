# Rebel Locate &nbsp;🌎

A UNLV-based geolocator, leveraging machine learning algorithms and dynamic learning. 
Image recognition and coordinate-based supervised learning were implemented to predict input images from a self-captured, 6,000+ photo dataset.

**[Demo](https://docs.google.com/presentation/d/1KvvYFAokP8HvaATkJNQrT9pRTmb60Vqf/edit?rtpof=true&sd=true)** &nbsp; **[Report](https://docs.google.com/presentation/d/1KvvYFAokP8HvaATkJNQrT9pRTmb60Vqf/edit?rtpof=true&sd=true)**


## Model Background &nbsp;⚙️

Rebel Locate's leverages transfer learning, courtesy of MIT's Places365 pretrained weights. The ResNet-18 model that was utilized for Rebel Locate can be found [here](https://github.com/CSAILVision/places365).

This model is divided into a sequential, three-stage process: EXIF metadata extraction, application of K-Nearest Neighbors (KNN), and a Convolutional Neural Network (CNN).

### 1.  EXIF Metadata Extraction

### 2.  K-Nearest Neighbors (KNN)

### 3. Convolutional Neural Network (CNN)


## Dataset &nbsp;📊

All photos were self-captured across UNLV from 3/3/25 - 3/29/25 on an iPhone 14 Pro. Every image contains EXIF coordinate metadata, metrics that are utilized within the model.

Dataset: [Rebel Locate Dataset](https://drive.google.com/drive/u/4/folders/1_shwU9ab9lqalvdD6KOODgHzUmMU253G)
Samples: 6,000+  
Building Labels: 50 building names
Room Labels: Dependent on building selected.

### Example: Hospitality Hall (HOS)
Dataset: [Hospitality Hall Dataset](https://drive.google.com/drive/u/4/folders/1pN7sW4Xw3pnXTsJtfu0VeUBef2c0_TbY)
Samples: 393
Building Label: HOS
Room Labels:
- bathroom
- hallway
- lounge
- outside
- stairwell

