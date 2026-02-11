# Rebel Locate &nbsp;🌎

A UNLV-based geolocator, leveraging machine learning algorithms and dynamic learning. 
Image recognition and coordinate-based supervised learning were implemented to predict input images from a self-captured, 6,000+ photo dataset.

**[Demo](https://docs.google.com/presentation/d/1KvvYFAokP8HvaATkJNQrT9pRTmb60Vqf/edit?rtpof=true&sd=true)** &nbsp; **[Report](https://docs.google.com/presentation/d/1KvvYFAokP8HvaATkJNQrT9pRTmb60Vqf/edit?rtpof=true&sd=true)**



## Model Background &nbsp;⚙️

This model is divided into a sequential, three-stage process: EXIF metadata extraction, application of K-Nearest Neighbors (KNN), and a Convolutional Neural Network (CNN).

#### 1. &nbsp;&nbsp; EXIF Metadata Extraction  
> All images contain EXIF metadata such as latitude, longitude, and phone information. Every image, building label, and 
converted coordinate are stored within an internal Pandas dataframe.

> For visualization, a Matplotlib scatter plot is displayed after parsing, outlining all image locations and building perimeters.

#### 2. &nbsp;&nbsp; K-Nearest Neighbors (KNN)
> A building label is predicted for an input image based on EXIF metadata and relative distance to nearby coordinates. K-Fold Cross Validation was implemented to find the optimum K value among all labels. 

> Based on final building prediction (i.e. BEH), CNN will only train on that corresponding directory name, increasing resource efficiency, training times, and classification accuracy.

#### 3. &nbsp;&nbsp; Convolutional Neural Network (CNN)  
> Rebel Locate's leverages transfer learning, courtesy of MIT's Places365 pretrained weights. The ResNet-18 model that was utilized for Rebel Locate can be found [here](https://github.com/CSAILVision/places365).

> CNN is trained only within the directory determined via KNN (i.e. BEH). Output nodes are dynamically determined based on the number of available room labels within this directory (i.e. classroom, bathroom, etc.). Classification accuracies from both KNN and CNN are displayed to the user after an 80/20 training split.



## Dataset &nbsp;📊

All photos were self-captured across UNLV from 3/3/25 - 3/29/25 on an iPhone 14 Pro. Every image contains EXIF coordinate metadata, metrics that are utilized within the model.

#### Rebel Locate Dataset
> Dataset: [Rebel Locate Dataset](https://drive.google.com/drive/u/4/folders/1_shwU9ab9lqalvdD6KOODgHzUmMU253G)  
> Samples: 6,000+  
> Building Labels: 50 building names  
> Room Labels: Dependent on building selected.

#### Example: Hospitality Hall (HOS)
> Dataset: [Hospitality Hall Dataset](https://drive.google.com/drive/u/4/folders/1pN7sW4Xw3pnXTsJtfu0VeUBef2c0_TbY)  
> Samples: 393  
> Building Label: HOS  
> Room Labels:  
>  -  &nbsp;bathroom  
>  -  &nbsp;hallway  
>  -  &nbsp;lounge  
>  -  &nbsp;outside  
>  -  &nbsp;stairwell  

