# Poinsettia-Challenge-Find the Pot and Plant
Here we will release the dataset and pre-training VGG model (https://drive.google.com/drive/folders/12DaZ6mhukyye1Y5j69CmK6TH6Q_EC0M2?usp=sharing)

## Poinsettia Dataset Structure
The dataset is based on 100 base images. The 100 image dataset is classified into two groups for the purpose of testing and training. Each training and testing sub-groups contains 80 and 20 images. Each image in the dataset consists of a corresponding detection label.

Numeric values of the labelled coordinates are stored in .xml files. All the label coordinates in train dataset are stored in a single .xml file. The zip file in Google Dirve mainly contain:

```bash
├── Annotations
│   ├── *.xml (those files containing image label coordinates for 100 base images )
├── ImageSets
│   ├── *.txt (the poinsettia for training and testing)
├── JPEGImages
│   ├── *.JPEG (the poinsettia images)
```
## Running the SSD code
Step1: new weights folder, zip the VOC2007.zip file into data/, put vgg16_reducedfc.pth into weights folder;

Step2: running the train.py;

Step3: eval.py to get mAP

## Detection results
The red rectangular is ground truth, and the blue is the prediction bounding box of SSD detection model, the score is confidence of pot and poinsettia classification.

poinsettia64            |  poinsettia35
:-------------------------:|:-------------------------:
![poinsettia64box](https://user-images.githubusercontent.com/6768016/145016510-f5630e9d-1903-49ef-8cb1-8876c48afa42.jpg)  |  ![poinsettia35box](https://user-images.githubusercontent.com/6768016/145016649-1e698dcb-ef1d-4243-a253-2991de48ac83.jpg)



