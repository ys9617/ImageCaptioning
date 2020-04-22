[image1]: ./img/image-captioning.png "Image Captioning Model"
[image2]: ./img/coco-examples.jpg "Sample Dataset Example"
[image3]: ./img/ResNet50-architecture.png "ResNet50"

# Image Captioning

Automatically generate captions from images.

![Image Captioning Model][image1]



## Overview

1. Initialize COCO API and obataining batches by using the data loader
2. Defining CNN encoder and RNN decoder architecture
3. Image Captioning result


## Data (MS COCO dataset)

The Microsoft C*ommon *Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.

![Sample Dataset Example][image2]

You can read more about the dataset on the [website](http://cocodataset.org/#home) or in the [research paper](https://arxiv.org/pdf/1405.0312.pdf).



## Defining CNN encoder and RNN decoder architecture

#### CNN encoder architecture

The encoder uses the pre-trained ResNet-50 architecture (with the final fully connected layer removed) to extract features from a batch of pre-processing images. The output is then flattend to a vector, before being passed through a linear layer to transform the feature vector to have the same size as the word embedding.

![ResNet50][image3]


#### RNN decoder architecture







