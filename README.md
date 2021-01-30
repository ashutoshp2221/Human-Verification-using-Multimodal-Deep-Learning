# Human-Recognition-using-Multimodal-Deep-Learning

This project explores the use of an aggregate network to verify the identity of a person using his facial and speech data.
The conventional method of deep learning is to process huge amounts of a certain type of data to do a certain job. But us humans don't 
process data in a similar fashion. We use both audio and visual cues to understand data. The objective of this project is to explore 
how deep learning models can perceive data in a similar way. 

We thus applied this concept to make a Multimodal Network that helps us verify the identity of a person. The idea is to make a neural 
network that can incorporate multimodalities and train with less amounts of data.

***
## Siamese Network

Our deep learning model uses a siamese network to perform a one-shot learning task from the aggregated features of both the individual face and audio networks. Using a metric space method we have trained the model to identify if two given data points belong to the same class or not.

<img src="./images/image1.png" alt="Sample" style="text-align:center; width: 400px;"/>

***
## Models

+ Base model: [VGG16](https://arxiv.org/abs/1409.1556)

***
## Datasets

+ [VoxCeleb1 Test](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)