---
title:  "Convolution Neural Network"
search: false
excerpt: 'Colorization, Skip Connections and Semantic Segmentation'
categories: 
  - Computer Vision
  - CNN
  - Python
  - Data Visualization
last_modified_at: 2020-07-18T08:06:00-07:00
comments: true
mathjax: true
toc: true
header:
    image: https://i1.wp.com/bdtechtalks.com/wp-content/uploads/2019/01/computer-vision-object-detection.png?fit=2000%2C1118&ssl=1
    teaser: https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQRlJQ5PW9ZldsW5qikzkz77wBYepnnNtmbNQ&usqp=CAU
# sidebar:
#   - title: "t-SNE visualization"
#     image: ../assets/imgs/posts/language_model_files/language_model_40_1.png
#     image_alt: "t-SNE visualization"
#     text: "Feature Vector Alignment"
gallery1:
  - url: https://miro.medium.com/max/875/1*SZnidBt7CQ4Xqcag6rd8Ew.png
    image_path: https://miro.medium.com/max/875/1*SZnidBt7CQ4Xqcag6rd8Ew.png
    alt: "Visual 2"
  - url: https://www.researchgate.net/publication/322148855/figure/fig1/AS:577424834662400@1514680216761/Heterogeneousness-and-diversity-of-the-CIFAR-10-entries-in-their-10-image-categories-The.png
    image_path: https://www.researchgate.net/publication/322148855/figure/fig1/AS:577424834662400@1514680216761/Heterogeneousness-and-diversity-of-the-CIFAR-10-entries-in-their-10-image-categories-The.png
    alt: "Visual 3"
---
> [<i class="fas fa-infinity"></i>](https://colab.research.google.com/drive/1LgpiuXeB8U7pFv2-7OBQFGIQZHDoQ295?usp=sharing) Code Source 

# Introduction
This project works with Convolutional Neural Networks and exploring its applications. We will mainly focus on two famous tasks:

* Image Colorization: given a grey scale image, we need to predict its color in each pixel
    - difficulties: ill-posed problem -- multiple equally valid colorings
    - dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) constains 60000 32x32 colour images in 10 classes, with 6000 images per class, where 50000 are training images, and 10000 are test images. The 10 classes are: horse, automobile, bird, cat, deer, dog, frog, horse, ship and truck. Our main focus in the horse class

* Semantic Segmentation: cluster areas of an image which belongs to the same object/label, and color with the same color section
    - dataset: [Oxford 17 Flowers Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/) 17 categories of flowers with 80 images in each set
    - approach: Using [Microsoft COCO Dataset](https://arxiv.org/abs/1405.0312), [deeplabv3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) especially as a finetuning base, and perform semantic segmentation

{% include gallery1 caption="CIFAR-10 images" %}


