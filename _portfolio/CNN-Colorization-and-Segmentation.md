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
    image: https://resources.appen.com/wp-content/uploads/2019/04/SLIDER-Appen_image_annotation_05.jpg
    teaser: https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQRlJQ5PW9ZldsW5qikzkz77wBYepnnNtmbNQ&usqp=CAU
# sidebar:
#   - title: "t-SNE visualization"
#     image: ../assets/imgs/posts/language_model_files/language_model_40_1.png
#     image_alt: "t-SNE visualization"
#     text: "Feature Vector Alignment"
gallery:
  - url: ../assets/imgs/posts/CNN/color.png
    image_path: ../assets/imgs/posts/CNN/color.png
    alt: "Visual 2"
  - url: ../assets/imgs/posts/CNN/color2.png
    image_path: ../assets/imgs/posts/CNN/color2.png
    alt: "Visual 3"
  - url: ../assets/imgs/posts/CNN/color_g.png
    image_path: ../assets/imgs/posts/CNN/color_g.png
    alt: "Visual graph"
  - url: ../assets/imgs/posts/CNN/color_g2.png
    image_path: ../assets/imgs/posts/CNN/color_g2.png
    alt: "Visual 3"
gallery2:
  - url: ../assets/imgs/posts/CNN/Segment.png
    image_path: ../assets/imgs/posts/CNN/Segment.png
    alt: "Visual 2"
  - url: ../assets/imgs/posts/CNN/Segment2.png
    image_path: ../assets/imgs/posts/CNN/Segment2.png
    alt: "Visual 3"
  - url: ../assets/imgs/posts/CNN/Segment3.png
    image_path: ../assets/imgs/posts/CNN/Segment3.png
    alt: "Visual graph"
  - url: ../assets/imgs/posts/CNN/Segment4.png
    image_path: ../assets/imgs/posts/CNN/Segment4.png
    alt: "Visual 4"
  - url: ../assets/imgs/posts/CNN/Segment5.png
    image_path: ../assets/imgs/posts/CNN/Segment5.png
    alt: "Visual 5"
  - url: ../assets/imgs/posts/CNN/Segment6.png
    image_path: ../assets/imgs/posts/CNN/Segment6.png
    alt: "Visual 6"
  - url: ../assets/imgs/posts/CNN/Segment7.png
    image_path: ../assets/imgs/posts/CNN/Segment7.png
    alt: "Visual 6"
---
> [<i class="fas fa-infinity"></i>](https://colab.research.google.com/drive/1LgpiuXeB8U7pFv2-7OBQFGIQZHDoQ295?usp=sharing) Code Source 

# Introduction
This project works with Convolutional Neural Networks and exploring its applications. We will mainly focus on two famous tasks:

* Image Colorization: given a grey scale image, we need to predict its color in each pixel
    - difficulties: ill-posed problem -- multiple equally valid colorings
    - dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) constains 60000 32x32 colour images in 10 classes, with 6000 images per class, where 50000 are training images, and 10000 are test images. The 10 classes are: horse, automobile, bird, cat, deer, dog, frog, horse, ship and truck. Our main focus in the horse class

    <figure>
	<a href="https://miro.medium.com/max/875/1*SZnidBt7CQ4Xqcag6rd8Ew.png">
    <img src="https://miro.medium.com/max/875/1*SZnidBt7CQ4Xqcag6rd8Ew.png"></a>
	<figcaption>CIFAR-10 Dataset</figcaption>
    </figure>

    {% include gallery caption="CIFAR-10 Training Results" %}

* Semantic Segmentation: cluster areas of an image which belongs to the same object/label, and color with the same color section
    - dataset: [Oxford 17 Flowers Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/) 17 categories of flowers with 80 images in each set
    - approach: Using [Microsoft COCO Dataset](https://arxiv.org/abs/1405.0312), [deeplabv3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) especially as a finetuning base, and perform semantic segmentation

    <figure>
	<a href="https://www.researchgate.net/profile/Junbin_Gao/publication/289587379/figure/fig11/AS:614171376902164@1523441275987/Some-images-of-the-Oxford-Flowers-17-dataset.png">
    <img src="https://www.researchgate.net/profile/Junbin_Gao/publication/289587379/figure/fig11/AS:614171376902164@1523441275987/Some-images-of-the-Oxford-Flowers-17-dataset.png"></a>
	<figcaption>17 Flowers Dataset</figcaption>
    </figure>

    {% include gallery id="gallery2" caption="17 Flowers Training Results" %}


