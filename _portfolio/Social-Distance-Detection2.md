---
title:  "Social Distance Detection Application - Phase 2"
search: false
excerpt: 'Object Detection Model Benchmark on YOLOv5 & Detectron2, Bird View Coordinates Transformation'
categories: 
  - Computer Vision
  - Transfer Learning
  - Python
  - Data Visualization
  - Software Engineering
last_modified_at: 2020-08-07T08:06:00-07:00
comments: true
mathjax: true
toc: true
toc_sticky: true
header:
    image: https://cdn.dribbble.com/users/1403099/screenshots/4191931/chopsticks.gif
    teaser: https://i.pinimg.com/originals/1d/19/80/1d19807590683341b67c770284e2d3e8.gif
# sidebar:
#   - title: "t-SNE visualization"
#     image: ../assets/imgs/posts/language_model_files/language_model_40_1.png
#     image_alt: "t-SNE visualization"
#     text: "Feature Vector Alignment"
gallery:
  - url: ../assets/imgs/posts/Social/eucl.gif
    image_path: ../assets/imgs/posts/Social/eucl.gif
    alt: "Euclidean Distance"
  - url: ../assets/imgs/posts/Social/bbird-eye.gif
    image_path: ../assets/imgs/posts/Social/bbird-eye.gif
    alt: "Bird-eye Converted"
gallery1:
  - url: ../assets/imgs/posts/Social/yolo.gif
    image_path: ../assets/imgs/posts/Social/yolo.gif
    alt: "YOLO benchmark"
gallery2:
  - url: https://images.unsplash.com/photo-1488034976201-ffbaa99cbf5c?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=334&q=80
    image_path: https://images.unsplash.com/photo-1488034976201-ffbaa99cbf5c?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=334&q=80
    alt: "Toronto"
  - url: https://imgur.com/ToIf1na
    image_path: https://i.imgur.com/ToIf1na.jpg
    alt: "YOLOv5 Toronto"
  - url: https://i.imgur.com/g7IVrbJ.jpg
    image_path: https://i.imgur.com/g7IVrbJ.jpg
    alt: "Detectron2 Toronto"  
---
<object data="../../assets/imgs/posts/Social/Report.pdf" width="1000" height="1000" type='application/pdf'></object>

# DEMO
{% include gallery caption="Distance Measuring with Euclidean(left) & Bird-eye view(right)" %}

<!-- {% capture fig_img %}
[![image-center](https://i.imgur.com/cS7Fqci.gif)](https://i.imgur.com/cS7Fqci.gif){: .align-center}
{% endcapture %} -->

<!-- <figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure> -->

# Project Discription
This article continues the ["Social Distance Detection Application - Phase 1"](https://dapraxis.github.io/portfolio/Social-Distance-Detection/) topic. In this post, we will talk about *object detection models* in depth, performance for *YOLOv5 and Detectron2 in Colab* default GPU and the *bird eye view conversion* improves social distance measurement

Mask ON 😷. 

# YOLOv5
YOLOv5 is the latest model(up to 8/11/2020) of YOLO family, and is inarguably the *fastest* and *most accurate* YOLO model amongst or versions, all even in the industry. YOLO is notoriously famous for its inference speed and weight/model size, and the team is also crazy about enhancing the performance on those aspects. Currently the fastest speed for video object detection is up to 140FPS, and even compatible on mobile devices. 

<figure>
	<a href="https://arxiv.org/pdf/2004.10934.pdf">
    <img src="https://miro.medium.com/max/815/1*32ucN5yYa3ldqEDJEqJEpA.png"></a>
	<figcaption>Previous YOLO Family Models on YOLOv4 Paper</figcaption>
    </figure>

<figure>
	<a href="https://github.com/ultralytics/yolov5">
    <img src="https://user-images.githubusercontent.com/26833433/85340570-30360a80-b49b-11ea-87cf-bdf33d53ae15.png"></a>
	<figcaption>YOLOv5 Model Performance on YOLOv5 Github</figcaption>
    </figure>

## Model Architecture
* Backbone: [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929)

  Backbones mainly are used for extracting importance features from input image/frames. YOLOv5 uses CSPNet backbone to accelerate the image processing speed.

  <figure>
	<a href="https://github.com/WongKinYiu/CrossStagePartialNetworks">
    <img src="https://github.com/WongKinYiu/CrossStagePartialNetworks/blob/master/fig/cmp3.png?raw=true"></a>
	<figcaption>CSPNet Backbone Performance</figcaption>
    </figure>

* Neck: [Path Aggregation Network for Instance Segmentation(PANet)](https://arxiv.org/abs/1803.01534)

  Model necks are mainly used for feature pyramids generation, which works perfectly on detecting same objects in different scales. Model neck passes image features to prediction layer and feed into the head. Some other famous feature pyramids are [FPN](https://arxiv.org/abs/1708.02002) and [BiFPN](https://arxiv.org/abs/1911.09070), which works pretty well with Fast R-CNN and Faster R-CNN

    <figure>
	<a href="https://arxiv.org/pdf/1612.03144.pdf">
    <img src="https://miro.medium.com/max/875/1*UtfPTLB53cR8EathGBOT2Q.jpeg"></a>
	<figcaption>Feature Pyramid Networks for Object Detection </figcaption>
    </figure>

* Head: Detection on features, generating bounding boxes and predict categories. 
  ![image-center]({{ site.url }}{{ site.baseurl }}../assets/imgs/posts/Social/YOLO architecture.jpg){: .align-center}

## Inferencing & Predicting
### Data Labeling
  YOLO family has its very own labeling format. Before we get to the pipeline setup, let's first look at how it labels and outputs for each image

  For each `{image}.jpg`, we have a labeled file `{image}.jpg` with the same name on `{image}.txt`. 
  
  In the output file `{image}.txt`, we have format `<object-class><x><y><width><height>`, where
  * [`<object-class>`](https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml) represents object category with: 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', ... where 0 stands for 'person', 1 stands for 'bicycle' and so on
  * `<x><y><width><height>` are float point proportional to image actual height and width, i.e. `<height> = <abs_height>/<image_height>`. Ranges from 0.0-1.0, where `<x><y>` is the center of the image.

{% include gallery id='gallery2' caption="Object Detection on image Toronto.jpg(original, YOLOv5, Detectron2)" %}

  $$\begin{array}{|l|l|l|l|l|}
\hline \text { Class} & \text { x } & \text { y } & \text { width } & \text { height } \\
\hline 2 & 0.0699219 & 0.672917 & 0.139844 & 0.265278  \\
\hline
\end{array}$$

### Pipeline Setup

> [<i class="fas fa-infinity"></i>](https://colab.research.google.com/drive/1DH1l-Dfnnta0Lb58YEc_PgAs0kwXP5zy?usp=sharing) Code Source 

One huge improvement for YOLOv5 is that compared to previous YOLO models, YOLOv5 is purely written in PyTorch, and therefore well maintained and ready for production. 

The code setup in Colab is also very easy. First download and install the YOLOv5 dependency by running block
```ruby
!git clone https://github.com/ultralytics/yolov5  # clone repo
!pip install -U -r yolov5/requirements.txt  # install dependencies
%cd /content/yolov5
```
Make show your runtime is GPU in colab as well.

The the dependency
```ruby
import torch
from IPython.display import Image  # for displaying images
from utils.google_utils import gdrive_download  # for downloading models/datasets
print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.torch_utils import *
import utils.torch_utils as torch_utils
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from google.colab.patches import cv2_imshow
```
After that, put in all the images/video you want to detect inside colab directory `yoloV5/inference/images`, and then run either the following blocks. The result can be obtained in directory `yoloV5/inference/output`

The first block does all the jobs already, which simply runs the `detect.py` in `yoloV5`, but with preset parameters
```ruby
!python detect.py --source ./inference/images/ --save-txt --classes 0 --weights yolov5s.pt --conf 0.4
```
The following block spread out the `detect.py` and customize the benchmark.

You can also checkout YOLOv5 [official doc](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) for fine-tunning the model and training on your own dataset

# Benchmark On YOLOv5 & Detectron2
* Detectron2 Baseline

  > [<i class="fas fa-infinity"></i>](https://colab.research.google.com/drive/1Mvs5pGpYEKoq2EHxQS8eJlJdgRlqSNPb?usp=sharing) Code Source 

* Dataset

  The dataset we are using are Multi-camera pedestrians video from [EPFL](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/), [Joint Attention in Autonomous Driving (JAAD) Dataset](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) and some uncalibrated camera videos donated as ‘custom dataset’. Those dataset are selected purposely since our social distance detection program will mainly be used for public area pedestrian walks, and analyzing real time camera footages. The dataset from EPFL contains simulation for multiple person random walking, which can be used to test on the model's computation capability. Dataset JAAD contains footage shot in cars, and videos contain various crosswalks and pedestrians are selected to test on the scalability of the models. Finally, the custom dataset is also selected to increase the variability among datasets and test on models’ robustness. 

* Testing Environment

  This test is set up in the Colab notebook environment, facilitated by default GPU (Tesla K80 GPU) setting. For Detectron2, since there are a lot of model settings, we are using Fast R-CNN R50-FPN backbone on RPN & Fast R-CNN baseline model for top-class prediction accuracy. For YOLOv5, we accept all predefined parameters such as the number of full connections and CNN layers. We use pre trained YOLOv5l weight to maintain a fast GPU speed, while achieving high AP(Average Precision).

$$\begin{array}{|l|l|l|l|l|}
\hline \text { FPS} & \text { JAAD video 0067 } & \text { EPFL 6p-c1 } & \text { EPFL 4p-c0 } & \text { Custom video } \\
\hline \text { YOLOv5 } & 0.013 \pm 0.002 \mathrm{s} & 0.011 \pm 0.002 \mathrm{s} & 0.011 \pm 0.001 \mathrm{s} & 0.031 \pm 0.01 \mathrm{s} 28 \\
\hline \text { Detectron2 } & 0.512 \pm 0.201 \mathrm{s} & 0.332 \pm 0.521 \mathrm{s} & 0.385 \pm 0.028 \mathrm{s} & 0.529 \pm 0.511 \mathrm{s} \\
\hline
\end{array}$$
Table 1: Object Detection Inference Speed per 300 frames

$$\begin{array}{|l|l|l|l|l|}
\hline \text { Error} & \text { JAAD video 0067 } & \text { EPFL 6p-c1 } & \text { EPFL 4p-c0 } & \text { Custom video } \\
\hline \text { YOLOv5 } & 1 / 300 \mathrm{frm} & 3 / 300 \mathrm{frm} & 0 / 300 \mathrm{frm} & 0 / 300 \mathrm{frm} \\
\hline \text { Detectron2 } & 0 / 300 \mathrm{frm} & 2 / 300 \mathrm{frm} & 0 / 300 \mathrm{frm} & 0 / 300 \mathrm{frm} \\
\hline
\end{array}$$
Table 2: Object Detection Inference Accuracy per 300 frames


{% include gallery id='gallery1' caption="YOLOv5 Benchmark on EPFL 6p-c1" %}

# Bird Eye View Conversion
This part it very tricky, even a *trade secret* for some small companies. The Bird-Eye-View problem is mainly a problem of perspectives, where though distance between people is finite, it changes with the distance between objects and lens, i.e. the extrinsic matrix. 

<figure>
	<a href="https://i.stack.imgur.com/xGeeC.png">
    <img src="https://i.stack.imgur.com/xGeeC.png"></a>
	<figcaption>3D to 2D mapping in camera</figcaption>
    </figure>

In real life, the real-world distance would be much easier to estimate with extrinsic matrix, intrinsic matrix, or multiple cameras if you have a full control of the dataset, and even perform a 3D stereo reconstruction. 

<figure>
	<a href="https://www.researchgate.net/profile/Mohamad_Hanif_Md_Saad/publication/318452089/figure/fig6/AS:669681593614369@1536675942999/3D-reconstruction-from-stereo-camera-images.jpg">
    <img src="https://www.researchgate.net/profile/Mohamad_Hanif_Md_Saad/publication/318452089/figure/fig6/AS:669681593614369@1536675942999/3D-reconstruction-from-stereo-camera-images.jpg"></a>
	<figcaption>3D Stereo Reconstruction from Images</figcaption>
    </figure>

However, in our case, we have no way to find the real world distance with above matrices. Therefore, we used a four-point mapping technique, selecting a region in picture that is rectangular in real world(usually the road) and performs a 2D wrapping to actually make it rectangle. All the distance will be calculated based on the wrapped coordinates. 

We know that this method is limited, error-prone and needs human labeling for different videos, but the result is ideal. 

Take a look at the [github code](https://github.com/EvanSamaa/EyeDK/blob/master/eyedk_utils.py#L634-L832) 

# Contact Me
If you are interested in my projects or have any new ideas you wanna talk about, feel free to [contact](mailto:haoyanhy.jiang@mail.utoronto.ca) me!

A **BEER** would be perfect, but remember **NO CORONA!** 🍻 
<style>.bmc-button img{height: 34px !important;width: 35px !important;margin-bottom: 1px !important;box-shadow: none !important;border: none !important;vertical-align: middle !important;}.bmc-button{padding: 7px 15px 7px 10px !important;line-height: 35px !important;height:51px !important;text-decoration: none !important;display:inline-flex !important;color:#ffffff !important;background-color:#000000 !important;border-radius: 5px !important;border: 1px solid transparent !important;padding: 7px 15px 7px 10px !important;font-size: 20px !important;letter-spacing:-0.08px !important;box-shadow: 0px 1px 2px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;margin: 0 auto !important;font-family:'Lato', sans-serif !important;-webkit-box-sizing: border-box !important;box-sizing: border-box !important;}.bmc-button:hover, .bmc-button:active, .bmc-button:focus {-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;text-decoration: none !important;box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;opacity: 0.85 !important;color:#ffffff !important;}</style><link href="https://fonts.googleapis.com/css?family=Lato&subset=latin,latin-ext" rel="stylesheet"><a class="bmc-button" target="_blank" href="https://www.buymeacoffee.com/MaxJiang"><img src="https://cdn.buymeacoffee.com/buttons/bmc-new-btn-logo.svg" alt="Buy me a Beer"><span style="margin-left:5px;font-size:19px !important;">Buy me a Beer</span></a>