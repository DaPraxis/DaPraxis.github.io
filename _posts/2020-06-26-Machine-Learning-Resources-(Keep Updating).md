---
title:  "Machine Learning Resources (Keep Updating)"
search: false
excerpt: 'Machine Learning Resources, Dataset, Open-sourced Algorithm and Online Learning Resources'
categories: 
  - Machine Learning
  - Resource Collection
  - Datasets
last_modified_at: 2020-07-19 10:00
comments: true
toc: true
toc_sticky: true
header:
  image: https://images.unsplash.com/photo-1535350356005-fd52b3b524fb?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80
  teaser: https://images.unsplash.com/photo-1535350356005-fd52b3b524fb?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80
---
## Paper
+ CV
    - [Paper with Code](https://paperswithcode.com/)
    - [Distill](https://distill.pub/)

## Learning Resources
+ General
    - [AI Hub](https://aihub.cloud.google.com/)
    - [Kaggle](https://www.kaggle.com/)
    - [Awesome-Pytorch-List](https://github.com/bharathgs/Awesome-pytorch-list)

+ NLP
    - [The Big Bad NLP Database Notebooks](https://notebooks.quantumstat.com/)
    - [Speech and Language Processing](https://www.cs.colorado.edu/~martin/slp2.html#Chapter3)

+ CV
    - [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
    - [Awesome-Computer-Vision](https://github.com/jbhuang0604/awesome-computer-vision)
    - [MedicalTorch](https://medicaltorch.readthedocs.io/en/stable/getstarted.html)
        - Built on top of PyTorch. Basic functionalities of the tools offered in MedicalTroch: pre-processing of images, transformations, and data loaders.
    - [The Computer Vision Industry](https://www.cs.ubc.ca/~lowe/vision.html)
    - [Computer-Vision-Basics-with-Python-Keras-and-OpenCV](https://github.com/jrobchin/Computer-Vision-Basics-with-Python-Keras-and-OpenCV)
    - Blogs
        - [Learn OpenCV](https://www.learnopencv.com/)
        - [Tombone's Computer Vision Blog](https://www.computervisionblog.com/)
        - [Andrej Karpathy blog](http://karpathy.github.io/)
        - [AI Shack](https://aishack.in/)

## Database
+ NLP
    - [The Big Bad NLP Database](https://datasets.quantumstat.com/)
+ CV
    - Classical:
        * [Oxford 17 Flowers Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
        * [CIFAR-10 Classes Tiny Image Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
    - Object Detection:
        * [Facebook Detectron2 COCO Dataset](https://github.com/diegobonilla98/Detectron2-Facebook-COCO-dataset)
        * [Image Net](http://www.image-net.org/)
        * [Open Image](https://storage.googleapis.com/openimages/web/index.html)
        * [MS COCO](https://cocodataset.org/#home)
    - Object Classification:
        * [Image Net](http://www.image-net.org/)
        * [Open Image](https://storage.googleapis.com/openimages/web/index.html)
    - Key Points:
        * [MS COCO](https://cocodataset.org/#home)

## Pre-trained CV Models
### Tensorflow <a name="tensorflow"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [ObjectDetection]( https://github.com/tensorflow/models/tree/master/research/object_detection)  | Localizing and identifying multiple objects in a single image.| `Tensorflow`
| [Mask R-CNN]( https://github.com/matterport/Mask_RCNN)  | The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.     | `Tensorflow`
| [Faster-RCNN]( https://github.com/smallcorgi/Faster-RCNN_TF)  | This is an experimental Tensorflow implementation of Faster RCNN - a convnet for object detection with a region proposal network.     | `Tensorflow`
| [YOLO TensorFlow]( https://github.com/gliese581gg/YOLO_tensorflow)  | This is tensorflow implementation of the YOLO:Real-Time Object Detection.     | `Tensorflow`
| [YOLO TensorFlow ++]( https://github.com/thtrieu/darkflow)  | TensorFlow implementation of 'YOLO: Real-Time Object Detection', with training and an actual support for real-time running on mobile devices.     | `Tensorflow`
| [MobileNet]( https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)  | MobileNets trade off between latency, size and accuracy while comparing favorably with popular models from the literature.     | `Tensorflow`
| [DeepLab]( https://github.com/tensorflow/models/tree/master/research/deeplab)  | Deep labeling for semantic image segmentation.     | `Tensorflow`
| [Colornet]( https://github.com/pavelgonchar/colornet)  | Neural Network to colorize grayscale images.     | `Tensorflow`
| [SRGAN]( https://github.com/tensorlayer/srgan)  | Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.    | `Tensorflow`
| [DeepOSM]( https://github.com/trailbehind/DeepOSM)  | Train TensorFlow neural nets with OpenStreetMap features and satellite imagery.     | `Tensorflow`
| [Domain Transfer Network]( https://github.com/yunjey/domain-transfer-network)  | Implementation of Unsupervised Cross-Domain Image Generation.  | `Tensorflow`
| [Show, Attend and Tell]( https://github.com/yunjey/show-attend-and-tell)  | Attention Based Image Caption Generator.     | `Tensorflow`
| [android-yolo]( https://github.com/natanielruiz/android-yolo)  | Real-time object detection on Android using the YOLO network, powered by TensorFlow.    | `Tensorflow`
| [DCSCN Super Resolution]( https://github.com/jiny2001/dcscn-super-resolutiont)  | This is a tensorflow implementation of "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network", a deep learning based Single-Image Super-Resolution (SISR) model.     | `Tensorflow`
| [GAN-CLS]( https://github.com/zsdonghao/text-to-image)  | This is an experimental tensorflow implementation of synthesizing images.     | `Tensorflow`
| [U-Net]( https://github.com/zsdonghao/u-net-brain-tumor)  | For Brain Tumor Segmentation.     | `Tensorflow`
| [Improved CycleGAN]( https://github.com/luoxier/CycleGAN_Tensorlayer)  |Unpaired Image to Image Translation.     | `Tensorflow`
| [Im2txt]( https://github.com/tensorflow/models/tree/master/research/im2txt)  | Image-to-text neural network for image captioning.     | `Tensorflow`
| [Street]( https://github.com/tensorflow/models/tree/master/research/street)  | Identify the name of a street (in France) from an image using a Deep RNN. | `Tensorflow`
| [SLIM]( https://github.com/tensorflow/models/tree/master/research/slim)  | Image classification models in TF-Slim.     | `Tensorflow`
| [DELF]( https://github.com/tensorflow/models/tree/master/research/delf)  | Deep local features for image matching and retrieval.     | `Tensorflow`
| [Compression]( https://github.com/tensorflow/models/tree/master/research/compression)  | Compressing and decompressing images using a pre-trained Residual GRU network.     | `Tensorflow`
| [AttentionOCR]( https://github.com/tensorflow/models/tree/master/research/attention_ocr)  | A model for real-world image text extraction.     | `Tensorflow`

***
### Keras <a name="keras"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [Mask R-CNN]( https://github.com/matterport/Mask_RCNN)  | The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.| `Keras`
| [VGG16]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py)  | Very Deep Convolutional Networks for Large-Scale Image Recognition.     | `Keras`
| [VGG19]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)  | Very Deep Convolutional Networks for Large-Scale Image Recognition.     | `Keras`
| [ResNet]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py)  | Deep Residual Learning for Image Recognition.     | `Keras`
| [Image analogies]( https://github.com/awentzonline/image-analogies)  | Generate image analogies using neural matching and blending.     | `Keras`
| [Popular Image Segmentation Models]( https://github.com/divamgupta/image-segmentation-keras)  | Implementation of Segnet, FCN, UNet and other models in Keras.     | `Keras`
| [Ultrasound nerve segmentation]( https://github.com/jocicmarko/ultrasound-nerve-segmentation)  | This tutorial shows how to use Keras library to build deep neural network for ultrasound image nerve segmentation.     | `Keras`
| [DeepMask object segmentation]( https://github.com/abbypa/NNProject_DeepMask)  | This is a Keras-based Python implementation of DeepMask- a complex deep neural network for learning object segmentation masks.     | `Keras`
| [Monolingual and Multilingual Image Captioning]( https://github.com/elliottd/GroundedTranslation)  | This is the source code that accompanies Multilingual Image Description with Neural Sequence Models.     | `Keras`
| [pix2pix]( https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)  | Keras implementation of Image-to-Image Translation with Conditional Adversarial Networks by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A.    | `Keras`
| [Colorful Image colorization]( https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/Colorful)  | B&W to color.   | `Keras`
| [CycleGAN]( https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py)  | Implementation of _Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_.    | `Keras`
| [DualGAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/dualgan/dualgan.py)  | Implementation of _DualGAN: Unsupervised Dual Learning for Image-to-Image Translation_.   | `Keras`
| [Super-Resolution GAN]( https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py)  | Implementation of _Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_.   | `Keras`

***

### PyTorch <a name="pytorch"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [FastPhotoStyle]( https://github.com/NVIDIA/FastPhotoStyle)  | A Closed-form Solution to Photorealistic Image Stylization.   | `PyTorch`
| [pytorch-CycleGAN-and-pix2pix]( https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  | A Closed-form Solution to Photorealistic Image Stylization.   | `PyTorch`
| [maskrcnn-benchmark]( https://github.com/facebookresearch/maskrcnn-benchmark)  | Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch.   | `PyTorch`
| [deep-image-prior]( https://github.com/DmitryUlyanov/deep-image-prior)  | Image restoration with neural networks but without learning.   | `PyTorch`
| [StarGAN]( https://github.com/yunjey/StarGAN)  | StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation.   | `PyTorch`
| [faster-rcnn.pytorch]( https://github.com/jwyang/faster-rcnn.pytorch)  | This project is a faster faster R-CNN implementation, aimed to accelerating the training of faster R-CNN object detection models.   | `PyTorch`
| [pix2pixHD]( https://github.com/NVIDIA/pix2pixHD)  | Synthesizing and manipulating 2048x1024 images with conditional GANs.  | `PyTorch`
| [Augmentor]( https://github.com/mdbloice/Augmentor)  | Image augmentation library in Python for machine learning.  | `PyTorch`
| [albumentations]( https://github.com/albumentations-team/albumentations)  | Fast image augmentation library.   | `PyTorch`
| [Deep Video Analytics]( https://github.com/AKSHAYUBHAT/DeepVideoAnalytics)  | Deep Video Analytics is a platform for indexing and extracting information from videos and images   | `PyTorch`
| [semantic-segmentation-pytorch]( https://github.com/CSAILVision/semantic-segmentation-pytorch)  | Pytorch implementation for Semantic Segmentation/Scene Parsing on MIT ADE20K dataset.   | `PyTorch`
| [An End-to-End Trainable Neural Network for Image-based Sequence Recognition]( https://github.com/bgshih/crnn)  | This software implements the Convolutional Recurrent Neural Network (CRNN), a combination of CNN, RNN and CTC loss for image-based sequence recognition tasks, such as scene text recognition and OCR.   | `PyTorch`
| [UNIT]( https://github.com/mingyuliutw/UNIT)  | PyTorch Implementation of our Coupled VAE-GAN algorithm for Unsupervised Image-to-Image Translation.   | `PyTorch`
| [Neural Sequence labeling model]( https://github.com/jiesutd/NCRFpp)  | Sequence labeling models are quite popular in many NLP tasks, such as Named Entity Recognition (NER), part-of-speech (POS) tagging and word segmentation.   | `PyTorch`
| [faster rcnn]( https://github.com/longcw/faster_rcnn_pytorch)  | This is a PyTorch implementation of Faster RCNN. This project is mainly based on py-faster-rcnn and TFFRCNN. For details about R-CNN please refer to the paper Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.   | `PyTorch`
| [pytorch-semantic-segmentation]( https://github.com/ZijunDeng/pytorch-semantic-segmentation)  | PyTorch for Semantic Segmentation.   | `PyTorch`
| [EDSR-PyTorch]( https://github.com/thstkdgus35/EDSR-PyTorch)  | PyTorch version of the paper 'Enhanced Deep Residual Networks for Single Image Super-Resolution'.   | `PyTorch`
| [image-classification-mobile]( https://github.com/osmr/imgclsmob)  | Collection of classification models pretrained on the ImageNet-1K.   | `PyTorch`
| [FaderNetworks]( https://github.com/facebookresearch/FaderNetworks)  | Fader Networks: Manipulating Images by Sliding Attributes - NIPS 2017.   | `PyTorch`
| [neuraltalk2-pytorch]( https://github.com/ruotianluo/ImageCaptioning.pytorch)  | Image captioning model in pytorch (finetunable cnn in branch with_finetune).   | `PyTorch`
| [RandWireNN]( https://github.com/seungwonpark/RandWireNN)  | Implementation of: "Exploring Randomly Wired Neural Networks for Image Recognition".   | `PyTorch`
| [stackGAN-v2]( https://github.com/hanzhanggit/StackGAN-v2)  |Pytorch implementation for reproducing StackGAN_v2 results in the paper StackGAN++.   | `PyTorch`
| [Detectron models for Object Detection]( https://github.com/ignacio-rocco/detectorch)  | This code allows to use some of the Detectron models for object detection from Facebook AI Research with PyTorch.   | `PyTorch`
| [DEXTR-PyTorch]( https://github.com/scaelles/DEXTR-PyTorch)  | This paper explores the use of extreme points in an object (left-most, right-most, top, bottom pixels) as input to obtain precise object segmentation for images and videos.   | `PyTorch`
| [pointnet.pytorch]( https://github.com/fxia22/pointnet.pytorch)  | Pytorch implementation for "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.   | `PyTorch`
| [self-critical.pytorch]( https://github.com/ruotianluo/self-critical.pytorch) | This repository includes the unofficial implementation Self-critical Sequence Training for Image Captioning and Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering.   | `PyTorch`
| [vnet.pytorch]( https://github.com/mattmacy/vnet.pytorch)  | A Pytorch implementation for V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.   | `PyTorch`
| [piwise]( https://github.com/bodokaiser/piwise)  | Pixel-wise segmentation on VOC2012 dataset using pytorch.   | `PyTorch`
| [pspnet-pytorch]( https://github.com/Lextal/pspnet-pytorch)  | PyTorch implementation of PSPNet segmentation network.   | `PyTorch`
| [pytorch-SRResNet]( https://github.com/twtygqyy/pytorch-SRResNet)  | Pytorch implementation for Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.   | `PyTorch`
| [PNASNet.pytorch]( https://github.com/chenxi116/PNASNet.pytorch)  | PyTorch implementation of PNASNet-5 on ImageNet.   | `PyTorch`
| [img_classification_pk_pytorch]( https://github.com/felixgwu/img_classification_pk_pytorch)  | Quickly comparing your image classification models with the state-of-the-art models.   | `PyTorch`
| [Deep Neural Networks are Easily Fooled]( https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks)  | High Confidence Predictions for Unrecognizable Images.   | `PyTorch`
| [pix2pix-pytorch]( https://github.com/mrzhu-cool/pix2pix-pytorch)  | PyTorch implementation of "Image-to-Image Translation Using Conditional Adversarial Networks".   | `PyTorch`
| [NVIDIA/semantic-segmentation]( https://github.com/NVIDIA/semantic-segmentation)  | A PyTorch Implementation of Improving Semantic Segmentation via Video Propagation and Label Relaxation, In CVPR2019.   | `PyTorch`
| [Neural-IMage-Assessment]( https://github.com/kentsyx/Neural-IMage-Assessment)  | A PyTorch Implementation of Neural IMage Assessment.   | `PyTorch`
| [torchxrayvision](https://github.com/mlmed/torchxrayvision) | Pretrained models for chest X-ray (CXR) pathology predictions. Medical, Healthcare, Radiology  | `PyTorch` | 

## Useful API
+ General
    - [API Coding](https://apicoding.io/)