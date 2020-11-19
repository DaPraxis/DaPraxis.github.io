---
title:  "YOLOv5 on Customized Teapots Dataset"
search: false
excerpt: 'YOLOv5 Transfer Learning'
categories: 
  - Computer Vision
  - Transfer Learning
  - CNN
  - Python
  - Data Visualization
last_modified_at: 2020-11-19T08:06:00-07:00
comments: true
mathjax: true
author_profile: true
toc: true
toc_sticky: true
header:
    image: ../assets/imgs/portfolios/teapots.jpg
    teaser: ../assets/imgs/portfolios/teaser_obj_dect.jpeg
---

Recently, I have been working on edge-side computer vision models (🚗,📱), and YOLOv5 with its outstanding computation speed and accuracy quickly fit into my list. I have been working with dataset in [Open Image](https://storage.googleapis.com/openimages/web/index.html) for a while, but sometimes, what you really want is just a little subset of it, rather than the entire hundreds GB with 100,000 classes. In this project, we only looking at one class: teapot🍵, fetching only data from that class quickly from OpenImage database and transfer learning on our favorite model: YOLOv5

This project will be in four parts: Data Fetching, Data Formatting & Processing, Model Training and Model Predictions

# Table of Content
- Data Fetching: [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit)
- Data Formatting & Processing: [RoboFlow](https://roboflow.com/)
- Model Training
- Model Prediction

# Data Fetching
Normally, I would write a scrapper with beautiful soup, but I have found this amazing open-source data fetching tool called OIDv4_ToolKit, which explicitly gathers all the data we need for certain classes in Open Images Dataset V4.

However, this software only supports V4, as the name states, let me know if you guys find a more general tool that works for all versions in Open Image
{: .notice--warning}

The code follows will be effective on Google Colab, for the universal operating environment or equivalent OS in Ubuntu.

First, we need to clone the software and install all the dependencies

```ruby
!git clone https://github.com/EscVM/OIDv4_ToolKit.git
!pip install -r OIDv4_ToolKit/requirements.txt
```

Then you can go to its [Doc page](https://github.com/EscVM/OIDv4_ToolKit#10-getting-started) and scapes as desires. Here, since we only demonstrate on Teapot dataset, we run this command:

```ruby
!python OIDv4_ToolKit/main.py downloader --classes Teapot --type_csv all --multiclasses 1
```

The `Teapot` after `--classes` is the classes of data we want, the name must aligns exactly with Open Images website labels. 

The `--multiclasses 1` is making sure that all the data collected will be in the same folder. It is redundant here, but convenient when you have more than 1 class.

After you run this command, a prompt will show up like this:
![image-center]({{ site.url }}{{ site.baseurl }}/assets/imgs/portfolios/OID.PNG){: .align-center}
<figcaption>OIDv4 Prompt</figcaption>

You need to input `Y` three times to keep the program forward, for training, testing and validation respectively.  

after that, the date will be downloaded to folder `OID/Dataset/`, run this command to download all dataset from Colab to local:

```ruby
from google.colab import files
!zip -r /content/file.zip /content/OID/Dataset/
files.download("/content/file.zip")
```

Then, you will have a `file.zip` download to you local setup. Inside, you will find each folder for train, test and validation, which contains the images and bounding box information (.txt) inside. 

Take on txt file for example:

```ruby
Teapot 361.516032 2.175744 770.8211200000001 358.4448
```

If you follows my previous post, you will recognize that this is YOLO format, where the first `Teapot` is the class name, and `361.516032 2.175744` is location of bounding box bottom left corner, `770.8211200000001 358.4448` is location of bounding box top right corner

# Data Formatting & Processing
Since YOLO need to recognize the data to proceed the training process, we have to convert our data and labels in specific PyTorch YOLOv5 format. The PyTorch YOLOv5 takes a `data.yml` file to locate train, test, validate data and labels, which also arranged in a specific format. Luckily, we don't have to worry about he nuance with the help of RoboFlow. 

- Go to RoboFlow 
- Create a free account
- Create a dataset following the instructions 
- Drag and drop all the images in all three folder, then the labels, you will find that RoboFlow does the bounding box label automatically for you 🤟

![image-center]({{ site.url }}{{ site.baseurl }}/assets/imgs/portfolios/ROBOflow.PNG){: .align-center}
<figcaption>RoboFlow Interface</figcaption>

- After uploading and labeling, press "Finish Upload" on top right corner, customize your train test split, usually we leave it with 70-20-10 as default. 
- Continue, then "Generate" on top right corner with a version name
- After the web does all its works, hit "show download code" 
- Copy the command there. That command is what all this struggle actually for. It will looks like this: `!curl -L "SOME-CHARACTERS" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip`

- Paste that to our colab and run, you will find train, test, valid folders downloaded for you, with `data.yml` of course

Now, we can train our models! Yeaaaaaa! 🍕

# Model Training
As usual, clone YOLOv5 to Colab and test our GPU. Don't forget to set Colab runtime type to GPU for it will be painful ☠. 
```ruby
import torch
from IPython.display import Image  # for displaying images
from utils.google_utils import gdrive_download  # for downloading models/datasets
print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
```

Then, navigate into `yolov5` folder and download all its models: 'yolov5x.pt', 'yolov5m.pt', 'yolov5s.pt' and 'yolov5l.pt'. Besides their difference in sizes, they also different in structure and performances. We have already talked about this in previous post.

```ruby
%cd yolov5/
!weights/download_weights.sh
```

Now we TRAIN!

```ruby
!python train.py --img 640 --batch 8 --epochs 30 --data ../data.yaml --weights yolov5s.pt --device 0 --cfg ./models/yolov5s.yaml
```

This command specify that we will train our model at image size 640px, 8 batches, 30 epochs, from data.yaml, pretrained with yolov5s.pt and its structure yolov5s.yaml. You can change the structure in yolov5s.yaml. Here, we highly recommend pretain with one of the models to achieve good result even in limited data and resources. You can set `--weight ''`, which randomize the weight initialization, but the result will be really poor. Trust me on this.

When training, it will looks like this:
![image-center]({{ site.url }}{{ site.baseurl }}/assets/imgs/portfolios/training_yolov5.gif){: .align-center}
<figcaption>Training Process</figcaption>

Where first you can see the structure of the model you are training, then the training by epochs.

After training, you will find you model saved in address at the prompt bottom. We will use the `best.pt` model to test our model

The model will be saved in runs/train/exp/weights/best.pt; the second time you train it, it will be saved at runs/train/exp2/weights/best.pt. The more you train, the more increments on `exp` holder name.
{: .notice--info}

# Model Prediction
At last, run this
```ruby
!python detect.py --weights runs/train/exp/weights/best.pt --conf 0.4 --source ../test/images/
```

This command predicts base on model `runs/train/exp/weights/best.pt`, taking images from source file `../test/images/`, and bounding box all predictions have confidence larger or equal to 40%

The prediction will be super fast as it is.

# Finally
Well, we are done, take a rest and have a cup of tea 🍵. You know where to find 😋

If you are interested in my projects or have any new ideas you wanna talk about, feel free to [contact](mailto:haoyanhy.jiang@mail.utoronto.ca) me!
{% capture fig_img %}
[![image-center](https://media.tenor.com/images/60a80f76872a66d9a98024ebc90576a0/tenor.gif)](https://media.tenor.com/images/60a80f76872a66d9a98024ebc90576a0/tenor.gif){: .align-center}
{% endcapture %}

<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>

A **BEER** would be perfect, but remember **NO CORONA!** 🍻 
<style>.bmc-button img{height: 34px !important;width: 35px !important;margin-bottom: 1px !important;box-shadow: none !important;border: none !important;vertical-align: middle !important;}.bmc-button{padding: 7px 15px 7px 10px !important;line-height: 35px !important;height:51px !important;text-decoration: none !important;display:inline-flex !important;color:#ffffff !important;background-color:#000000 !important;border-radius: 5px !important;border: 1px solid transparent !important;padding: 7px 15px 7px 10px !important;font-size: 20px !important;letter-spacing:-0.08px !important;box-shadow: 0px 1px 2px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;margin: 0 auto !important;font-family:'Lato', sans-serif !important;-webkit-box-sizing: border-box !important;box-sizing: border-box !important;}.bmc-button:hover, .bmc-button:active, .bmc-button:focus {-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;text-decoration: none !important;box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;opacity: 0.85 !important;color:#ffffff !important;}</style><link href="https://fonts.googleapis.com/css?family=Lato&subset=latin,latin-ext" rel="stylesheet"><a class="bmc-button" target="_blank" href="https://www.buymeacoffee.com/MaxJiang"><img src="https://cdn.buymeacoffee.com/buttons/bmc-new-btn-logo.svg" alt="Buy me a Beer"><span style="margin-left:5px;font-size:19px !important;">Buy me a Beer</span></a>
