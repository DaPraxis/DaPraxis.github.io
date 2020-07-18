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
gallery2:
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
gallery3:
  - url: ../assets/imgs/posts/CNN/horse1.png
    image_path: ../assets/imgs/posts/CNN/horse1.png
    alt: "Visual 3"
  - url: ../assets/imgs/posts/CNN/horse2.png
    image_path: ../assets/imgs/posts/CNN/horse2.png
    alt: "Visual graph"
  - url: ../assets/imgs/posts/CNN/horse3.png
    image_path: ../assets/imgs/posts/CNN/horse3.png
    alt: "Visual 4"
  - url: ../assets/imgs/posts/CNN/horse4.png
    image_path: ../assets/imgs/posts/CNN/horse4.png
    alt: "Visual 5"
  - url: ../assets/imgs/posts/CNN/horse5.png
    image_path: ../assets/imgs/posts/CNN/horse5.png
    alt: "Visual 6"
  - url: ../assets/imgs/posts/CNN/horse6.png
    image_path: ../assets/imgs/posts/CNN/horse6.png
    alt: "Visual 6"
  - url: ../assets/imgs/posts/CNN/horse7.png
    image_path: ../assets/imgs/posts/CNN/horse7.png
    alt: "Visual 5"
  - url: ../assets/imgs/posts/CNN/horse8.png
    image_path: ../assets/imgs/posts/CNN/horse8.png
    alt: "Visual 6"
  - url: ../assets/imgs/posts/CNN/horse9.png
    image_path: ../assets/imgs/posts/CNN/horse9.png
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

## Starter Code
### Helper Functions
```ruby
import os
from six.moves.urllib.request import urlretrieve
import tarfile
import numpy as np
import pickle
import sys
from PIL import Image


def get_file(fname,
             origin,
             untar=False,
             extract=False,
             archive_format='auto',
             cache_dir='data'):
    datadir = os.path.join(cache_dir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)
    
    print('File path: %s' % fpath)
    if not os.path.exists(fpath):
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if untar:
        if not os.path.exists(untar_fpath):
            print('Extracting file.')
            with tarfile.open(fpath) as archive:
                archive.extractall(datadir)
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = pickle.load(f)
    else:
        d = pickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def load_cifar10(transpose=False):
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if transpose:
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    return (x_train, y_train), (x_test, y_test)
```
### Load Data
The below code may take a few minutes to load the data
```ruby
# Download cluster centers for k-means over colours
colours_fpath = get_file(fname='colours', 
                         origin='http://www.cs.toronto.edu/~jba/kmeans_colour_a2.tar.gz', 
                         untar=True)
# Download CIFAR dataset
m = load_cifar10()
(x_train, y_train), (x_test, y_test) = m

# 7 is horse category
indices = [i for i, x in enumerate(y_train) if x == [7]]

# visualize horse data in grey scale
im = Image.fromarray(x_train[indices[0]][0])
im.save('test1.png')
```
{% include gallery id="gallery3" caption="Visualize 9 loaded horses" %}
