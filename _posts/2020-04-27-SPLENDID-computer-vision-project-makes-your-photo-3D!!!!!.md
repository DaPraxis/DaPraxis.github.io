---
title:  "SPLENDID computer vision project makes your photo 3D!!!!!"
search: false
excerpt: 'Open Source Computer Vision project, image depth-field animation'
categories: 
  - Computer Vision
  - 3D Animation
  - Python
  - Deep Learning
last_modified_at: 2020-06-21 16:00
comments: true
toc: true
---
> [<i class="fas fa-link"></i>](https://colab.research.google.com/drive/1706ToQrkIZshRSJSHvZ1RuCiM__YX3Bz) Code Source in Colab 

# Your photo speaks to you!
<figure>
	<a href="https://miro.medium.com/max/1400/1*wibtMkDPQLqkCGEQ2ekheQ.jpeg"><img src="https://miro.medium.com/max/1400/1*wibtMkDPQLqkCGEQ2ekheQ.jpeg"></a>
	<figcaption>Photo by Author, on the road to Banff, 2018 Nov</figcaption>
</figure>

<figure>
	<a href="https://miro.medium.com/max/960/1*Inzb63-wcDzxmqYTgb2vWg.gif"><img src="https://miro.medium.com/max/960/1*Inzb63-wcDzxmqYTgb2vWg.gif"></a>
	<figcaption>Animated</figcaption>
</figure>

<figure>
	<a href="https://miro.medium.com/max/1400/1*TEugkgXWy4uZpgfzpaIebQ.jpeg"><img src="https://miro.medium.com/max/1400/1*TEugkgXWy4uZpgfzpaIebQ.jpeg"></a>
	<figcaption>Photo by Author, Yoho National Park in Banff, 2018 Nov</figcaption>
</figure>

<figure>
	<a href="https://miro.medium.com/max/1400/1*TEugkgXWy4uZpgfzpaIebQ.jpeg"><img src="https://miro.medium.com/max/960/1*wKuUmoXjE4cD9znfrSbQuA.gif"></a>
	<figcaption>Animated</figcaption>
</figure>

<figure>
	<a href="https://miro.medium.com/max/1400/1*eVbdrKzrSpJxxkbwqdhQHA.jpeg"><img src="https://miro.medium.com/max/1400/1*eVbdrKzrSpJxxkbwqdhQHA.jpeg"></a>
	<figcaption>Photo by Author, My partner in Montreal, 2019 Aug</figcaption>
</figure>

{% capture fig_img %}
[![image-center](https://miro.medium.com/max/640/1*M0HrxIT2L1pZcZ6px47rAg.gif)](https://miro.medium.com/max/640/1*M0HrxIT2L1pZcZ6px47rAg.gif)
{% endcapture %}

{% capture fig_caption %}
Image with a caption.
{% endcapture %}


<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>{{ fig_caption | markdownify | remove: "<p>" | remove: "</p>" }}</figcaption>
</figure>

<!-- <figure>
	<a href="https://miro.medium.com/max/640/1*M0HrxIT2L1pZcZ6px47rAg.gif"><img src="https://miro.medium.com/max/640/1*M0HrxIT2L1pZcZ6px47rAg.gif"></a>
	<figcaption>Animated</figcaption>
</figure> -->

----

Just AMAZING right? I was as astonished as you guys. This fantastic [project](https://shihmengli.github.io/3D-Photo-Inpainting/) was made by four great researchers [Meng-Li Shih](https://shihmengli.github.io/), [Shih-Yang Su](https://lemonatsu.github.io/), [Johannes Kopf](https://johanneskopf.de/), and [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/) in [IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020](https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/3DPhoto_paper.pdf).

Want to try it out on your photos as well? It is ok if you completely have no ideas of computer vision and CNN, just follow my step in setup section below to run all the code in the block in this [**link**](https://colab.research.google.com/drive/1706ToQrkIZshRSJSHvZ1RuCiM__YX3Bz)! I recommend setting up in Colab, since it take a certain amount of your computer resource to train, and Colab cached for you.

----

## Setup:
<figure>
	<a href="https://miro.medium.com/max/1400/1*6FqIB-OFSbxN1Gmbk-g3IQ.png"><img src="https://miro.medium.com/max/1400/1*6FqIB-OFSbxN1Gmbk-g3IQ.png"></a>
	<figcaption>Import image code block</figcaption>
</figure>

<figure>
	<a href="https://miro.medium.com/max/842/1*hpyCVm-BAYLkQxrH_OJ47Q.png"><img src="https://miro.medium.com/max/842/1*hpyCVm-BAYLkQxrH_OJ47Q.png"></a>
	<figcaption>Import image code block</figcaption>
</figure>

1. Run all the code before this code block, and drag all the photos you want to make 3D in the highlighted image folder, then run the code block to import your uploaded images.

2. Then simply run the last code block below:
```yml
!python main.py --config argument.yml
```
and you will need to wait 2–5 minutes for each training batch, depending on your computer specs, and picture attributes

3. BOOM! You get the RESULT!
<figure>
	<a href="https://miro.medium.com/max/626/1*eamT3dhwsrU69I1rSVq2bQ.png"><img src="https://miro.medium.com/max/626/1*eamT3dhwsrU69I1rSVq2bQ.png"></a>
	<figcaption>Import image code block</figcaption>
</figure>

You can find your results in the indicated area. It will output five output visuals, which include depth map estimated by [MiDaS](https://github.com/intel-isl/MiDaS), inpainted 3D mesh, and you 3D video demo in circle, swing, and zoom in motions. Simple enough huh? Keep reading if you want to know the logic behind it!

----

## Theory:

How does a machine predict that 3D view just from a 2D photo? I mean, for each object in the photo, if you want to “see” what is behind it, you have to somehow imagine it as a human. When people see a photo, they not only just see it as a static image, but also perceive it as a 3D object that is alive, and even makes up an imaginary scene or recalls some memories. But how does a machine deal with such a complex concept? Can it “imagine”??

>Well, a machine cannot imagine, but it can “learn” to “imagine”, or in other words, it can treat data and output in a way like a human. Basically, machine just does what they excel at: calculations.

<figure>
	<a href="https://miro.medium.com/max/1400/1*4aONSQV2oQGBJ8PeYPUJww.png"><img src="https://miro.medium.com/max/1400/1*4aONSQV2oQGBJ8PeYPUJww.png"></a>
	<figcaption><a href="https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/3DPhoto_paper.pdf" title="Pic from 3D Photography using Context-aware Layered Depth Inpainting">Pic from 3D Photography using Context-aware Layered Depth Inpainting</a>.</figcaption>
</figure>

Normally for AI learn RGB-D image, where the D represents ‘depth’, to relive 3D effects. Most smartphones in the market have two cameras right now to capture color and depth in view separately. However, what about normal RGB pictures without depth? Machine predicts! With some standard image preprocessing steps, we can find the depth plot easily (a to d)

With the predicted depth, machine can find where the depth discontinuities are, then categorize then and group to different color sections (e to f).
With all the preprocessing preparation, we are going to repair the 3D vision from our 2D photo. The most essential tool we are using is called **Layered Depth Image (LDI)**

<figure>
	<a href="https://miro.medium.com/max/1400/1*r8zj8T28i84YLyBiSMjthA.png"><img src="https://miro.medium.com/max/1400/1*r8zj8T28i84YLyBiSMjthA.png"></a>
	<figcaption><a href="https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/3DPhoto_paper.pdf" title="Pic from 3D Photography using Context-aware Layered Depth Inpainting">Pic from 3D Photography using Context-aware Layered Depth Inpainting</a>.</figcaption>
</figure>

On the edges, the pixel by two sides are connected by a sharp drop (a). The program first cut the drop connection into green and red regions (b), we call them foreground silhouette and background silhouette, spawn a synthesis region based on background silhouette, or context region (c), then merge into the model.

<figure>
	<a href="https://miro.medium.com/max/834/1*nIJ-e5XUR_Pi4oIGHQaBYA.png"><img src="https://miro.medium.com/max/834/1*nIJ-e5XUR_Pi4oIGHQaBYA.png"></a>
	<figcaption><a href="https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/3DPhoto_paper.pdf" title="Pic from 3D Photography using Context-aware Layered Depth Inpainting">Pic from 3D Photography using Context-aware Layered Depth Inpainting</a>.</figcaption>
</figure>

Now, since we have two regions separated already (context region and synthesis region), scientists use three repairing agents to complete the inpainting task: edge inpainting network, color inpainting network, and depth inpainting network. You can check out the resources below to see how those inpainting networks work in detail.

The edge inpainting network repairs the contour between context region and synthesis region, to predict the blocked edges. Then the machine uses color inpainting network and depth inpainting network to imagine the blocked color and depth respectively. After this, we feed the result back to the LDI model, and Bingo! We have the result!

>*Without further due, play with the model and relive your memories!*

<figure>
	<a href="https://miro.medium.com/max/960/1*Os_xXoFH6VyKZcJ9FJjbqQ.gif"><img src="https://miro.medium.com/max/960/1*Os_xXoFH6VyKZcJ9FJjbqQ.gif"></a>
	<figcaption><a href="https://pixabay.com/photos/toy-toy-story-childhood-little-2207781/" title="Photo by Coyot from Pixabay, processed by author">Photo by Coyot from Pixabay, processed by author</a>.</figcaption>
</figure>

## Reference:

### Relevant projects:
* [MiDaS](https://github.com/intel-isl/MiDaS)
* [StereoConvNet](https://github.com/LouisFoucard/StereoConvNet)

### Paper and Reference:

* [3D Photography using Context-aware Layered Depth Inpainting](https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/3DPhoto_paper.pdf) Paper
* [3D Photography using Context-aware Layered Depth Inpainting](https://shihmengli.github.io/3D-Photo-Inpainting/) Website
* [Edge inpainting network](https://arxiv.org/pdf/1901.00212.pdf) Paper
* [Color inpainting network](https://arxiv.org/abs/1804.07723) Paper
* [Depth inpainting network](https://arxiv.org/abs/1901.05945)  Paper
