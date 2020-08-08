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
# mathjax: true
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
---
# Project Show Case
{% include gallery caption="Distance Measuring with Euclidean(left) & Bird-eye view(right)" %}

<!-- {% capture fig_img %}
[![image-center](https://i.imgur.com/cS7Fqci.gif)](https://i.imgur.com/cS7Fqci.gif){: .align-center}
{% endcapture %} -->

<!-- <figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure> -->

# Project Discription
This article continues the ["Social Distance Detection Application - Phase 1"](https://dapraxis.github.io/portfolio/Social-Distance-Detection/) topic. In this post, we will talk about *object detection models* in depth, performace for *YOLOv5 and Detectron2 in Colab* default GPU and the *bird eye view conversion* improves social distance measurement

Mask ON 😷. 


# Contact Me
A **BEER** would be perfect, but remember **NO CORONA!** 🍻 
<style>.bmc-button img{height: 34px !important;width: 35px !important;margin-bottom: 1px !important;box-shadow: none !important;border: none !important;vertical-align: middle !important;}.bmc-button{padding: 7px 15px 7px 10px !important;line-height: 35px !important;height:51px !important;text-decoration: none !important;display:inline-flex !important;color:#ffffff !important;background-color:#000000 !important;border-radius: 5px !important;border: 1px solid transparent !important;padding: 7px 15px 7px 10px !important;font-size: 20px !important;letter-spacing:-0.08px !important;box-shadow: 0px 1px 2px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;margin: 0 auto !important;font-family:'Lato', sans-serif !important;-webkit-box-sizing: border-box !important;box-sizing: border-box !important;}.bmc-button:hover, .bmc-button:active, .bmc-button:focus {-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;text-decoration: none !important;box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;opacity: 0.85 !important;color:#ffffff !important;}</style><link href="https://fonts.googleapis.com/css?family=Lato&subset=latin,latin-ext" rel="stylesheet"><a class="bmc-button" target="_blank" href="https://www.buymeacoffee.com/MaxJiang"><img src="https://cdn.buymeacoffee.com/buttons/bmc-new-btn-logo.svg" alt="Buy me a Beer"><span style="margin-left:5px;font-size:19px !important;">Buy me a Beer</span></a>