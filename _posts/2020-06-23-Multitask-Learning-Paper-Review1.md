---
title:  "Multitask-Learning Paper Overview 1"
search: false
excerpt: 'Multitask-Learning, Transfer Learning, Paper overview'
categories: 
  - Paper Review
  - Multitask-Learning
  - Neural Net
last_modified_at: 2020-06-23 16:00
comments: true
toc: true
mathjax: true
header:
  image: https://miro.medium.com/max/3840/1*_gg1Te-7SJfk9E2D-mORfw.png
  teaser: https://miro.medium.com/max/3840/1*_gg1Te-7SJfk9E2D-mORfw.png
---

## An Overview of Multi-Task Learning in Deep Neural Networks
> Reference: [Ruder S "An Overview of Multi-Task Learning in Deep Neural Networks", arXiv:1706.05098, Hune 2017](https://arxiv.org/abs/1706.05098)

[<i class="fas fa-link"></i>](https://arxiv.org/pdf/1706.05098.pdf) paper source

### MTL in Deep Neural Net:
#### Sharing types:
+ Hard Sharing 
{% capture fig_img %}
[![image-center](https://ruder.io/content/images/2017/05/mtl_images-001-2.png)](https://ruder.io/content/images/2017/05/mtl_images-001-2.png){: .align-center}
{% endcapture %}
<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>
Hard sharing is most commonly used in MTL, especially in MTL+neural nets. This is applied by sharing hidden layers between all tasks, to lower the risk of overfitting.

> "The more tasks we are learning simultaneously, the more our model has to find a representation that captures all of the tasks and the less is our chance of overfitting on our original task."

+ Soft Sharing
{% capture fig_img %}
[![image-center](https://ruder.io/content/images/size/w2000/2017/05/mtl_images-002-2.png)](https://ruder.io/content/images/size/w2000/2017/05/mtl_images-002-2.png){: .align-center}
{% endcapture %}
<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>
The soft sharing on the other hand, each task has their own model and parameters. The parameters between each task are encouraged to be similar with regulations and penalties. 

#### Underlying MTL Mechanisms:
+ Implicit Data Augumentation
    - increase sample size
    - different noise pattern among tasks smoothing the overall noise
+ Attention Focusing  
    - Focus on features that really matters
    - Providing relevence and irrlevence between features
+ Eavesdropping
    - Some task A interact with feature G better, which share information to task B to learn feature G better
+ Representation Bias
    - Helps with model generalization
+ Regularization:
    - Act as a "regularizer" by introducing an inductive bias -> reduce overfiting rate

#### Recent works MTL in Deep Learning
+ Deep Relationship Networks
> [Long, Mingsheng and Jianmin Wang. “Learning Multiple Tasks with Deep Relationship Networks.” ArXiv abs/1506.02117 (2015): n. pag.](https://www.semanticscholar.org/paper/Learning-Multiple-Tasks-with-Deep-Relationship-Long-Wang/7c61efd58584451b8988c42f2b7006eddbb291f1)

  [<i class="fas fa-link"></i>](https://arxiv.org/pdf/1506.02117.pdf) paper source

{% capture fig_img %}
[![image-center](https://d3i71xaburhd42.cloudfront.net/7c61efd58584451b8988c42f2b7006eddbb291f1/4-Figure1-1.png)](https://d3i71xaburhd42.cloudfront.net/7c61efd58584451b8988c42f2b7006eddbb291f1/4-Figure1-1.png){: .align-center}
{% endcapture %}
<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>

+ Fully-Adaptive Feature Sharing
>  [Lu, Y., Kumar, A., Zhai, S., Cheng, Y., Javidi, T., and Feris, R. (2016). Fullyadaptive Feature Sharing in Multi-Task Networks with Applications in Person Attribute Classification.](https://arxiv.org/abs/1611.05377)

  [<i class="fas fa-link"></i>](https://arxiv.org/pdf/1611.05377.pdf) paper source

{% capture fig_img %}
[![image-center](https://ruder.io/content/images/2017/05/fully_adaptive_feature_sharing.png)](https://ruder.io/content/images/2017/05/fully_adaptive_feature_sharing.png){: .align-center}
{% endcapture %}
<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>


+ Cross-stitch Networks
>  [Misra, I., Shrivastava, A., Gupta, A., and Hebert, M. (2016). Cross-stitch Networks for Multi-task Learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.](https://arxiv.org/abs/1604.03539)

  [<i class="fas fa-link"></i>](https://arxiv.org/pdf/1604.03539.pdf) paper source

{% capture fig_img %}
[![image-center](https://ruder.io/content/images/2017/05/cross-stitch_networks.png)](https://ruder.io/content/images/2017/05/cross-stitch_networks.png){: .align-center}
{% endcapture %}
<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>

+ Low Supervision
>  [Søgaard, A. and Goldberg, Y. (2016). Deep multi-task learning with low level tasks supervised at lower layers. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, pages 231–235.](https://www.aclweb.org/anthology/P16-2038/)

  [<i class="fas fa-link"></i>](https://www.aclweb.org/anthology/P16-2038.pdf) paper source

+ A Joint Many Task Model
> [Hashimoto, K., Xiong, C., Tsuruoka, Y., and Socher, R. (2016). A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks.](https://arxiv.org/abs/1611.01587)

  [<i class="fas fa-link"></i>](https://arxiv.org/pdf/1611.01587.pdf) paper source

{% capture fig_img %}
[![image-center](https://media.arxiv-vanity.com/render-output/3005732/x1.png)](https://media.arxiv-vanity.com/render-output/3005732/x1.png){: .align-center}
{% endcapture %}
<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>

+ Weighting Losses with Uncertainty
>  [Kendall, A., Gal, Y., and Cipolla, R. (2017). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.](https://arxiv.org/abs/1705.07115)

  [<i class="fas fa-link"></i>](https://arxiv.org/pdf/1705.07115.pdf) paper source

{% capture fig_img %}
[![image-center](https://ruder.io/content/images/2017/05/weighting_using_uncertainty.png)](https://ruder.io/content/images/2017/05/weighting_using_uncertainty.png){: .align-center}
{% endcapture %}
<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>

+ Tensor Factorization for MTL
>  [Yang, Y. and Hospedales, T. (2017a). Deep Multi-task Representation Learning: A Tensor Factorisation Approach. In Proceedings of ICLR 2017](https://arxiv.org/abs/1605.06391)

  [<i class="fas fa-link"></i>](https://arxiv.org/pdf/1605.06391.pdf) paper source

{% capture fig_img %}
[![image-center](https://d3i71xaburhd42.cloudfront.net/468a80bcd4ff9b3f47beb9145ff81140777bb3f3/6-Figure1-1.png)](https://d3i71xaburhd42.cloudfront.net/468a80bcd4ff9b3f47beb9145ff81140777bb3f3/6-Figure1-1.png){: .align-center}
{% endcapture %}
<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>

+ Sluice Networks
>  [Ruder, S., Bingel, J., Augenstein, I., and Søgaard, A. (2017). Sluice networks: Learning what to share between loosely related tasks.](https://www.semanticscholar.org/paper/Sluice-networks%3A-Learning-what-to-share-between-Ruder-Bingel/e242ba1a62eb2595d89afbec2657f33d9ab4abe3)

  [<i class="fas fa-link"></i>](https://arxiv.org/pdf/1705.08142.pdf) paper source

{% capture fig_img %}
[![image-center](https://d3i71xaburhd42.cloudfront.net/e242ba1a62eb2595d89afbec2657f33d9ab4abe3/3-Figure1-1.png)](https://d3i71xaburhd42.cloudfront.net/e242ba1a62eb2595d89afbec2657f33d9ab4abe3/3-Figure1-1.png){: .align-center}
{% endcapture %}
<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>

### MTL in non-neural models
+ Block-sparse Regularization: 
    - Assumes: tasks used in multi-task learning are closely related
    - enforce Lasso (L1 norm) to 0 out features
    - Block-sparse regularization: $$l_1/l_q$$ norms
+ Learning task relationships:
    - No assumption between tasks
    - A constraint that enforces a clustering of tasks: penalizing both the norms of our task column vectors $$a_{·,1},\dots, a_{·,t}$$ as well as their variance:
    > $$\Omega = ||\bar{a}||^2+\frac{\lambda}{T}\sum^{T}_{t=1}||a_{·,t}-\bar{a}||^2$$
    where $$\bar{a}=(\sum_{t=1}^{T}a_{.t})/T$$ is the mean parameter vector. The penalty on the other hand forces all parameter $$a_{.t}$$ to their mean $$\bar{a}$$

### Auxiliary Tasks
> MTL is a natural fit in situations where we are interested in obtaining predictions for multiple tasks at once. Such scenarios are common for instance in **finance** or **economics forecasting**, where we might want to predict the value of many possibly related indicators, or in **bioinformatics** where we might want to predict symptoms for multiple diseases simultaneously. 

> In scenarios such as **drug discovery**, where tens or hundreds of active compounds should be predicted, MTL accuracy increases
continuously with the number of tasks

+ Related task
  - Using a **related task** as an auxiliary task for MTL
    - uses tasks that predict different characteristics of the road as auxiliary tasks for predicting the steering direction in a self-driving car;
    -  use head pose estimation and facial attribute inference as auxiliary tasks for facial landmark detection;
    - jointly learn query classification and web search;
    - jointly predicts the class and the coordinates of an object in an image;
    - jointly predict the phoneme duration and frequency profile for [text-to-speech.](http://proceedings.mlr.press/v70/arik17a.html)

+ Adversarial
  > Often, labeled data for a related task is unavailable. In some circumstances, however, we have access to a task that is opposite of what we want to achieve.
  - **maximize the training error** using a [*gradient reversal layer*](http://proceedings.mlr.press/v37/ganin15.html). 
+ Hints
  > learn features that might not be easy to learn just using
the original task
+ Focusing attention
  >  focus attention on parts of the **image** that a network
might normally ignore. 
+ Quantization smoothing
+ Predicting inputs
+ Using the future to predict the present
+ Representation learning

