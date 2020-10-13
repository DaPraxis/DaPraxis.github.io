---
title:  "Pytorch Study Note 1"
search: false
excerpt: 'Tensor Vector and Linear Regression'
categories: 
  - Pytorch
  - Max's Study Note
  - Machine Learning
last_modified_at: 2020-10-12 10:00
comments: true
toc: true
author_profile: true
toc_sticky: true
mathjax: true
header:
  teaser: https://analyticsindiamag.com/wp-content/uploads/2019/06/pytorch.png
---
# Prelude

*Pytorch* is one of the most basic framework machine learning engineers and researchers use these days to model and train and tune. I personally have used Pytorch for almost two years and stumbled upon all the pros and cons in PyTorch code as every researcher does. I think most people learn PyTorch like me, get insights from other people's work, and adapts to it just by rumbling through the PyTorch documentation. There hasn't really been a chance for me to slow down and  **LEARN** PyTorch. As I know for all the Machine Learning courses I have taken in the University of Toronto those 4-5 years, there is also no courses focusing on using PyTorch or TensorFlow, they are mostly on the theory, and give you the code and ask you to fill the blank 😪. Now, with the time to spend in this pandemic time, I want to take a deeper look at PyTorch, especially how to coordinate between syntaxes and APIs. All the API and knowledge listed below are updated until 2020 PyTorch 1.6.0

# Table of Content
* Tensor and its initializations
* Basic operations on tensors (tensor joins, splits, indexing, transformation, and mathematical operations)
* Linear Regression for demo
* Summary

# Tensor and its initializations
## 1. Tensor
> In mathematics, a tensor is an algebraic object that describes a (multilinear) relationship between sets of algebraic objects related to a vector space. Objects that tensors may map between include vectors and scalars, and even other tensors. 
    -Wikipedia

{% capture fig_img %}
[![image-center](https://res.cloudinary.com/practicaldev/image/fetch/s--oTgfo1EL--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://raw.githubusercontent.com/adhiraiyan/DeepLearningWithTF2.0/master/notebooks/figures/fig0201a.png)](scalar, vector, matrix, tensor){: .align-center}
{% endcapture %}

<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>

I think you are familiar with the concept of scalar, vector, and matrix, which is usually used to represent data of 1D, 2D, and 3D. Mostly, the Tensor is just another term to explain dimensional data that larger than 3D in Mathematics.  You can see it as a meta version of data in any dimension, and that is what it is in PyTorch -- a data container compatible for data of all dimensions -- even better, it contains the in-built functions for its calculations and attributes, which will be discussed later. 

## 2. Tensor & Variable
Variable is a data type under `torch.autograd`, and been categorized into Tensor already. The current tensor 'absorbs' autograd's neat property of handling derivatives very conveniently by taking in `torch.autograd.Variable` entirely. To understand Tensor thoroughly, we can take a look at its simpler version *Variable* first. 

* torch.autograd.Variable
    + data: data been encapsulated
    + grad: gradient of data
    + grad_fn: stands for gradient function, keep track of the methods/operation we used to create the tensor
    + required_grad: indicator of if taking derivative is needed
    + is_leaf: indicator if current node is a leaf node

noted that grad_fn is very important, which related to how we are going to derive the parameter in the computation graph. This will be elaborated in later chapter

Let's take a look at Tensor as well:

* torch.tensor
    + data: ...
    + dtype: tensor data type, e.g. torch.FloatTensor, torch.cuda.FloatTensor, data inside usually float32, int64
    + shape: keep track of tensor shape in tuple e.g. (m, n, i, j)
    + device: tensor device, e.g. CPU/GPU
    + required_grad: ...
    + grad: ...
    + grad_fn: ...
    + is_leaf: ...

Pretty similar huh?

## 3. Create a Tensor
### 1. Create directly from data
```ruby
torch.Tensor(data, 
            dtype=None,
            device=None,
            required_grad=False,
            pin_memory=False)
```
The variable `data` here can be any common python data types: list or numpy. `dtype` is the data type same to `data` by default. The `pin_memory` when set to True, it loads your samples in the Dataset on CPU and pushes it to the GPU to speed up the host with page-lock memory allocation. For now, we leave it as False as default.

> **NOTE**: when you create tensor from numpy array, the two data structure will alias in memory, i.e. change of content in numpy array will result changes in tensor and wise-versa

### 2. Create with filled numbers
+ We can create a tensor of only 0s inside
    ```ruby
    torch.zeros(*size,
                out=None,
                dtype=None,
                layout=torch.strided,
                device=None,
                required_grad=False)
    ```
    `layout` refers to layout pattern in memory, usually leave as default. `out` can take another variable to point to the same tensor location. 

    Similarly, `torch.ones()` create a tensor of only 1s, `torch.full()` create a tensor of any identical number.

+ Create a tensor of same shape as input, but filled with 0s, 1s or any other numbers
    - torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False)
    - torch.ones_like()
    - torch.full_like()

    The names are quite self-explanatory. 

+ arithmetic sequence
    works like `range()` in python,
    ```ruby
    torch.arrange(start=0,
                end,
                step=1,
                out=None,
                dtype=None,
                layout=torch.strided,
                device=None,
                required_grad=False)
    ```
    similarly, `torch.linspace()` also gives arithmetic sequence, but it works in closed interval [start, end], while `torch.arrange()` work in semi-closed interval [start, end). Just like `linspace()` in numpy

+ log sequence
    ```ruby
    torch.logspace(start,
                    end,
                    steps=100, 
                    base=10.0
                    out=None,
                    dtype=None,
                    layout=torch.strided,
                    device=None,
                    required_grad=False)
    ```
    `base` here is the base for logistic calculation

+ diagonal matrix/tensor
    ```ruby
    torch.eye(n,
            m=None,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            required_grad=False)
    ```
    `n` and `m` are the number of rows and columns for matrix respectively

### 3. Create with probabilities or distributions


