---
title:  "Pytorch Study Note 2"
search: false
excerpt: 'Dynamic Computation Graph, Auto Gradient and Logistic Regression'
categories: 
  - Pytorch
  - Max's Study Note
  - Machine Learning
last_modified_at: 2020-10-26 10:00
comments: true
toc: true
author_profile: true
toc_sticky: true
mathjax: true
header:
  teaser: https://analyticsindiamag.com/wp-content/uploads/2019/06/pytorch.png
  image: https://images.unsplash.com/photo-1476966502122-c26b7830def9?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1072&q=80
---

# Prelude

Last article we talked about basic Tensor unit in PyTorch, the manipulation about it and a light tough on the implementation of linear regression in PyTorch architecture. If you missed the last section, you can review [here](https://dapraxis.github.io/pytorch/max's%20study%20note/machine%20learning/Pytorch-Study-Note(1)/). As the teaser image suggests, **IT IS MATH TIME!** Nothing really hard and theoretical, but some basic differentiation techniques will suffice. In this article, we will discuss some very important logics behinds PyTorch that makes it unique and become the favorite of many -- the dynamic computation graph and auto gradient. Also, a demo of how to setup logistic regression in PyTorch in practice.

# Table of Content
- Computation Graph
- Auto Gradient
- Logistic Regression

# Computation Graph
Previously, we have talked about tensors, and in machine learning, tensors operations will are error-prone and complicated to work with. Especially when you have a large deep neural net and each layer is a high-dimensional tensor, the interaction between them can be troublesome. 

Imagine you have a neural net need for forward propagation, you pass in the value can calculate the value propagation in a chain, you need to link them in a particular way such that the system can automate the process and output your current error; then when you back propagate the gradients and update weights, you reverse the direction of your forward propagation. This is like a ripple, which you hit with forward pass, and feedback with backward propagation. The path forward propagation and backward propagation is finite, just differ in terms of direction. 

{% capture fig_img %}
[![image-center](https://miro.medium.com/max/6216/1*6q2Rgd8W9DoCN9Wfwc_9gw.png)](Forward and Backward Propagation){: .align-center}
{% endcapture %}

<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>

Computation graph, in this case, is a path, a graph, that clarify and simplify the logic between tensors and layers, and automate the data propagations in between. 

## From an example
Let's build a computation graph for 

$$y=(x+w)\times (w+1)$$

We do it this way:

$$a = x+w$$

$$b = w+1 $$

Then

$$y = a \times b$$

And the graph simply looks like this:
![image-center]({{ site.url }}{{ site.baseurl }}/assets/imgs/posts/study_note/comp_graph1.png){: .align-center}
<figcaption>Computation graph for y=(x+w)*(w+1)</figcaption>

Then let's do back propagation to derive $$\frac{\partial y}{\partial w}$$

By mathematic derivations, we have:

$$ \begin{align*}
  \frac{\partial y}{\partial w} &= \frac{\partial y}{\partial a} \times \frac{\partial a}{\partial w} + \frac{\partial y}{\partial b} \times \frac{\partial b}{\partial w} \\\\
  & = b \times 1 + a \times 1 \\\\
  & = (w+1) + (w + x) \\\\
  & = 2w + x + 1 \\\\
\end{align*}
$$

With w = 1, x = 2, we have $$y = (1+1)\times (1+2) = 6$$, and $$\frac{\partial y}{\partial w} = 2\times 1+1+1 = 5$$

The derivative of y with respect to w, is simply finding all the path from y to w in the graph, and then sum them together. Indicated in red below:
![image-center]({{ site.url }}{{ site.baseurl }}/assets/imgs/posts/study_note/comp_graph2.png){: .align-center}
<figcaption>Backward Pass Computation graph for y=(x+w)*(w+1)</figcaption>

Let's verify this with code as well:
```ruby
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
print(w.grad)   # tensor([5.])
```
## Properties
Now we know what a computation graph is, and we can carry on talk about its properties in PyTorch:

- `is_leaf`:

  Remember there is a parameter in `Torch.Tensor` called `is_leaf`? You can recap [here](https://dapraxis.github.io/pytorch/max's%20study%20note/machine%20learning/Pytorch-Study-Note(1)/#2-tensor--variable). Leaf nodes are basically all the nodes that created by user, i.e. `w` and `x` in our example. This is very important, since all the propagations are based on our leaf nodes, and `is_leaf` is an indicator telling weather it is a leaf node

  We can check by this code:

  ```ruby
  print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
  # is_leaf:
  # True True False False False
  ```

  But why leaf node? I mean why we especially separate those nodes that created by user and those are not? Well PyTorch do this to **save memory**. Normally, only gradients of leaf nodes are accessed by users. PyTorch is programed especially to save only the leaf node gradients after propagation, but release the memory of non-leaf node gradients. 

  Let's check:

  ```ruby
  print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)
  # gradient:
  # tensor([5.]) tensor([2.]) None None None
  ```

  Well, if you really need to access the gradients in non-leaf gradients, say debugging purpose, you can use `retain_grad()` to do so:

  ```ruby
  w = torch.tensor([1.], requires_grad=True)
  x = torch.tensor([2.], requires_grad=True)

  a = torch.add(w, x)
  # we want to keep the gradient in a
  a.retain_grad()
  b = torch.add(w, 1)
  y = torch.mul(a, b)

  y.backward()

  print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

  ## gradient of a is preserved
  # gradient:
  # tensor([5.]) tensor([2.]) tensor([2.]) None None
  ```

- `grad_fn`

  Record how the tensor is calculated i.e. the function to derive the result. 

  ```ruby
  print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)

  #None None <AddBackward0 object at 0x0000029AECF56D08> <AddBackward0 object at 0x0000029AEEFEB248> <MulBackward0 object at 0x0000029AEEFEB748>
  ```

  You can see that a and b is acquired by summation, and y is acquired by multiplication. This is important for back propagation

## Dynamic v.s. Static
Why is PyTorch called **'Dynamic'** computation graph, while TensorFlow is **'Static'**?

The difference is: 
- in TensorFlow, we build *graph first*, then compute or propagate the value. This is very efficient, but hard for tuning
- in PyTorch, graph and computation *went together*. This is easing the process of graph changing and tuning

Let's see what it looks like in above scenario:

```ruby
# Declare two constants
w = tf.constant(1.)
x = tf.constant(2.)

# build graph
a = tf.add(w, x)
b = tf.add(w, 1)
y = tf.multiply(a, b)

# Propagation not started yet, y is just a node
print(y)   
# Tensor("Mul_4:0", shape=(), dtype=float32)

with tf.Session() as sess:
    # computation happens in the session  
    print(sess.run(y))  
    # 6.0
```

You can see that in TensorFlow, you have to build the graph structure first, then feed in the value to make it flows, while in PyTorch, again:
```ruby
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)
print(y)    # tensor([6.], grad_fn=<MulBackward0>)
```
computation takes place once the graph is made immediately

# Auto Gradient

