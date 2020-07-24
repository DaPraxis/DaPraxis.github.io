---
title:  "State-of-the-Art Edge Computing"
search: false
excerpt: 'Current Edge Computing Achievements and Future Directions'
categories: 
  - Edge Computing
  - Machine Learning
  - IoT
last_modified_at: 2020-07-19 10:00
comments: true
toc: true
toc_sticky: true
mathjax: true
header:
  image: https://miro.medium.com/max/2600/1*RTGHo8x278rzhj2cZSjwtA.gif
  teaser: https://images.unsplash.com/photo-1592438710388-bf464c011df9?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80
---
# Introduction
Some notes and summaries I had when learned about concepts in Edge Computing, a broad review of the field in general.

This blog will first talk about the *emerging* of edge computing in the history, then briefly go through its *development* in recent years. The main focus will lay on its *applications* and *difficulties* faced by researchers, so that we can have a broader idea in terms of problem solving, especially combined with insights in *Machine Learning*. 

# Edge Computing & Cloud Computing
Most people will be familiar with the concept *Cloud Computing* right? We have Google Cloud, AWS, Azure all in public, and most developer or engineering roles right now requires cloud background as well. Then what on earth is *Edge Computing*? 

Let's first look at some official definitions:

> **Edge computing**(EC) is a distributed computing paradigm that brings computation and data storage closer to the location where it is needed, to improve response times and save bandwidth.

> **Cloud computing**(CC) is the on-demand availability of computer system resources, especially data storage (cloud storage) and computing power, without direct active management by the user. 

Some terminologies here: 
+ **Edge** refers to 'edge devices', they are the leaf node for information collection, also somtimes refered as 'edge node'
+ **Data center** refers to the actural stationary data collection facility that run by Google or Amazon, where all the data storage, data distribution and calculation happens. Can be treated as a *SUPER* computer that does all for you

In short, CC collect all the resources from edges nodes and just give back to data center. It let all data center to do the work; while EC tries to share the burden from the data centers and distibute to each edge node, such that instead of feedback the whole data chunk, edges only feeds analytical results back to data center.

{% capture fig_img %}
[![image-center](https://media0.giphy.com/media/3oKIPpFhwsMNrRIjN6/giphy.gif)](https://media0.giphy.com/media/3oKIPpFhwsMNrRIjN6/giphy.gif){: .align-center}
{% endcapture %}

<figure>
  {{ fig_img | markdownify | remove: "<p>" | remove: "</p>" }}
</figure>
From the definition, some disadvantages of traditional Cloud Computing have already surface:

1. Instananeity:
  - In the real-time senario of IoT, data feedback and traversal is heavily dependent on the Internet speed. 

2. Internet Bandwidth:
  - CC transfers all the data back to data center, which creates very large pressure to the Internet itself.

3. Huge Energy Comsumption:
  - CC groups all data together and process locally in data center, while EC distribute the computation to edge and alleviate the burdon in data centers

4. Privacy:
  - When CC groups all data together, this colossal source of data is proned to be attacked, since the uploaded data containes raw information of users. While EC preprocess or encrypts the data to be anonymous and insensitive so that it is less likely to leak user information

This bring us to the three main advantages EC has to rescue:
1. Process data on edges:
  - Process data on the edge side will relieve the limitations coming from internet bandwidth and reduce the data center energy consumption
  - e.g. Boeing 787 creates more than 5GB/s data, while the connection between satellite and plane is inefficient for the in-time communication
  

2. Process data close to edges:
  - If we process data in a node that close to edges, rather than data center directly, the Internet delay will be shorten based on closer communication distance
  - For example, a auto-driving car in Cloud Computing senario has to first send data back to data center, then request for data processing. It can only carry on after feedbacks from the data center. The result could be catastrophic if a unexpected internet cutoff happens. 
  - In contrast, Edge Computing allows edges(the car) or a local node close to the edges to process the data itself and only send process result to the data center, this saves request time and more robust to the environment. 

3. Privacy:
  - Private data will no longer be uploaded to data center directly, rather stay in edge or intermediate nodes

# Path of Developments & Milestones
<figure>
	<a href="https://blog.bosch-si.com/wp-content/uploads/Edge-cloud-history-1136x730.jpg"><img src="https://blog.bosch-si.com/wp-content/uploads/Edge-cloud-history-1136x730.jpg"></a>
	<figcaption>Cloud and edge computing in IoT: a short history
</figcaption>
</figure>
##  Some special timeline to mark on:

+ ### 2012: Mobile Edge Computing(MEC), Cloud Computing & Cloud-Sea
  - MEC
    > Multi-access edge computing (MEC), formerly mobile edge computing, is an ETSI-defined network architecture concept that enables cloud computing capabilities and an IT service environment at the edge of the cellular network and, more in general at the edge of any network.

    This technology establishes edge servers between edges and cloud to achieve lower delay and higher bandwidth. It is also the first time academia assume edge device has the capacity to compute and process data. This is pretty much the foundamentals and standard for all smart phones
  - Fog Computing and Cloud-Sea from Cisco
    > Fog computing or fog networking, also known as fogging,is an architecture that uses edge devices to carry out a substantial amount of computation, storage, and communication locally and routed over the internet backbone.

    The world **Fog** is a metaphor indicates that it is closere to gound/edge; **Sea** referres to the physical world we are standing upon
    
    Similar to MEC, Fog Computing inherites the MEC's edge server idea, while more focused on the resouce distribution between edges and services. It abstracts the architecture to a *high-level virtual computation platform*. It is also the first time talking about cooperations(or cooperative computing architecture): *Edge-to-Edge Cooperation*, *Cloud-Edge Cooperation*, *Edge-Terminal Cooperation* and *Cloud-Edge-Terminal Cooperation*. We will further talk about those concepts later

  **In general, Edge Computing is the foundation, while Fog Computing is the standard**

+ ### 2018: Kubernates
  - Kubernates for Edge Computing

    Kubernates sets up standard computation method and development environment for Edge Computing, brings the idea of container images, similar to Docker images, to the real-life Edge Computing production. This marks the development of Edge Computing is going into a phase of **steady development**

# Edge Computing and Machine Learning
<figure>
	<a href="https://images.unsplash.com/photo-1531746790731-6c087fecd65a?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=995&q=80"><img src="https://images.unsplash.com/photo-1531746790731-6c087fecd65a?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=995&q=80"></a>
	<figcaption>IoT, Machine Learning and Robot?
</figcaption>
</figure>
Well,

> Machine Learning + Edge Computing = Edge Intelligence

## Some Constraints in Edge Intelligence
+ Edge Computing Resource Supply & Demand
  - Simply as the title suggests, your smart phone is not "smart" enough to run machine learning algorithms. Some models even take four GPUs to run simultaneously, which will take forever to run on your edge(or just blow up, who knows)
  - Current open source machine learning packages such as Pytorch, MXNet, TensorFlow cannot run on edges perfectly. 

+ Data Diversity v.s. Privacy Protection
  - For machine learning and data mining, data diversity is always 'the more the marrier'. However, more diversed data fields sent to data center would more proned to expose user privacy to public

+ Monatone Edge Device Functionality v.s. Intelligent Service Complicity
  - Edge device sometimes can only perform simple tasks or functionalities,, while more diversed services such as data preperation, data cleaning, data selection and model learning are required in the field. The hardware can not satisfy all the tasks

## THE SOLUTION: Cooperation Computing
<figure>
	<a href="https://sloanreview.mit.edu/wp-content/uploads/2017/08/MAG-FR-Kiron-Collaboration-1200.jpg"><img src="https://sloanreview.mit.edu/wp-content/uploads/2017/08/MAG-FR-Kiron-Collaboration-1200.jpg"></a>
	<figcaption>Cooperation & Compute
</figcaption>
</figure>

### Cooperation Types:
+ Edge-Cloud Cooperation
  - Train-and-Predict
  - Cloud Focused Edge-Cloud Cooperation
  - Edge Focused Edge-Cloud Cooperation
+ Edge-Edge Cooperation
  - Edge-Edge Prediction Cooperation
  - Edge-Edge Distributive Training Cooperation
  - Edge-Edge Federal Training Cooperation
+ Edge-Terminal Cooperation
+ Cloud-Edge-Terminal Cooperation

### Edge-Cloud Cooperation
+ Train-and-Predict

  Very classic and well-applied in real-life senarios. A model is first architected on cloud, and continously updated by data feedback from edges. The edge will also use the updated model from cloud in-time to predict on edges. This method is already used in many applications such as Auto Driving. TensorFlow Lite combined with TensorFlow model on cloud has already make this approach feasible. 

+ Cloud Focused Edge-Cloud Cooperation

  Similar to Train-and-Predict, but cloud also share some tasks in prediction. The key is to 'cut' neural network properly, to find a balance between computation and data traffic for edges. This concept is still in research and not wildly applied

+ Edge Focused Edge-Cloud Cooperation

  In this concept, cloud only in charge of inital training task. The model will be downloaded to edge, and updated and trained by the edges themselves.

> Note: This task invovles a pretrained model on cloud to save training time/computation, which lies in the domain of *transfer learning*

### Edge-Edge Cooperation
- Edge-Edge Prediction Cooperation

  This is generally a divide-and-conquer problem, which divides the model from cloud and train seperately on each edges, such that we can still train large models even on computation-limited devices such as IPhones and smart watches. e.g. MoDNN

- Edge-Edge Distributive Training Cooperation

  Each edge has the whole or part of model and train on its own datasets. When reaches some stages, each edge updates parameters to central nodes and joint together as a complete model.

- Edge-Edge Federal Training Cooperation

  This model is proposed for privacy protection. Similar to Edge-Edge Distributive Training Cooperation, but we have certain edge node contains the 'optimal model'. Each edge updates parameters to this node without violating any privacy protocals. 

> Notes: Federal Learning has been successully implemented by Google in 2017 to solve the problem of dynamically updating models in local devices. This concept is similar to the architecture of Multi-task Learning, which, likewise, actively used on medical data.

# Reference Papers:
Shi, Weisong & Zhang, Xingzhou. (2019). Edge Computing: State-of-the-Art and Future Directions. Journal of Computer Research and Development. 56. 69-89. 10.7544/issn1000-1239.2019.20180760. 