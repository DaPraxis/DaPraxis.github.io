---
title:  "Competitive Programming (CP) Study Note 1"
search: false
excerpt: 'Dynamic Programming Practices'
categories: 
  - Competitive Programming
  - Max's Study Note
  - Data Structure
  - Interview
last_modified_at: 2020-12-16 10:00
comments: true
toc: true
author_profile: true
toc_sticky: true
mathjax: true
header:
  teaser: ../assets/imgs/posts/visual4.png
  image: https://images.unsplash.com/photo-1516570161787-2fd917215a3d?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80
---

# Prelude

It's been a while since my last update on posts. I just past a very busy time, when one final exam and one final project are due in the same week. Very unfortunately, my teammate in my final project team was freeloading🤬, which is a huge drag of the progress. Hope the project grade ends up well 🙏

Anyway, now I have passed that difficult time, and get tired of all the bugs in Cyberpunk 2077, it is time to work on some serious stuff -- Dynamic Programming💥. Dynamic Programming(DP) is one of the most interesting while also the most challenging part in computer science contests. It is unavoidable, while powerful enough to make you distinguished from all the participants. This study note tries to categories all the common DP problems, first with some examples and thought process, then reusable code structures that can be reapplied to all similar DP problems. 

When you look at the teaser, you may think this is a post about investment. Hell no! We are interested in the shortest runtime to manage your money ;) Joke aside, in this Post, we will talk about one very popular problem: "making changes" and efficiently applied with DP. We will cover two cases that normally would show up in your contest and generalize to a formula that can be exploited further.

Without further overdue, let's hop on!

# Table of Contents
* What's Up With Your Changes💲?
* Distilled Structure

# What's Up With Your Changes💲?
## Case 1: "Least amount of changes pls"
Suppose you want to get change of **$11**, but the cashier only have infinite **$1**, **$2**, and **$5** to break, what is the minimum pieces of changes you can get?

While, you can try it yourself but I am just gonna spit it out, it is **3**, since **$11 = $5 + $5 + $1**

Then what if the change is **$12**? While it is still**3** since **$12 = $5 + $5 + $2**

What about **$13**? While it is still**4** since **$13 = $5 + $5 + $2 + $1**

What if I ask you, for any amount of changes (integer), and any type of changes (still integer) to break? Like change = **$57** and the cashier has **$1**, **$12**, **$15**, **$17**?

While it is impossible to have bill of **$12**, but that is the charm of programming, one solution works for all!

Let's make a table from above example and try to find a pattern:

| Changes         | Amount |        Optimal Combination                                                    | # of Changes
| --------         | ------ | -------------------------------------------- |----------------------------- |
| [**$1**, **$2**, **$5**]    | **$0**     | $0 = ...           | 0|
| [**$1**, **$2**, **$5**]    | **$1**     | $1 = $1           | 1|
| [**$1**, **$2**, **$5**]    | **$2**     | $2 = $2           | 1|
| [**$1**, **$2**, **$5**]    | **$3**     | $3 = $1 + $2           | 2|
| [**$1**, **$2**, **$5**]    | **$4**     | $4 = $2 + $2           | 2|
| [**$1**, **$2**, **$5**]    | **$5**     | $5 = $5           | 1|
| [**$1**, **$2**, **$5**]    | **$6**     | $6 = $5 + $1           | 2|
| [**$1**, **$2**, **$5**]    | ...    | ...          | ...|
| [**$1**, **$2**, **$5**]    | **$11**     | $11 = $5 + $5 + $1           | 3|
| [**$1**, **$2**, **$5**]    | **$12**  | $12 = $5 + $5 + $2         |   3|
| [**$1**, **$2**, **$5**] | **$13**  | $13 = $5 + $5 + $2 + $1   |        4|
| [**$1**, **$2**, **$5**] | **$14**  | $14 = $5 + $5 + $2 + $2                 | 4|

Any thoughts? Hint: Each row in the table is the **optimized** amount of coins to make changes for certain **changes** and certain **amount**
 {: .notice--info}

 If we want to calculate when amount = **$4** for example, 
  - *Step 1*. we first compare $4 with $1, which is the scenario we try to break $4 with $1. $4-$1=$3. We know that $3 (from 4th row of table)can have $1+$2 for the optimal way making changes, therefore we won't bother making three $1 but just use the optimal value of $4 -> **$3** + $1 -> **$2 + $1** + $1 -> **3 changes**
  - *Step 2*. Then break $4 with $2. From table 3rd row, we know the optimal way to break the remaining $2 is just $2 it self. Then we have $4 -> **$2** + $1 -> **2 changes**
  - *Step 3*. Repeat with $5, $5 is too large, we give up

This procedure conclude with `min(step1, step2, step3) = 2`, which breaking a $4 has to give at least 2 pieces of changes. 

We can repeat the above procedures for any `amount` and any `changes`, by iterating from `amount` = 0 to `amount`, and compare with each `changes` respectively.
