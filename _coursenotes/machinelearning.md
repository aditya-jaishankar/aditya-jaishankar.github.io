---
title: "Machine Learning with Python - From Linear Models to Deep Learning"
categories:
toc: true
layout: single
permalink: /coursenotes/machinelearning/
author_profile: true
toc: true
date: September 2018
read_time: true
---

# Class Notes: 6.86x - Machine Learning and Deep Learning with Python

## Unit 1: Linear classifiers and generalizations

### Basics

**Definiton:**

* Machine learning aims to design, understand and apply computer programs that learn from experience for the purposes of prediction, control, modeling of outcomes and systems. 

**Supervised learning**

* In supervised learning, we aim to find a mapping between the example and a label. There is a full set of mappings, with each mapping specified by a parameter. In supervised learning, we aim to determine the best possible mapping (according to some criteria that defines best) i.e. we aim to find the parameters that produces the best possible mapping. 


* As a concrete example, I can take the movie recommender problem. Let's say I have seen a whole bunch of movies. I can now construct *feature vectors* for each movie i.e. $x^{(i)} = [1\ 0\ 0\ 1\ 1\ 0\ 1]^T$ where each component of the vector refers to some feature of the movie (comedy, action, rating, actors, etc.). I also associate each movie (or equivalently each feature vector) with a label: +1 if I liked it, and -1 if I didn't. The task at hand is: given all these feature vectors and their associated labels, can I predict the label of a hitherto unseen movie (feature vector)?


* Classifiers are mappings that take feature vectors as inputs and generate a label as output. Notationally, $h: X \mapsto \{-1,1\}$ in the case of a binary classifier. For each classifier, we can also quantify how well it performs by calculating the *training error*, which is the proportion of misclassifications of the test set. 

    $$
    \begin{align}
        \varepsilon_n(h) = \dfrac{1}{n}\sum\limits_{i=1}^n [[h(x^{(i)}) \neq y^{(i)}]]
    \end{align}
    $$

    where the $[[ \cdot ]]$ notation denotes the truth value. If the condition inside the brackets is satisfied it returns 1, else zero. It is identical to the indicator $\mathbf{1}\{ \cdot \}$ in the statistics class. I can define both a *training error* and a *prediction error* or *test error*, depending on what I am interested in.
    

* **Notation for the training set:** $S_n = \{X^{(i)}, y^{(i)} \}$


* A common kind of classifier is the linear classifier, that divides the hyper plane linearly into 2 regions. Given a point $x$ on the hyperplane, $h(x) = 1$ or $h(x) = -1$, depending on which region the point lies.


* **Hypothesis plane**: The set of all possible classifiers.


* If we make our classifier too specialized, then it will generalize poorly to the test set. We always seeks the simplest model that produces very low training error and low prediction error.


* There are various other kinds of machine learning algorithms: Unsupervised learning, semi-supervised learning, regression, transfer learning, active learning, reinforcement learning, etc. 


* Let's review some of the concepts:

    * Feature vectors, labels: $x \in \mathbb{R}^d$, $y \in \{-1,1\}$
    * Training set: $S_n = \{(x^{(i)},y^{(i)}, i=1,\ldots,n)\}$
    * Classifier: $h: x^{(i)} \mapsto \{-1,1\}$
    * Training error: $\varepsilon_n(h) = \frac{1}{n}\sum\limits_{i=1}^n [[h(x^{(i)} \neq y^{(i)}]]$
    * Test error: $\varepsilon(h)$
    * Set of classifiers: $\mathcal{H}$

### Linear Classifiers
    
* The linear classifier divides $\mathbb{R}^d$ into two halves: on one side $h(x^{(i)}) = 1$, and on the other side, $h(x^{(i)}) = -1$


* If $x$ were 1D, the decision boundary is just a point; if $x \in \mathbb{R}^2$, then the decision boundary is a line and if $x \in \mathbb{R}^3$, then the decision boundary is a plane. 


* Let us take the case of a 2D boundary. Let us say that we are seeking the set of all linear classifiers that pass through the origin. We are interested in defining the decision boundary as a locus of points. Therefore, we define the normal vector $\boldsymbol{\theta} = [\theta_1\ \theta_2]^T$. We can then draw a simple diagram and easily see that the decision boundary is the set of all points $\{\boldsymbol{x}: \boldsymbol{x}\cdot\boldsymbol{\theta} = 0\}$. Picking some value for $\boldsymbol{\theta}$ defines a classifier i.e. there is one classifier for each value of $\boldsymbol{\theta}$  The classifier itself becomes $h(\boldsymbol{x}; \boldsymbol{\theta}) = \mathrm{sgn(\boldsymbol{x}\cdot\boldsymbol{\theta})}$.


* Finding a linear classifier decision boundary which does not pass through the origin is somewhat trickier, but drawing a figure lets you see that the decision boundary in this case is given by $\{\boldsymbol{x}: \boldsymbol{\theta} \cdot \boldsymbol{x}+\theta_0 = 0\}$. You will see that $\theta_0$ is the negative of the perpendicular distance from the origin to the decision boundary. In this case, $h(\boldsymbol{x}; \boldsymbol{\theta}) = \mathrm{sgn(\boldsymbol{x}\cdot\boldsymbol{\theta} + \theta_0)}$, where $\theta_0$ is the offset parameter.


* Linear classifiers are extremely constained. Take the example of four points arranged as the vertices of a square in alternating fashion. There is *no* linear classifier that would correctly classify all points in this case. We can introduce the notion linear separability. 

    Training examples $S_n = \{(x^{(i)}, y^{(i)}\}$ are said to be linearly separable if there exists a classifier $h(x, \hat{\theta})$ such that $y^{(i)}(\hat{\theta}_1\cdot x^{(i)} + \hat{\theta}_0)>0$ for all $i = 1, \ldots, n$. 


* Actually going about finding the classifier involves the **Perceptron algorithm.** Below we exemplify the perceptron algorithm for the case of a linear classifier with an offset. The idea is that we run through the training set multiple time, and for each example encountered on each run, we update our guess for $\theta$ and $\theta_0$. The algorithm is below
# the data is provides as tuples (xi, yi), where xi in R^d
theta = np.zeros((d,1))
theta_0 = 0
for t in range(T):
    for i in range(n):
        if yi*(np.dot(theta,xi) + theta_0) < 0:
            theta = theta + yi*xi
            theta_0 = theta_0 + yi
return (theta, theta_0) # with this we can construct the decision boundary
* An important concept in determining the decision boundary in the linear classifier is the idea that we want a large margin classifier. The idea here is that it is likely that the test examples are the training example with some small noise. If the decision boundary is too close to one or the other set of training examples, then the noise would push it over the deicision boundary and likely cause the example to be misclassified. Therefore, we want a classifier that offers a **large margin boundary**. 


* We are now ready to recast this problem of finding a classifier as a optimization problem. We have two terms in the objective function:
    * A loss function that accounts for how many examples are being misclassified. i.e. how large is the training error
    * A *regularization* terms, that tries to push the margin boundaries (boundaries passing through the first examples of either kind, i.e., the boundaries very close to each type of training example.)
    The goal is to minimize the objective function while respecting both terms.


* ` obj. fun. = loss + regularization`


* We can play with the margin boundaries by realizing that we can use $\lVert \theta \rVert$ to control how far apart the margin boundaries are from the decision boundary. Remember that the decision boundary is given by $\theta \cdot x + \theta_0 = 0$, and hence it is also true that

    $$
    \begin{align}
        \dfrac{\theta \cdot x + \theta_0}{\lVert \theta \rVert} = 0
    \end{align}
    $$

    However, the distance of a point from the decision boundary (at which we can draw a margin boundary) is
    
    $$
    \begin{align}
         \dfrac{\theta \cdot x^{(i)} + \theta_0}{\lVert \theta \rVert}
    \end{align}
    $$
    
    which can be controlled by the value of $\lVert \theta \rVert$. In other words, the margin boundary is **defined** as the set of points whose distance from the decision boundary is $1/\lVert \theta \rVert$.


* We are finally ready now to write out the objecive function. As described above, we need to define a loss function and define the regularization parameter appropriately. First we tackle the loss function (here it is a **hinge loss**):

    $$
    \begin{align}
        \textrm{Loss}_h(z_i; \theta, \theta_0) = \begin{cases}
                                               0 & z_i \geq 1\\
                                               1 - z_i & z_i < 1
                                           \end{cases}
    \end{align}
    $$
    where $z_i = y^{(i)}(\theta \cdot x^{(i)} + \theta_0)$. Intuitively, $z_i$ is a measure of how far from the decision boundary I lie, and multiplying that with $y^{(i)}$ tells me how correct or wrong I was. The hinge-loss defined above applies a penalty in proportion to how wrong I was. If I am correct, there is no penalty.
    
    Next, we know that we want to push the margin boundaries outward, i.e., we want to maximize $1/\lVert \theta \rVert$. This is the same as minimizing $\frac{1}{2}\lVert \theta \rVert^2$. We can now combine everything into a single objective function as follows:
    $$
    \begin{align}
        J(\theta, \theta_0) = \dfrac{1}{n}\sum\limits_{i=1}^n\textrm{Loss}_h(z_i) + \dfrac{\lambda}{2}\lVert \theta \rVert^2
    \end{align}
    $$
    where $z_i$ has been defined above, and $\lambda$ is a regularization parameter that is to be chosen depending on how you want to weight the objective function. Note that it is not a Lagrange multiplier. It is not something that is solved for as part of the equations. 
  
  
* We go through the trouble of defining the objective function because while the perceptron algorithm will give you some solution if the conditions of convergence of the algorithm are satisfied, this solution may not be robust. There are possibly many decision boundaries that will classify the data, but it may not be robust. If the decision boundary is too close to the data points, then if we get another point near the boundary but with some small error, it might be pushed into the wrong region due to the error. So we want to push the margin boundaries as far wide as possible. Hence we define the loss function to make sure we are correctly classifying the points, and then the regularization to make sure our margins are wide (the decision boundary is as far away as possible from the first point). 


* Intuition about the objective function: If I were to divide both sides by $\lambda$, minimizing the objective function is the same as minimizing $(1/\lambda)$ times the objective function. Therefore, everything now collapses into a new parameter $C = 1/n\lambda$, and the objective function $J'(\theta, \theta_0) = (1\lambda)J(\theta,\theta_0)$ is given by
$$
\begin{align}
    J'(\theta, \theta_0) = C \cdot \textrm{Loss}_n(y^i(\theta \cdot x^i + \theta_0)) + \lVert \theta \rVert^2
\end{align}
$$
    where the loss function could be a hinge loss or someother suitable loss function. What tends to happen is that as we increase $C$, we weight minimizing the loss more and hence the training error decreases with increasing $C$. However, if we were to look at the test error, there is a U-shaped curve with a sweet spot in terms of $C$. This is because if $C$ is too small, then we are not adequately capturing all the structure in the data and hence we would have a poor test error. On the other hand, if $C$ were too large, we would be overfitting the training data and hence we would again have a large test error. We want to be as close to the sweet spot as possible. How do we achieve this? We divide the training set into a subset training set and a *validation set*, and we test the validation error as a function of $C$. The validation error is used as an approximate proxy for the real test error, and the hope is that we come close to the real test error by doing this. 
    
**Some additional information on the convergence of the perceptron algorithm**

* There is a proof for the convergence of the perceptron algorithm under certain conditions. Below we state the result for the case of no offset (i.e. $\theta_0 = 0$). We first make the following assumptions:
    * There exists a $\theta^*$ such that 
        $$
        \begin{align}
            y^i\dfrac{\theta^* \cdot x^i}{\lVert \theta^* \rVert} > \gamma 
        \end{align}
        $$
        for all pairs $x^i, y^i$ and for some $\gamma > 0$. That is, we assume that the points are linearly separable. 
    
    * All examples are bounded i.e. $\lVert x^i \rVert < R$. 
    Then it can be shown that $k < R^2/\gamma^2$, where $k$ is the number of errors the perceptron algorithm makes. There is a similar (more messry result for the case for when $\theta_0 \neq 0$. See this reference [here](http://www.cs.columbia.edu/~mcollins/courses/6998-2012/notes/perc.converge.pdf). There is also a paper on perceptron mistake bounds i.e. how large mistakes can be. See [here](https://arxiv.org/pdf/1305.0208.pdf). 
    
### Gradient Descent

* One algorithm to use to implement the optimization is to use Gradient Descent. The idea here is to move in the direction of decreasing gradient because at the optimum, the gradient is 0. Therefore, we can formulate the algorithm as stepping through different values of $\theta$ such that

    $$
    \begin{align}
        \theta^{(k+1)} = \theta^{(k)} - \eta \nabla_{\theta} J(\theta,\theta_0)\rvert_{\theta = \theta^{(k)}}
    \end{align}
    $$
    
    where $\eta$ is the *learning rate*. 
    
* Now, gradient descent can be very slow in the case of large data sets becuase we have to evaluate the loss function at each data point. One way around this is to use *stochastic gradient descent* (SGD), where instead of doing the whole average at each optimization step, we just pick one particular value of $i$. We illustrate this algorithm for the case where there is no offset. Instead of using the average loss function, we simply pick a value of $i$ at random and then find $J_i(\theta) = \textrm{Loss}(y^i\theta \cdot x^i) + (\lambda/2)\lVert \theta \rVert^2$ and then our update is 

    $$
    \begin{align}
        \theta \xleftarrow[]{}\theta - \eta_t \nabla J_i(\theta)
    \end{align}
    $$

    where $\eta_t$ is the learning rate that is now different for each step. Because of the stochasticity of the process, we need to now decrease the learning rate at each step. There are actually slightly stricter conditions to ensure that the scheme converges. We require the following conditions:
    
    * As $t \to \infty$, $\eta_t \to 0$.
    * $\sum \eta_t = \infty$ i.e. we want the learning rate to be large enough to allow for convergence.
    * $\sum \eta_t^2 < \infty$ i.e. they should be *square summable*. We want to decrease the variance from the stochasticity that we are introducing. 
    
* Note that this algorithm is different from the perceptron update algorithm because even in the case the loss is zero (i.e.) the example is correctly classified, we stil have the contribution coming from the regularization term becuase we are still trying to push the margin boundaries outwards. Therefore, $\theta$ **is updated even when there is no mistake.**

**Support vector machines**

* The support vector machine is a specific case of a linear classifier that ensures that the training loss is zero, and tries to push the margins as far apart as possible. In other words, for the training loss to be zero, we require that 

    $$
    \begin{align}
        y^i(\theta \cdot x^i + \theta_0) \geq 1
    \end{align}
    $$
    
    Therefore, the optimazation problem becomes a quadratic problem where we want to minimize $\lVert \theta^2 \rVert$ subject to the constraint $y^i(\theta \cdot x^i + \theta_0) \geq 1$. The latter conditions comes from the fact that we want the distance of the point from the decision boundary to be at least as large as the distance between the decision boundary and the margin boundary. That is, we want 
    
    $$
    \begin{align}
        \dfrac{y^i(\theta \cdot x^i + \theta_0)}{\lVert \theta \rVert} \geq \dfrac{1}{\lVert \theta \rVert}
    \end{align}
    $$


### Learnings from Project 1

* Project 1 was really great. We implemented three different linear classifiers from scratch to do sentiment analysis using a bag of words technique. The idea here is to get a dataset of reviews which has been labeled as being positive or negative. We then first form a dictionary of all the unique words in the entire dataset. We can then define feature vectors for each review, where each component of the vector is either 0 if the word does not appear in the review, or 1 if the word does appear in the review. Note that we can make this more sophisticated - for each example, instead of using a binary count, we could use the actual count in the feature vector (i.e. the frequency of the appearance of the words), or we could capture other information, such as if there are a lot of exclamation points, or if the review is in all caps. We could also have a "stopwords" file, which is populated with useless non predictive words like 'me', 'and', 'yours', etc. which essentially act as corrupting noise, and then remove these words from the bag. I now have a feature matrix $\mathbb{X} \in \mathbb{R}^{n \times d}$, where each row is a feature vector corresponding to a review. Associated with this is a label vector $\mathbf{Y} \in \mathbb{R}^n$, which tells us if the review was positive or negative. The project also included the ideas of n-grams, as opposed to the unigram model introduced above, where we use n word sequences as the features. If a particular n-word sequence appears in a review, then the corresponsing component in the feature vector is 1, else it is zero. **It have been shown in some publications that 3-grams and 4-grams do particularly well in spam filtering.** 


* The next step is to divide this dataset into a training set, and a validation set. In the case of a linear classfier, we wish to find a vector $\theta$ and an offset parameter $\theta_0$ that best classifies the the training dataset. I can either use a regularization parameter and then define the objective functiontion involving this parameter and a Loss function (say hinge loss), or I can use one of the three algorithms introduced: the perceptron, the average perceptron, and the pegasos algorithm. I input the feature matrix, and the algorithm goes about implementing the corresponding sequence of steps to find $\theta$ and $\theta_0$. 


* Once the training set has been classified, I can do the validation, find training and validation errors, and then use hitherto unseen data (the test set) to make predictions about the sentiment of the review is a positive (+1) or a negative (-1), respectively. 


* This was a very detailed and nice project where pretty much was implemented from first principles, except for a bunch of helper functions. Raschka's book also has a chapter dedicated to sentiment analysis `scikit`'s vectorizers and `nltk`, which might be worth reading through at a later time. 

## Unit 2: Nonlinear classification, linear regression and collaborative filtering

### Linear Regression

* Linear regression allows us to predict not just binary labels but it allows us to predict tha label $y \in \mathbb{R}$. By proper engineering of feature vectors, it turns out that this approach is actually quite powerful. As we saw in the statistics class, we define the model to be $\mathbf{Y} = \mathbb{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$. I am interested in finding (in the case of either Gaussian noise or in the case of least square regression) the following:
$$
\begin{align}
    \hat{\boldsymbol{\beta}} = \textrm{argmin}_\beta (\boldsymbol{Y} - \mathbb{X}\boldsymbol{\beta})^T(\mathbf{Y} - \mathbb{X}\boldsymbol{\beta})^T
\end{align}
$$

    This can be solved exactly, as seen in the statistics class, and it turns out that $\hat{\boldsymbol{\beta}} = (\mathbb{X}^T\mathbb{X})^{-1}\mathbb{X}^T \mathbf{Y}$. 


* There are two kinds of errors when we talk about regression. One is *structural error*, where our model is not refined or complicated enough to capture all the features of the data i.e. there aren't enough parameters in the model. The other kind of error is an estimation error, where we do not have enough data to correctly estimate all the parameters in the model. Note that these two errors pull in opposite directions. If our model is very complicated, we may not have enough data to find all parameters properly and we have an estimation error. On the other hand, if we have only a few parameters, our estimation error will be small but we will not have enough refinement/complexity in the model to capture all the features. This is in essence what is known as the bias-variance tradeoff. The more complex the model, the more the variance and the smaller the bias. So there is usually a sweet-spot. We can use regularization to try and optimize the bias-variance tradeoff.


* We can define the emperical risk as the following function:
$$
\begin{align}
    R_n(\theta) = \dfrac{1}{n}\sum\limits_{i=1}^n\textrm{Loss}(y^{(t)} - x^{(t)}\theta)
\end{align}
$$

    where the Loss can either be hinge loss or a squared loss or something else. 


* It is easy to show that for the squared loss (with a 1/2 factor to make the derivative easier), the updates to $\theta$ in the gradient descent algorithm expression can be written as 
$$
\begin{align}
    \theta^{(k)} = \theta^{(k-1)} + \eta(y^{(t)} - \theta x^{(t)})x^{(t)} 
\end{align}
$$


* Some intuition for regularization: Regularization is one way to try and prevent overfitting. By forcing the norm of the vector to be 0, we are simplifying the problem and decreasing the number of parameters. 


* Gradient descent with regularization, otherwise called ridge regression. The objective function in this case is defined as:
$$J(\theta) = \dfrac{1}{n}\sum\limits_{i=1}^n \dfrac{1}{2}(y^{(i)} - \theta \cdot x^{(i)})^2 + \dfrac{\lambda}{2}\lVert \theta \rVert ^2$$

    It is easy to derive both a numerical implementation of gradient descent as well as a closed form solution. The gradient can be written as:
    $$
    \begin{align}
    \nabla_\theta J(\theta) = -\dfrac{1}{n}\sum(y^{(i)} - \theta \cdot x^{(i)}) x^{(i)} + \lambda\theta = 0
    \end{align}
    $$
    and we can solve for $\hat{\theta}$ as $\hat{\theta} = A^{-1}b$. It's easy enough to derive what the matrix $A$ and the vector $b$ are. There is also a parallel with what we learned in the statistics class, where for linear regression wit Gaussian noise, $\hat{\theta} = (\mathbb{X}^T\mathbb{X})^{-1}\mathbb{X}^T\mathbf{Y}$.
    
    Numerically, if we were performing stochastic gradient descent, we woulf first pick a $t$ at random for $t \in \{1, 2, ..., n\}$. We initialize $\theta = 0$. Then, we have
    $$
    \begin{align}
    \nabla_\theta J(\theta) = -(y^{(t)} - \theta \cdot x^{(t)}) x^{(t)} + \lambda\theta
    \end{align}
    $$
    and hence we update $\theta$ as
    $$
    \begin{align}
    & \theta \leftarrow \theta -\eta(\lambda\theta - (y^{(t)} - \theta \cdot x^{(t)}) x^{(t)} \\
    \Rightarrow &\theta \leftarrow (1-\eta\lambda)\theta + \eta(y^{(t)} - \theta \cdot x^{(t)}) x^{(t)}
    \end{align}
    $$
    and proceed until convergence is achieved. Regularization is something we will see over and over again in this class.
    
### Nonlinear classification

* There are many instances where the data is not linearly separable i.e. there does not exist a $\theta^*$ such that $y{(i)}(\theta^* \cdot \boldsymbol{x}^{(i)}) > 0$ for all $i$. To form classifiers for such data, we turn to nonlinear regression. The idea here to take our data points $x \in \mathbb{R}^d$ and construct a feature vector i.e. apply a transformation to the datapoints $\phi(x) \in \mathbb{R}^p$, where $p >> d$. We then find a linear classifier for the feature vector in this higher dimensional space. In other words, we convert our classifier from $h(x; \theta, \theta_0) = \textrm{sgn}(\theta \cdot x + \theta_0)$ to a problem that looks of the form $h(x; \theta, \theta_0) = \textrm{sgn}(\theta \cdot \phi(x) + \theta_0)$.


* For example, let's say we have data in $\mathbb{R}^2$ and $x = [x_1\ x_2]^T$. If we have the points (and labels) as $((1,1), 1), ((-1, -1), 1), ((-1,1), -1) \textrm{ and } ((1, -1), -1)$, there is no linear classifier that works for this dataset. In this case, we define $\phi(x) = [x_1\ \ x_2\ \ x_1x_2]^T$ (we always want to atleast keep the datapoints because we don't want to throw out any data and we minimize the possibility that we do worse) and hence $\theta \in \mathbb{R}^3$ also. Now, we try to find the $\theta$ such that $h(\phi(x); \theta, \theta_0) = \textrm{sgn}(\theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1 x_2 + \theta_0)$.  


* It seems that there is no obvious or easy way to figure out how to define the feature vector. One commonly used trick is to find all possible polynomials of a certain degree and define $\phi(x)$ through that. If we over do it, we hope that things like drop-out (leaving out a point to see how well the resulting classifier does) and regularization can help draw us toward the balance between having adequate complexity and yet not overfitting. 


* For example, if we had $x = [x_1\ \ x_2]^T$ and we want to define a feature vector up to degree 3, we would have $\phi(x) = [x_1\ \ x_2\ \ x_1^2\ \ x_2^2\ \ x_1 x_2\ \ x_1^3\ \ x_2^3\ \ x_1^2 x_2\ \ x_2^2x_1]^T$ i.e. the data itself plus every possible polynomial upto degreee 3 that can be formed with the data.


* Another example to provide intuition. Let's say we have data points such that all points within a circle of radius $r$ are labeled +1, and all points outside of that circle are labeled -1. Of course, there is no linear classifier that would work for this data. We can intuit that we want a classifier that tells that if $x_1^2 + x_2^2 - r < 0$, then we want a label +1, and a label -1 otherwise. We then do the transformation $\phi: \mathbb{R}^2 \rightarrow \mathbb{R}^3$, $\phi(x) = [x_1\ \ x_2\ \ x_1^2\ \ x_2^2]^T$ and attempt to find $\theta, \theta_0$. In this case we can do this by inspection but of course it can in general be more difficult. By defining the decision boundary as $h(x; \theta, \theta_0) = \textrm{sgn}(\theta \cdot c + \theta_0)$ Here, we will find that $\theta = [0\ \ 0\ \ -1\ \ -1]^T$, and $\theta_0 = r$. Then we get a prediction of +1 if $\textrm{sgn}(\theta \cdot x + \theta_0) > 0$. This would be easy enough to implement: first define the points and add some randon noise. Then we add the regularization term and use gradient descent or the linear perceptron algorithm to converge onto the solution. It would be worth trying to implement this for the following cases:

    1. Perceptron algorithm without regularization
    2. Perceptron algorithm with regularization
    3. Gradient descent without regularization
    4. Gradient descent with regularization
    
    Maybe come back and implement it here


```python
# Placeholder for implementing the algorithms above
#
#
#
```

### Kernel Methods

Note: There are some really nice materials available to understand Kernels and Kernel tricks better. I assume this is a really important and powerful part of ML and NN in general, so I will have to read these and internalize at some point:

1. https://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/lecture-notes/MIT15_097S12_lec13.pdf

2. http://members.cbio.mines-paristech.fr/~jvert/publi/04kmcbbook/kernelprimer.pdf

3. https://pdfs.semanticscholar.org/2eb2/ca05a79d1d81033237aad416ad4a1ce90a70.pdf

4. https://www.ics.uci.edu/~welling/teaching/KernelsICS273B/svmintro.pdf


* These methods are useful in the case where calculating inner products is computationally cheap. Consider the two feature vectors $\phi(x) = [x_1\ \ x_2\ \ x_1^2\ \ \sqrt{2}x_1x_2\ \ x_2^2]^T$ and $\phi(x') = [x_1'\ \ x_2'\ \ x_1'^2\ \ \sqrt{2}x_1'x_2'\ \ x_2'^2]^T$. When the feature vectors are defined this way, it is easy to see that $\phi(x)\cdot\phi(x') = (x\cdot x') + (x\cdot x')^2$. We define the **Kernel** as 
    $$
    K(x, x') = \phi(x)\cdot \phi(x')
    $$
    In the cases where we have feature maps that look of this form, it is advantageous to define our decision boundaries and perform our convergence algorithms by focussing on $K(x, x')$ instead of $h(x; \theta, \theta_0)$.


* How exactly is this useful? Acutualy defining and writing out a nonlinear map to transform the data into a higher dimensional space can be very computationally expensive. Moreover, in some cases the nonlinear map can be infitinte dimensional, like in the RBF kernel (this is an infinite series). In these cases, Kernel methods are very handy. 

    Their utility is best seen when implementing this in practice in an algorithm such as the Kernel perceptron algorithm. Keep in mind again that lets say we have data that is distributed as a circle (the example used above). Consider again the pseudocode for the perceptron algorithm:

theta = np.zeros(d)
for i in range(n):
    if y[i]*np.dot(theta, x[i]) < 0:
        theta += y[i]*x[i]
* Taking a step back,  let us write out mathematically what this algorithm is doing. Any any point in the agorithm's run, we can write the current value of $\theta$ as
    $$
    \theta = \sum\limits_{j=1}^n \alpha_j y^{(j)}\phi(x^{(j)})
    $$
    where $\alpha_j$ is the number of mistakes made with the $j$-th example until that point in the algorithm. This representation is easily seen when we look at what the update step is doing and how these updates accumulate as we proceed through the algorithm. Now, if we look at what the checking or prediction step is doing, we have a term that looks like $y^{(i)}\theta \cdot \phi(x^{(i)})$. Therefore, we take the dot product on both sides of the above equation to get
    $$
    \begin{align}
    \theta \cdot \phi(x^{(i)}) & = \sum\limits_{j=1}^n \alpha_j y^{(j)}\phi(x^{(j)}) \cdot \phi(x^{(i)}) \\
                               & = \sum\limits_{j=1}^n \alpha_j y^{(j)}k\left(\phi(x^{(j)}), \phi(x^{(i)})\right)
    \end{align}
    $$
    by the definition of the kernel function. Note, I can also construct a Kernel matrix which contains the pair wise inner products for nonlinear maps between the data vectos taken pairwise i.e. $K_{ij} = k(\phi(x)^{(i)}, \phi(x)^{(j)}) = \phi(x)^{(i)} \cdot \phi(x)^{(j)}$. Therefore, if I am given the Kernel, I don't have to think about the details of how to construct the feature maps and the feature vector and deal with very high dimensional vectors. I can calculate cheap inner products instead. With this in mind, I can now rewrite the perceptron algorithm as the **kernel perceptron algorithm** (for a specifically chosen kernel):
x = np.array([ [1,2],[3,4],[5,6] ])
y = np.array([1, 1, -1])
n = x.shape[0]
d = x.shape[1]

alpha = np.zeros(n)

def phi(x):
    return np.array([x[0], x[1], x[0]^2 + x[1]^2])

def kernel(x1, x2):
    return (1 + np.dot(x1, x2))^2

def kernel_checker(alpha, i):
    return np.sum(alpha * y * kernel(x, x[i]), axis=0)

def prediction(xp):
    return 2*(np.sum(alpha*kernel(x, xp), axis=0) > 0) - 1 

for i in range(n):
    if y[i]*kernel_checker(alpha, i) <= 0:
        alpha[i] += 1

```python

```

* We can now define a whole bunch of functions $K: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}$, but not every such function is a kernel. A function $K$ is a kernel only if I can represent that function as some innner product of a feature vector. There is a more formal conditon called *Mercer's condition* that specifies which functions are valid kernels, but this is outside the scope of the class. For now, all we need to know is that if I can define a function in terms of some inner product space, then it is a valid kernel.


* Note that when using the kernel perceptron algorithm, we eliminate the need the explicitly defining both the feature map $\phi(x)$ as well as explicitly checking classifications in terms of theta.


* If we do have a valid kernel, then we can apply composition rules to construct kernels from kernels. The following composition rules apply:

    1. $K(x, x') =1$ is a kernel function
    2. Let $f: \mathbb{R}^d \rightarrow \mathbb{R}$ and $K(x, x')$ a valid kernel function. Then $\tilde{K}(x, x') = f(x) K(x, x')f(x')$ is also a kernel.
    3. The sum of two valid kernels is a kernel.
    4. The product of two valid kernels is a kernel.
    
    
* In summary, a kernel is a function that can be written as inner products of some feature map. The reason for introducing the kernel is (1) it allows us to handle non-linear problems in a natural way and (2) it is computationally much more efficient than calculating feature maps at every step/iteration. From the formulation above, it is easy to see how for example the perceptron algorithm can be rewritten in terms of a kernel function. There are rules on what constitutes a kernel function, and to compose kernels from other valid kernels.


* One super powerful kernel is the **radial basis function** or radial basis kernel which is defined as:
$$
K(\mathbf{x}, \mathbf{x'}) = \exp\left(-\dfrac{1}{2}\lVert \mathbf{x} -  \mathbf{x'} \rVert^2\right) 
$$
  Based on the composition rules described above, it can be shown that the RBF is infinite dimensional. Can you quickly reason about this to show this is true?


* There are other non-linear algorithms that are good for almost any setting is a random forest. Learn about random forest and write some notes (Raschka and ESLI). 


**Some notes from "A primer on kernel methods" by Vert, Tsuda and Scholkopf**

* Any function that can be written as an inner product of two vectors is a valid kernel. We can then define a kernel matrix. For $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n \in \mathbb{R}^d$
    $$
    K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)
    $$
    For the simplest case of a linear kernel, $k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$. 
 
 
* The nice thing about kernel methods is that it opens up a way to represent anything - strings, images, molecules, etc. All I need to a have is a feature map that transforms the data into a feature vector and then the kernel is defined by the inner product of these two feature vectors. A kernel boils down to taking the data, applying the feature map to get a feature vectors, and then finding the pairwise dot-products of all these feature maps. In practice, however, we don't have to define of know what the feature map is. Sometimes, it is impossible to tell. 


* The real power of kernel methods is that I don't need to actually explicitly define the feature vectors. This saves a lot of computation time. Indeed, in some cases, the feature vector can be infinite dimensional, for example the RBF kernel where the inner product is over a function. Using kernel methods allows us to bypass the need for actually knowing what the feature vectors are. 


* More generally, a nice intutition to have is that a kernel defines the similarity between two examples. When we implement the kernel method in the SVM, we are actually weighting points according to how similar they are to all the points seen so far. If I make many (high $\alpha_j$) large (high $K_{ij}$) mistakes, then this has to be appropriately weighted when running the algorithm. 


**Implementation of the kernel perceptron algorithm for a radial basis kernel**

This has been done in the project.

### Recommender systems

* A good example of the recommender system problem is that of netflix trying to figure out what movies I would like next, given the movies I have liked so far. One may immediately think about regression: I can come up with a bunch of features in a feature vector (who directed, did it have a happy ending, were there particular actors, etc.) but it is hard to build such a large feature vector. It is not even clear which features to include. Moreover, the matrix $Y_{n\times m}$ of users as rows and movie rating for that user as column is extremely sparse. Most movies are only rated a handful of times. So this linear regression approach becomes difficult because each user might have participated enough to collect enough data just for him. So here the idea is to borrow for other users based on what we know about the test user. 


* **Another approach is k-NN.**

See Introduction to Statistical Learning by Gareth et al. for a really nice explanation of what's going on. 


* This is a fairly simple algorithm which works surprisingly well. The idea is to try and emulate the gold standard of the Bayesian algorithm, which (in the context of a classifier) uses as the estimator the maximum conditional probability. In other words if we want to predict the label $Y$ for a dataset, given $X = x_0$, we are interesting in finding
    $$
    \hat{Y} = \max_j \mathbf{P}(Y=j|X=x_0)
    $$
    In reality, I don't always know this conditional probability (in the case of Naive Bayes, I assume independence and hence the conditional probability can be handled). Therefore, I need to come up with some approximate ways to evaluate the conditional probability. One way is to approximate the conditional probability with a proportion. Assume that I am given a feature vector $X=x_0$ and an integer $K$. The k-NN algorithm evaluates uses a user-defined distance metric, find the K nearest points to the test point based on this metric, and then identifies the label given by
    $$
    \textrm{argmax}_{y_i}\dfrac{1}{K}\sum_i\mathbf{1}\{Y=y_i|X=x_0\}
    $$
    where $i$ is an index given to the set of points $\mathcal{N}_0$ that constitute K nearest neighbors to $X=x_0$ and assigns this label. Intuitively, the algorithm tries to find the most frequently occuring label in $\mathcal{N}_0$. Note that I can also use this for making quantitative predictions. 
    

* The factor $K$ is hugely important in determining how the algorithm does. If $K$ is too small, then I capture all the details of the dataset and I overfit leading to poor performance i.e. I have low bias and large variance. If $K$ is too large, I don't capture the details of the dataset, and I have large bias and low variance. Equivalently, Increasing $K$ leads to lower training error and larger test error. Therefore, in this case too, there is a classic U-shaped curve in the test error as $K$ is increased. One way to choose an optimal $K$ is to plot the test error as a function of $K$ (by splitting the dataset) and then seeing where the minimum in the test error occurs.

**Back to matrix factorization**

* k-NN does not work very well in this case because the matrix $Y$ is so sparse. Very few users rate lots of movies, and very few movies are rated by lots of users. 

* The naive approach to doing this is to borrow from what we have seen in regression, but it turns out to be wrong. Under this assumption, let's say that we denote as $D$ the set of all users for which a rating is given i.e. $D = \{(a, i) \textrm{ for which } a \textrm{ and } i \textrm{ are given.} \}$. We want to fill up the matrix $Y$ and have entries in all the empty spots. Let us call this matrix $X$. Now, we want to fill $X$ in such a way that we minimize the errors in the cells that belong to $D$, and we want to apply regularization to decrease complexity and hence we also have a regularization term. The objective function is then

    $$
    J(X) = \sum\limits_{(a,i) \in D} (Y_{ai} - X_{ai})^2/2 + \dfrac{\lambda}{2}\sum\limits_{(a,i) \notin D} X_{ai}^2
    $$
    
    If we assume that all the entries are independent, i.e. a users rating for one movie does not affect the rating of his other movies, or that rating across users are independent, we get the absurd result that for $(a, i) \in D$, $X_{ai} = Y_{ai}/(1+\lambda)$ (when we expected them to be equal), and for $(a, i) \notin D$, $X_{ai} = 0$. This is wrong because of the independence assumption. Users are not independent, and the rating across movies for a single user are also not independent. Hence we totally ignored cross terms in the derivative above, which all evaluated to zero because of the condition of independence.  
    
    
* The correct approach to take here is Matrix Factorization. The idea here is that we assume the recommendation matrix $\mathbb{X}$ to be of the form $$\mathbb{X} = UV^T$$, where $U$ and $V$ are vectors (if $\mathbb{X}$ is rank 2). The idea here is that the factorization helps separate out the user's baseline tendencies to assign movie rating in the vector $U$ and the vecotr $V$ captures intrinsic features or qualities of the movie itself. We can generalize further to matrices of higher rank, for example, trying to capture a user's tendency for ranking movies of a particular genre, or the intrinsic qualities of certain movies (in which case we will not have vectors but higher order tensors being factorized into lower order matrices). 


* The algorithm itself proceeds by splitting up the objective function as

    $$
    J(U, V) = \sum\limits_{(a,i) \in D} (Y_{ai} - UV^T_{ai})^2/2 + \dfrac{\lambda}{2}\lVert U \rVert^2 + \dfrac{\lambda}{2}\lVert V \rVert^2
    $$
    
    We now compare the imcomplete rating matrix with the constructed rating matrix. For example, if we have the following matrices (assuming rank 1 matrices for now):
    $$
    Y = \begin{align}
        \left[
            \begin{matrix}
            1 & 3 & ?\\
            4 & ? & 3
            \end{matrix}
        \right]
    \end{align}
    $$
    
    $$
    X = UV^T = \begin{align}
                \left[
                    \begin{matrix}
                    u_1v_1 & u_1v_2 & u_1v_3\\
                    u_2v_1 & u_2v_2 & u_3v_3
                    \end{matrix}
                \right]
            \end{align}
    $$
    
    And let's say I want to find the vectors $U$ and $V$, I use an iterative approach where I first assume some values for $V$ for example, and then solve the set of equations
    $$
    \nabla_U J(U,V) = 0
    $$
    Once $U$ is found in this iteration, I go back and solve for $V$ using a similar procedure (find $\nabla_V J(U,V) = 0$) and repeat this process iteratively until convergence is found. Note that there are no guarantees for global optimization, only for local convergence, and hence the choice of the initial value is important. Perhaps it behooves me to random sample the space of initial conditions and then find the solution with the smallest objective function.
    
## Project 2 Learnings

* `sklearn.svm.LinearSVC()` does support vector classification for one vs. rest type problems (i.e soluble in NMP or not soluble in NMP). The methods that I would need the most are `clf.fit` and `clf.predict`, where I assigned the classifier to a variable `clf = sklearn.svm.LinearSVC()`. The `random_state` parameter determines how to shuffle the data while doing the fit, and the parameter `C` is the penalty parameter on the error term (identical to the parameter $C = 1/n\lambda$ in the class notes). Docmentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html). 

    Note that nothing different needs to be done if I need to do a multiclass classification. `LinearSVC` automatically implements multiclass classification by using the one-vs-rest strategy and the predictions are assigned labels according to this one-vs-rest strategy. How does this actually work in my own words? I guess what it is doing is to train `n_classes` (the total number of unique class labels it finds in the training label vector). Then while making predictions, I run through all the classes I have trained and see if I get theta.x >0 or theta.x <0. In the ideal case, most predictions will be <0 and I return that class label for which it is >0. In the real case, what it is probably doing is returning the class label that is most positive. Verify on the forum if this is infact true.
    
    
* This project was really nice. I especially found the need and strategies for vectorization and sparse matrices for large matrices particularly compelling. Check out `scipy.sparse.coo_matric()` which allows me to build sparse matrices by passing a set of values to be filled in at particular coordinates that is also passed. 


* Although intuitive, there is a nice little snippet to reconstruct an image given the transfromed PCA. Essentially, make the transformation by doing 
    $$
    \begin{align}
    X' = XV
    \end{align}
    $$
    where $X \in \mathbb{R}^{n \times d}$ is the raw data and $V \in \mathbb{R}^{d \times k}$ is the eigenvector matrix of the first $k$ eigenvectors of the covariance matrix (i.e) the eigenvectors corresponsing to the largest $k$ eigenvalues of the covariance matrix. This matrix multiplication is essentially a projection of the dataset onto the eigenvector directions. Therefore, $X' \in \mathbb{R}^{n \times k}$. Therefore, to get back the dataset, we do $X = X'V^T$ (by orhtogonality of the eigenvectors. If $V$ was square, we could use the argument that $V^{-1} = V^T$. 
    

* This lends itself to a general strategy using PCA. Find the principal components, and select the first $k$ significant principal components, project the data set onto these principal components, and then apply the ML agorithms on this dimensionality reduced dataset. We can also use the top two PCs to visulaize the data on 2D plots and see if there is some global structure that is captured by just the first two principal components. 


* Nice motivation for the need for kernel methods. Consdier the MNIST database. Each image is 28 x 28 = 784 dimensinal (i.e. a pixel value for each pixel). Now if we want to apply a cubic map such as $\phi(x)^T\phi(x') = (1 +x^Tx')^3$, the resulting mapping vector would be of a **huge dimension** and essentially computationally intractable. Hence we find that we don't actually have to generate the map $\phi$, but just calculate things implicitly.  


* Implementation of the RBF kernel involves finding pairwise distances of all rows in each of the two matrices, and then taking the L2 norm of these pairwise distances. There is a really neat trick you can use to do this - realize that for two vector $x, y \in \mathbb{R}^d$,
    $$
    \lVert x-y \rVert^2 = \lVert x \rVert^2 + \lVert y \rVert^2 - 2*x.y 
    $$

    We essentially want to use broadcasting tricks to be able to do this efficiently and quickly without resorting to for loops. We use the expansion trick above to find the Kernel matrix of two matrices $X \in \mathbb{R}^{n \times d}$ and $Y \in \mathbb{R}^{m \times d}$. Note that given matrix $X$ and $Y$ as
    $$
    \begin{align}
    X &= \left(
            \begin{matrix}
            ---x^{(1)}---  \\
                  \vdots \\
                  ---x^{(n)}---
            \end{matrix}
        \right)_{n \times d}\\
    Y &= \left(
            \begin{matrix}
            ---y^{(1)}---  \\
                  \vdots \\
                  ---y^{(m)}---
            \end{matrix}
        \right)_{m \times d}
    \end{align}
    $$
    
    I can use `np.linalg.norm()` and `np.ones()` cleverly to construct the RBF kernel below. Make sure you can verbalize why this works.
    
    ```
    def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    n = X.shape[0]
    m = Y.shape[0]
    term1 = (np.array([np.linalg.norm(X, ord=2, axis=1)]).T)**2*np.ones((n,m))
    term2 = np.array([np.linalg.norm(Y, ord=2, axis=1)])**2*np.ones((n,m))
    term3 = np.matmul(X,Y.T)
    arg = term1 + term2 - 2*term3
    return np.exp(-gamma*arg)
    ```
    
    Essentially I am constructing the Kernel using
    $$K_{ij} = \exp\left(\lVert x^{(i)} \rVert^2 + \lVert y^{(j)} \rVert^2 - 2 x^{(i)}\cdot y^{(j)}\right)$$
    Hence you can see the need for the `np.ones()` calls of appropriate dimensions. Also think clearly about what should be row vectors and what should be column vectors (and hence the need for the slightly clunky looking transpose operation.


```python
# Also, for the heck of it, continue with finding feature vectors for the solubility parameter stuff, 
# find an RBF kernel, implement the kernel perceptron algorithm until convergence, 
# and then see how the predictions vary. This would be a great exercise to really nail down the concepts here. 




```

## Unit 3: Neural Networks

### Introduction to feedforward neural networks

* Recall the case of using SVMs to do a classification task. Imagine that I am given a vector $x \in \mathbb{R}^d$. I define some feature representation $\phi(x)$ and then find a $\theta$ that determines the classification given the feature representation. But the problem here is that I am not tuning the feature representation based on the specific problem that I am trying to solve. I simply define the feature representation and then go about finding $\theta$. Even if kernel methods are used, the kernel (or the implicit feature vectors) are chosen, not learned. This is where neural networks are different. They try to findt the optimal feature representation based on the specfic problem that I am trying to solve. But this presents a bit of a chicken and egg problem. I need to know the weights ($\theta$) to get the output of the classification to fine tune the feature representation, but on the other hand, I cannot find the classification output unless I have a good feature representation. But neural networks allow us to solve this problem. The feature vectors are learned jointly with the classifier.


* Below is a schematic diagram of a **neural netwrok unit** (image taken from EdX), which is a primitive neural network that consists of an input layer, and an output later with only one output. It takes the input $x_i$, calculates the weighted sum of them (plus an offset) as $z = w \cdot x + w_0$ and then applies a non-linear function $\hat{y} = f(z)$ to determine the output. The function $f$ is called the activation function. 

    <p align="center">
      <img width="400" src="./notes_images/neural_network_schematic.png">
    </p>

    A commonly used example for the activation function is the ReLu, or the rectified linear unit given by $f(z) = \textrm{max}(0, z)$. Another one is the hyperbolic tangent function, which maps any number $z$ to a number in $(-1, 1)$. 
    
    
* A deep neural network is a neural network that contains many so-called hidden layers between the input layer and the output layer. The number of nodes in each layer is called the width of the NN, and the number of layers is called the depth on the NN. A schematic image is shown below (taken from EdX). It turns out a simple stochastic gradient descent algorithm goes a long way in training the NN and finding the weights of the NN.
    <p align="center">
      <img width="400" src="./notes_images/deepneuralnet.png">
    </p>
    
    One of the big advantages of using a deep neural network is that the layers can extract features from the data almost automatically. For example, detect edges or gradients in images. Usually, the initial few layers detect simple features in the images, while the deeper layers begin to extract the richer features in the image. 
    
    
* The course has an exercise that shows how a simple neural network can be used to implement the basic logical functions such as NAND, NOT and OR. Take two input nodes, and then just chose appropriate weights $w_1, w_2$ and an offset $w_0$ along with the step activation function $H(z) = 0$ is $z<=0$ and $H(z) = 1$ otherwise, and just by inspection we can figure out what the weights and the offset needs to be to implement the NAND. We also know that the NAND function is a kind of universal gate because I can construct all other logical operation using the NAND gate (a property called functional completeness. See here: https://en.wikipedia.org/wiki/NAND_gate)


* We will now use the example of a classifier problem to show what the hidden layer units are actually doing. By finding linear combinations of the input and then applying a nonlinear function to them (essentially what each of the neural network units are doing in a repeated fashion), we are applying nonlinear feature mappings to the data and trying to map it into a higher dimensional space where the input data is linearly separable i.e. linearly separable in $(f_1, f_2)$ space, where $f_1$ and $f_2$ are outputs of two hidden layer units for example. 

**Back propagation algorithm**

* We claimed earlier than neural networks learn both the feature mapping as well as the classifier jointly using a simple variation on stochastic gradient descent. We now address this - the (locally) optimal values of the weights in the model can be found using the back propagation algorithm.


* Consider the figure below (taken from EdX):
    <p align="center">
      <img width="800" src="./notes_images/backprop.png">
    </p>
    
    In general, we are interested in using stochastic gradient descent to find all the weights $w_L$. In this figure $w_L$ is a scalar, but more generally, we can think of $w_{Lj}$ as being a vector that contains all the weight connecting node $j$ in layer $L$ to every preceeding node in layer $L-1$. Now, let us return to the 1D case. To proceed with using stochastic gradient descent, we first start with the output layer. We define the loss function to be:
    $$
    \mathcal{L} = (y - f_L)^2
    $$
    where $y$ is the output label that we are trying to match, and $f_L$ is the output of the output node. Further, let us assume that the activation function is $\textrm{tanh}(\cdot)$. We are interested in finding $\dfrac{\partial\mathcal{L}}{\partial w_i}$ for each weight $w_i$. We start with the outermost node and we have:
    $$
    \dfrac{\partial\mathcal{L}}{\partial w_L} = 2(y - f_L)
    $$
    Furthermore, we also have the following: 
    $$
    \dfrac{\partial\mathcal{L}}{\partial w_L} = \dfrac{\partial\mathcal{L}}{\partial f_L} \dfrac{\partial f_L}{\partial w_L}
    $$
    But we know that $f_L = \textrm{tanh}(z_{L}w_L)$ and hence:
    $$
    \dfrac{\partial f_L}{\partial w_L} = 1 - \textrm{tanh}^2(z)
    $$

## Unit 3: Neural Networks

### Introduction to feedforward neural networks

* Recall the case of using SVMs to do a classification task. Imagine that I am given a vector $x \in \mathbb{R}^d$. I define some feature representation $\phi(x)$ and then find a $\theta$ that determines the classification given the feature representation. But the problem here is that I am not tuning the feature representation based on the specific problem that I am trying to solve. I simply define the feature representation and then go about finding $\theta$. Even if kernel methods are used, the kernel (or the implicit feature vectors) are chosen, not learned. This is where neural networks are different. They try to findt the optimal feature representation based on the specfic problem that I am trying to solve. But this presents a bit of a chicken and egg problem. I need to know the weights ($\theta$) to get the output of the classification to fine tune the feature representation, but on the other hand, I cannot find the classification output unless I have a good feature representation. But neural networks allow us to solve this problem. The feature vectors are learned jointly with the classifier.


* Below is a schematic diagram of a **neural netwrok unit** (image taken from EdX), which is a primitive neural network that consists of an input layer, and an output later with only one output. It takes the input $x_i$, calculates the weighted sum of them (plus an offset) as $z = w \cdot x + w_0$ and then applies a non-linear function $\hat{y} = f(z)$ to determine the output. The function $f$ is called the activation function. 

    <p align="center">
      <img width="400" src="./notes_images/neural_network_schematic.png">
    </p>

    A commonly used example for the activation function is the ReLu, or the rectified linear unit given by $f(z) = \textrm{max}(0, z)$. Another one is the hyperbolic tangent function, which maps any number $z$ to a number in $(-1, 1)$. 
    
    
* A deep neural network is a neural network that contains many so-called hidden layers between the input layer and the output layer. The number of nodes in each layer is called the width of the NN, and the number of layers is called the depth on the NN. A schematic image is shown below (taken from EdX). It turns out a simple stochastic gradient descent algorithm goes a long way in training the NN and finding the weights of the NN.
    <p align="center">
      <img width="400" src="./notes_images/deepneuralnet.png">
    </p>
    
    One of the big advantages of using a deep neural network is that the layers can extract features from the data almost automatically. For example, detect edges or gradients in images. Usually, the initial few layers detect simple features in the images, while the deeper layers begin to extract the richer features in the image. 
    
    
* The course has an exercise that shows how a simple neural network can be used to implement the basic logical functions such as NAND, NOT and OR. Take two input nodes, and then just chose appropriate weights $w_1, w_2$ and an offset $w_0$ along with the step activation function $H(z) = 0$ if $z<=0$ and $H(z) = 1$ otherwise, and just by inspection we can figure out what the weights and the offset needs to be to implement the NAND. We also know that the NAND function is a kind of universal gate because I can construct all other logical operation using the NAND gate (a property called functional completeness. See here: https://en.wikipedia.org/wiki/NAND_gate)


* We will now use the example of a classifier problem to show what the hidden layer units are actually doing. By finding linear combinations of the input and then applying a nonlinear function to them (essentially what each of the neural network units are doing in a repeated fashion), we are applying nonlinear feature mappings to the data and trying to map it into a higher dimensional space where the input data is linearly separable i.e. linearly separable in $(f_1, f_2)$ space, where $f_1$ and $f_2$ are outputs of two hidden layer units for example.


* Note: When you have multiple output nodes, the representation of the weights in the weight matrix is flipped. For example if I have 2 input nodes, and a hidden layer of 4 units, then the weight matrix $W \in \mathbb{R}^{4 \times 2}$ (assuming no bias). If there was a bias term, then it would be $\mathbb{R}^{4 \times 3}$. Can you see why). However, if these 4 hidden units connect two an output layer with two nodes, the the weight matrix $V \in \mathbb{R}^{2 \times 4}$. This is subtle but seems to be an important convention.

**Back propagation algorithm**

* We claimed earlier than neural networks learn both the feature mapping as well as the classifier jointly using a simple variation on stochastic gradient descent. We now address this - the (locally) optimal values of the weights in the model can be found using the back propagation algorithm.


* Consider the figure below (taken from EdX):
    <p align="center">
      <img width="800" src="./notes_images/backprop.png">
    </p>
    
    To determine all the weights in the neural network, we are interested in minimizing the loss $\mathcal{L}$ with respect to all the weights $w_i$. This involves finding all the derivatives $\dfrac{\partial \mathcal{L}}{\partial w_i}$. In this figure $w_L$ is a scalar, but more generally, we can think of $w_{Lj}$ as being a vector that contains all the weight connecting node $j$ in layer $L$ to every preceeding node in layer $L-1$. Now, let us return to the 1D case. For the purposes of this illustration, we will work with a neural network with a width of only 1 unit. Consider the diagram below (taken from EdX). Let us also define the Loss function to be $\mathcal{L} = (y - f_L)^2$, where $y$ is the target output. Furthermore, we will assume the activation function $f(z) = \textrm{tanh(z)}$.

    The overall approach with backpropagation is to find a recursive relation between the partial derivatives $\dfrac{\partial \mathcal{L}}{\partial w_i}$ and $\dfrac{\partial \mathcal{L}}{\partial w_{i+1}}$. The recursion itself proceeds outwards in, that is, it starts from the output layer and then proceeds towards the input. Let us first find the edge case:
    $$
    \begin{align}
    \dfrac{\partial \mathcal{L}}{\partial w_L} &= \dfrac{\partial (y - f_L)^2}{\partial w_i}\\
                                               &= 2(f_L-y)\dfrac{\partial f_L}{\partial z_L}\dfrac{\partial z_L}{\partial w_L}\\
                                               &= 2(f_L-y)\dfrac{\partial \textrm{tanh}(z_L)}{\partial z_L}\dfrac{\partial w_Lf_{L-1}}{\partial w_L}\\
    \dfrac{\partial \mathcal{L}}{\partial w_L} & = 2(f_L-y)(1-f_l^2)f_{L-1}
    \end{align}
    $$

    In general, I can write
    $$
    \begin{align}
    \dfrac{\partial \mathcal{L}}{\partial w_i} & = \dfrac{\partial \mathcal{L}}{\partial f_i} \dfrac{\partial \mathcal{f_i}}{\partial z_i} \dfrac{\partial \mathcal{z_i}}{\partial w_i}\\
    & = \dfrac{\partial \mathcal{L}}{\partial f_i} (1-f_i^2)f_{i-1} \ \ \ \ (\textrm{equation wifi})
    \end{align}
    $$

    Now let us now introduce a recurrence relationship. We rewrite
    $$
    \begin{align}
    \dfrac{\partial \mathcal{L}}{\partial f_i} & = \dfrac{\partial \mathcal{L}}{\partial f_{i+1}}\dfrac{\partial f_{i+1}}{\partial z_{i+1}}\dfrac{\partial z_{i+1}}{\partial f_i} \\
                                               & = \dfrac{\partial \mathcal{L}}{\partial f_{i+1}} (1-f_{i+1}^2) \dfrac{\partial f_i w_{i+1}}{\partial f_i}\\
                                               & = \dfrac{\partial \mathcal{L}}{\partial f_{i+1}} (1-f_{i+1}^2)w_{i+1}
    \end{align}
    $$
    To proceed with the recursion, we start at the outermost layer (the output layer) and hence we need another edge case:
    $$
    \dfrac{\partial \mathcal{L}}{\partial f_{L}} =2(f_L - y) 
    $$
    Now we have everything we need to unpack the recursion and calculate all the derivatives $\dfrac{\partial \mathcal{L}}{\partial w_i}$. The steps involved are to find the edge cases above, then use the recursion to calculate outwards in the derivative of interest $\dfrac{\partial \mathcal{L}}{\partial f_i}$ and finally we use the relationship derived using the chain rule that connects $\dfrac{\partial \mathcal{L}}{\partial w_i}$ to $\dfrac{\partial \mathcal{L}}{\partial f_i}$ (equation wificonnection). Finally, we also need to calculate the case of $\dfrac{\partial \mathcal{L}}{\partial w_1}$ carefully:
    $$
    \dfrac{\partial \mathcal{L}}{\partial w_1} = \left(\dfrac{\partial \mathcal{L}}{\partial f_1}\right)(1-f_1^2)x
    $$
    where the term in the large parenthesis is known from the recursive relations. I then have calculated all the partial derivatives of interest and I can use these in a stochastic gradient descent algorithm to minimize the objective function. If we go through this recursion, we finally end up with:
    $$
    \dfrac{\partial \mathcal{L}}{\partial w_1} = x(1-f_1^2)(1-f_2^2)\cdots(1-f_L^2)w_2 w_3 \cdots w_L (2(f_l-y))
    $$
    Note that this is for the special case of the $\textrm{tanh(z)}$ activation function. Each activation function will have its own specific form of course, but the recursive nature of the backpropagation algorithm will remain the same. 

    **See the Neural Network chapter in Tibhisrani et al. for a formulation of the back propagation algorithm in the more general case where we have weight matrices associated with each layer. Furthermore, see this [article](https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d) for a matrix formulation of backpropagation and how we go about finding the gradients of the loss function. Keep in mind the dimensionalily and the order of pre- and post multiplying matrices when doing the more general case of weight matrices.**

    Note that backpropagation is an algorithm only to find the gradients of the loss function in a computationally efficient manner. After computing the gradients we use stochastic gradient descent to actually find the local minimum (or some other optimization algorithm). 

### Recurrent Neural Networks

* Recurrent neural networks are primarily used to predict time series - for example, what is the stock price tomorrow given some history of price movements, or what is the next word in a sentence given a sentence fragment (in this case, we generalize the notion of a time series as just being a sequence of values)

* Recurrent Neural Networks are especially useful when it comes to making predictions in semantics tasks, such as trying to complete a sentence given a set of words. One way to do this is to take a "moving" subset of words and then use one hot encoding to convert those into features. For example, I could take every pair of words starting from the last known words and move this window along the sentence, and apply one hot encoding for every pair that appears. However, there are problems with this because it is unclear how large the window should be, and sometimes, depending on the context, words can be correlated across very large distances. In these cases, a recurrent neural network is the way to go. What is it?


* This is realy a problem of encoding. In a RNN, we will do continuous vector embedding of objects. The goal would be to take any sort of object - like a sentence, an image, or a sentence, and then turn them into vectors. The encoding can also be reversed, meaning that we can convert back from a vector into words, a sentence, an image. So this actually means that we can convert a sentence into an image, etc. This sort of encoding gives us a lot of power when being used in recurrent neural networks. 


* Some of the more common linguistic tasks done with a RNN is sentence prediction, semantic analysis, and machine translation (from one language to another). 


* The idea here is fairly simple. Refer to the figure below, taken from EdX.

    <p align="center">
      <img width="600" src="./notes_images/RNN_Schematic.png">
    </p>
    
    Let us use the example of wanting to convert a sentence into a vector representation using a RNN. We start with the null vector (i.e. the zero state), and then some new information comes in (represented by the arrow pointing upwards). In this case, the new information coming in is a word in a sentence. The box marked $\theta$ is decribed by a set of parameters $\theta$ and takes in the current state, the new information and produces a mixed output. In effect, the output of each box (or layer) can be written as
    $$
    s_t = \textrm{tanh}(W^{s,s}s_{t-1} + W^{s,x}x_t)
    $$
    where $s_t$ is the current state, $s_{t-1}$ is the previous state and $x_t$ is the new information (or new word in this case, that has been fed in). If the states $s_i$ are $m \times 1$ vectors, and $x_k$ are $d \times 1$ vectors, then we can see that $W^{s,s}$ is an $m \times m$ matrix, and $W^{s,x}$ is a $m \times d$ matrix. In this manner, the recurrent neural network proceeds to incorporate new information and the current state to produce a new state. Note however, that the parameters $\theta$ are the same in all the boxes, i.e., all the layers share the same boxes. 


* There are some key differences between a feedforward neural network and a recurrent neural network. 
    
    *Feed forward neural network vs recurrent neural network*
        * fixed depth vs. variable depth (each word enters the box, so each box is equivalent to a layer).
        * All inputs fed into the input layer vs. input fed in the middle of the network into each layer.
        * Each layer has its own parameters vs. all layers share the same parameter.
        

* One of the issues in a recurrent neural network is that each update of state completely overwrites the previous state. While the mixed vector still has some memory of the previous state because of the linear combination nature of calculating new states, it still could very quickly erase the previous state. It is unclear to me at this point, but this somehow leads to problems such as zero gradients or exploding gradients. Moreover, there is a paper by [Begio et al.](http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf) that says that it is difficult for a recurrent neural network to learn the parameters $W^{s, s}$ and $W^{s,x}$. It is possible for a human to sit and carefully modify the all the parameters, but this is not the point. To avoid this, the concept of gated recurrent neural networks is introduced, where we define some abstract concepts of gates, which essentially meter out how much of the current state we want to remember, and meter in how much of the new information we want to include. For example, in a simple gated RNN, we introduce a gate vector $g_t$ that is of the same dimensionality as the state vector $s_t$ as
    $$
    g_t = \textrm{sigmoid}(W^{s,s}s_{t-1} + W^{s,x}x_t)
    $$
    Next, we meter out the relative influences of the previous state and the new information by using the gate $g_t$ as
    $$
    s_t = (1-g_t) \odot s_{t-1} + g_t \odot \textrm{tanh}(W^{s,s}s_{t-1} + W^{s,x}x_t)
    $$
    where the symbol $\odot$ represents the Hardamard product, or element-wise multiplication (similar to broadcasting in `numpy`)
* Another approach is to use a **Long Short Term Memory RNN (LSTM)**. Here we define some conceptual gates that meter out how much memory to keep of the previous state, and how much importance to give to the new information coming in.  Let us make this more concrete. We introduce the following gates:
    * The forget gate: $f_t = \textrm{sigmoid}(W^{f,h}h_{t-1}+W^{f,x}x_t)$
    * The input gate: $i_t = \textrm{sigmoid}(W^{i,h}h_{t-1}+W^{i,x}x_t)$
    * The output gate: $o_t = \textrm{sigmoid}(W^{o,h}h_{t-1}+W^{o,x}x_t)$
    * The memory cell: $f_t \odot c_{t-1} + i_t \odot \textrm{tanh}(W^{c, h}h_{t-1} + W^{c,x}x_t)$
    * The visible state: $h_t = o_t \odot \textrm{tanh}(c_t)$
    

### Markov Models

* As we saw previously, some of the most major applications of recurrent neural networks is in language and linguistics based applications. The most common tasks here are in predicting the next word in a sentence, determining the sentiment of a sentence, or doing machine translation where we are translating from one language to another. Here we will exemplify the utility of a Markov model to perform prediction of the next word in a sentence tasks. 

    A $k$-th order Markov model is a model that predicts the next word in a sentence given the previous $k$ words it has seen so far. We will exemplify the utility of this model using the first order Markov model. We could imagine that we have a vocabulary $V$ that consists of a set of words $w$. The goal here is to predict word $w_{i}$ in a sentence given word $w_{i-1}$. In the spirit of a Markov model, if we had all the transition probabilites from one word to the next (say in the form of a look up table), then we could calculate the probabilities of different sentences simply by referring to the table. We would simply have to look up $\mathbf{P}(w_i\lvert w_{i-1})$. But now the question is how do we construct this table? In effect, let's assume that we have a corpus of sentences $S$, and a whole bunch of sentences $s \in S$. To build the table, what we are looking for is to maximize the probability of seeing sentences given all the possible words in our corpus $S$. That is, we are interested in maximizing the probability:
    
    $$
    \prod\limits_{s \in S}\left(\prod\limits_{i=1}^{\lvert s \rvert}\mathbf{P}(w_i^s \lvert w_{i-1}^s)\right)
    $$
    In effect, this is a case of maximum likelihood estimation. So we can use the usual trick of maximizing the log likelihood instead of the likelihood (so we turn all products into sums). If we do, we find that the solution to the transition probability $\mathbf{P}(w, w')$ of going from word $w$ to word $w'$ is just a count of how many times that particular transition occurs in the corpus as compared to all other transitions from $w$, appropriately normalized. That is
    $$
    \mathbf{P}(w, w') = \dfrac{\textrm{count}(w \rightarrow w')}{\sum_{\tilde{w} \in S}\textrm{count}(w \rightarrow \tilde{w})}
    $$
    This is a pretty intuitive result and what I would have expected.
    
**Converting the Markov model for word prediction into a Neural Network**

* We can now also convert the above Markov model into a feedforward neural network. The advantage here is that the Markov model very quickly grows in complexity with the size of the vocabulary. Assume that we have $v$ words in the vocabulary. Also assume that we want to use a trigram model (i.e. we want to use the preceeding two words to determine the next word in the sequence). This means that we have $v^2$ choices for word pairs, which can then transition to $v$ words, and hence there are $v^3$ transition probabilities. This is humongous. Converting this transiton probability architecture into a feed forward neural network architecture greatly decreases dimensionality. Why? It's because I can use a trick like one-hot encoding. 

    Consider again the situation that you want to use the trigram model. We can encode each of these words using a one-hot vector of the form $[0\ 0\ \ldots 1\ 0 \dots 0]$ where the position of the 1 indicates that that particular word of the corpus is present. So this has dimensionality $v$ and hence for two words, the total dimensionality is $2v$. So we can assign a node to each of the components of these vectors, and hence we have $2v$ nodes. Similiarly, we also have $v$ output nodes, whose value can correspond to the probability that the word succeeding the previous two words is that particular word i.e. the value of the output node $p_k$ is the probability that word $w_i = k$ is the succeeding word, given words $w_{i-1}$ and $w_{i-2}$ are the succeeding words. Now if we do not use a hidden layer and no further output transformations, we want to connect these $2v$ nodes to the output nodes through a set of weights and biases in the neural network architecture. Moreover, because the outputs are proabilities, they have to respect the rules of probabilities such as always being non-negative and all outputs summing to one. All told, we have the following neural network architecture:
    $$
    z_k = \sum_j x_j w_{jk} + w_{0k}\\
    p_k = \exp(z_k)/\sum_j\exp(z_j)
    $$
    
    This trick of using softmax to convert things into probabilities is really nifty. Because now I can naturally have maximum likelihood estimators. 
    
    
* So we have seen that the feed forward neural network offers a nice alternative to solve the Markov model for word prediction in a somewhat more manageable way. However, we had to pick the value of $n$ and chose an $n$-gram model. In the above example we chose $n=2$. But this is too restrictive. We want to have more flexibility and have the model chose what level of memoery to use and how far back into history it must use. Therefore RNNs offer this possibility because there is this recursive element to it, where $s_t$ clearly depends on $s_{t-1}$ all the way until $s_0$. In effect, RNNs have a hidden state that is fed both by the input information $x_t$ and the previous state $s_{t-1}$. The output layer just adds the softmax function to the hidden layer i.e. $p_k = \textrm{softmax}(W^o s_t)$.  Note that Chris Olah in his blogpost on  LSTMs states that regular RNNs are not so great at learning parameters and LSTMs do much better in practice. So it seems that the more complicated structure is preferrable.


* One interesting thing about using a recurrent neural network is that I just need to be able to feed in a vector. So for example, I can feed in a vector that is a result of a convolutional neural network and then feed that into an LSTM and generate auto-captions for the image. This is really cool, and really worth implementing on my own. That way, I will learn about convolutional neural networks as well as LSTM. 


* Read Andrej Karpathy's blogpost [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). It's a really cool application of recurrent neural networks. 


### Convolutional Neural Networks

* Convolutional Neural Networks offer a way to efficiently employ the architecture of feed-forward neural networks to image processing and associated object classification tasks. Consider the case where we have a picture (say of a mushroom in an otherwise green field). Presumably, one of the classifications of this picture would be "mushroom". However, consider a training set where we have perhaps a million images with thousands of class labels. Consider also that each image is a 1000 x 1000 pixels large (so not even the greatest resolution of this image). The regular feed forward neural network architecture to go about making class predictions is as follows. We would have an input node corresponding to the value of each pixel so the input layer has dimensionality $10^6$. Now, if we were to connect this to a hidden layer, also with dimensionality $10^6$, then the total number of weights we need for just this one layer is $10^6 \times 10^6 = 10^{12}$. Which is atrociously large. So even for small images, this naive approach quickly becomes infeasible. What's more, if the network is trained ona set of images with mushroom, and now I feed in an image that is very similar except that the mushroom is slightly translated, then the network would choke up and it would not make good predictions. Convolutional neural networks enable us to overcome these issues of exploding number of weights as well as extracting features of images without necessarily remembering where in the image the feature was located.


* At it's core, a convolutional neural network converts an image into a "feature map". The feature map is obtained by first converting the image into a series of "patches" i.e. sub-images of which the image is composed of. In additon, we define a "kernel" which is a matrix of weights (which are initialized randomly but learned as as we train the network). We then pick a stride size and stride or raster across the image with this stride step, replacing each patch with a number. Therefore, this process reduces the dimensionality of the image (see below) as well as uses the same weight kernel for each stride i.e. each feature map shares the same weights. 

    We proceed by replacing each patch by first finding the Hadamard product of the filter with that particular stride, finding the sum of the resulting hadamard product matrix, and then applying some suitable non-linear activation such as ReLU. We then move to the next stride and repeat this process. At each raster step, i.e. each stride, we perform a Hadamard product between the filter (or feature map) and the patch of the image, then sum all the elements of that matrix, and apply an activation function to this scalar (such as ReLU, which is very commonly used). This process of doing the Hadamard product is a convolution operation. We can have as many feature maps (or "channels") as we wish, each of which extracts a different feature of the image - this could be edges, or color, or shape, etc. These convolutions followed by non-linear activation is done in a repetitive manner across many layers. This is akin to the feed-forward architecture. One of the big differences is that in each layer and channel, as the feature map is being rastered across the image, the weights are not changed - it remains the same. Moreover, these weights and filters are completely self learned; we start with randomly initialized weight matrices and then apply back-propagation followed by stochastic gradient descent. The random initialization leads the feature maps to learn different aspects of the image automatically.


* A note about the dimensionality: let's say we have an input image of $n \times n$ pixels. Let the dimensionality of the feature map be $p \times p$. Let us also have a stride length $s$. Note that we need to choose appropriately because we require an integer number of moves of the feature map across the image. Let us assume that we can make $m$ moves of the feature map of size across the image. Then we require that $(n - p)/s$ be and integer. Moreover, the dimensionality of the convolved layer would be $(n - p)/s +1$ (i.e. first image plus one image for every stride). 


* By using the same weight matrix, we are essentially looking for a particular feature all across the image - we are looking for how much of the feature is present in each stride of that image and extract those features.


* The main difference between the feed-forward neural network and the CNN is that (i) in the CNN, every patch sees the same weights, so there is sharing and (ii) each element in the feature map matrix connects only locally to each patch. There is no cross talk between patches.


* One of the important goals of a convolutional neural network is to extract features into feature maps without actually remembering where in the image that feature was present. This helps when trying to identify objects in an image regardless of where exactly in that image the object is present. This is called **Pooling.**


* Pooling proceeds as follows: After the convolution layer, we introduce a pooling layer (say of dimensionality) $l \times l$. We again raster this window, collapsing $l \times l$ windows of the convolution layer into a single number. This single number is usually the maximum value of the elements in that window. Note that the pooling layer does not have any weights. It is just a moving window where I throw out spatial information.

## Unit 4: Unsupervised Learning

### Clustering

* In clustering, we seek to determine underlying meaningful structure in unlabelled data. We are given only the feature vectors $x^{(i)}$ and no associated labels. One example is in Google News - given a news article, al algorith automatically determines other 'related' news articles that are close. It does this by first constructing feature vectors by using something like a bag-of-words approach for example, and then using some similiarity measure between these feature vectors (for example cosine similarity). Then we need a clustering algorithm that places points in appropariate clusters given these similarity measures. **Idea**: Can I use this for predicting solubility based on just knowing some sort of polarity or electronic surface charge density of molecules, and then given some knowledge of what these known molecules are soluble in, can I use a clustering algorithm to predict the solubility of new molecules. 


* Another example of using clustering is in image quantization or image compression. Let's say that I have an image that is 1024 x 1024 pixels large. We also have three color channels RGB, each with 8 bits (for 256 color levels each) for a total of 24 bits. This means that the total number of bits required to fully represent this image is 1024 x 1024 x 24 = 3 million bits. If I were to now use clustering, by mapping this 24 component color feature vector to say 32 colors (which can be represented by 5 bits), then to save an image, I just need to use 1024 x 1024 x 5 bits. In addition, I need to remember my map, where for each feature vector (of which there are 24), I need to remember 5 bits, and hence my map needs a memory of 24 x 32 bits (each color in the 24 bit representation maps onto one of 32 colors). Therefore, my total memory required is 1024 x 1024 x 5 + 24 x 32, which is now much smaller. I give up on quality but gain on memory, depending on the number of clusters I decide to use (in this example, 5). This is a sort of trade-off in clustering algorithms. 


* There are two ways to think about clustering. One way is to think of partitioning our training set into $K$ partitions $C_1, C_2, \ldots, C_K$ such that all partitions are non empty. Each partition does NOT store the training set, but rather stores the index of the training set such that $\bigcup\limits_{j}C_j = \{1, 2, ..., n\}$. Moreover, in hard clustering, a training example cannot be located in two different clusters simultaneously i.e. $C_i \bigcap C_j = \phi$ for $i \neq j$. 

    Another way to think about the output of a clustering algorithm is to think of it in terms of **representatives** $z_1, ..., z_K$. For example, if I have $K$ clusters, I could choose the centroid of each of the clusters as the representative. These two ways of thinking about the output of a clustering algorithm are equivalent, as will become clearer below. 
    
    
* Because there are different and equivalent ways to perform clustering, we need some objective measure of the cost of a clustering operation so that we can then compare the different clusters and then pick the best one based on this cost. The key idea here is that the cost of the entire clustering operation can be written as the sum of the individual costs of each cluster. That is,
    $$
    \textrm{Cost}(C_1, ..., C_K)= \sum_j \textrm{Cost}(C_j)
    $$
    Now the problem reduces to calculating the cost of each individual cluster. There are multiple ways to do this: (i) Cosine distance (which is independent of the length of the feature vectors) (ii) Diameter of the cluster, or the distance between the two most extreme points, (iii) Euclidean distance between all pairs of vectors.
    
    
* What works better in practice is to find the representative for each cluster and then calculate the sum of distance measure between each point in the cluster with the representative. Therefore, we can now write that the total cost of the clustering operation is given by
    $$
    \textrm{Cost}(C_1, ..., C_K, z^1, ..., z^k) = \sum\limits_{j=1}^K \sum\limits_{i \in C_j}\lVert x^i - z^j \rVert^2
    $$
    

* The K-means algorithm proceeds as follows:

    1. Randomnly assign $K$ points as being representatives of the $K$ clusters required. There are some strategies on how to go about assigning these clusters, because some bad initialization can lead to really bad results. We would ideally like to have these initial representatives somwhat far apart in the high-dimensional space. In practice, another way to ensure insensitivity to initial choice of clusters is to try many different initilizations and pick the output with the lowest cost. 
    
    2. Iterate over the following two steps:

        1. Assign each point to the representative closest to it. That is, do the assignment so that
            $$
            \textrm{Cost}(z_1, ..., z_K) = \sum_i\textrm{min}_{j=1, ..., K}\lVert x^{(i)} - z_j \rVert^2
            $$
        2. Given the clusters $C_1, ..., C_K$, find new representatives $z_1, ..., z_K$ so that
            $$
            z_j = \textrm{argmin}_z \sum_{i \in C_j}\lVert x^{(i)} - z \rVert^2
            $$
            
    It turns out that when we carry out this minimization in 2(b) by finding the gradient, we find the intuitive result that $z_j$ is the centroid of all the points $x^{(i)}, i \in C_j$. Note that this result is only true if we use the Euclidean distance as the similarity metric. If we used a different similarity metric, we would have arrived at a different answer. 
    
    
### K -Medoids algorithm

* K-means has two limitations. The first is that depending on the distance metric used, it is possible that the representatives are not part of the original points $x$. This is okay in some applications, but in other applications, this would be problemati - for example in the Google news example, we would end up with a representative news story that does not perhaps exist. The other liminations is that the assignemnt of new representatives as being the centroid of the points of the cluster only works if the squared norm distance is used. If some other similarity metric is used, then the algorithm presented above would not work. We therefore seek an algorithm that is general enough to accomodate the above two issues.


* The K-medoid algorithm seeks to overcome both these limitations. The algorithm is different from K-means in step 1 and step 2(b). The algorithm is as follows:

    1. Randomnly assign $\{z_1, ..., z_K\} \subset \{x^{(1)}, ..., x^{(n)}$ as being representatives of the $K$ clusters required. Note the difference with the K-means here - we force the initial representatives to be part of the intial set of points. We would again need to try many different initilizations and pick the output with the lowest cost. 

    2. Iterate over the following two steps:
        1. Assign each point to the representative closest to it. That is, do the assignment so that
            $$
            \textrm{Cost}(z_1, ..., z_K) = \sum_i\textrm{min}_{j=1, ..., K}\textrm{dist}(x^{(i)}, z_j)
            $$

        2. Given the clusters $C_1, ..., C_K$, find new representatives $z_j \in \{x^{(1)}, ..., x^{(n)}\}$ so that
            $$
            \sum_{x^{(i)} \in C_j}\textrm{dist}(x^{(i)}, z_j), j=1, ..., K
            $$
            is minimal.
            
            
* The tradeoff of using K-medoids vs. K-means is that the comutational complexity of K-medoids is much larger. It is $O(n^2Kd)$ vs. $O(nKd)$ where $n$ is the number of points, $K$ is the number of clusters and $d$ is the dimensionality of the data. So I will have make choices appropriately. 


* Although we use the term unsupervised, we are actually giving a lot of indirect information to the algorithm, by using by deciding on the distance metric, deciding on the number of clusters, etc. 


### Generative Models


* Consider the support vector machine. This is an example of a discriminative model, where there is a classifier that discriminates between data points. The discriminative model learns the decision boundary between the classes. Another approach is probabilistic, where we try and fit a probability distribution to the data and then when we see a new data point, we can calculate the probability of the new data point belonging to one or the other class. This is an example of a generative model. There are two chief ideas that we will be concerned with in a generative model: (1) The estimation problem i.e. fitting of the probability distribution and (2) the prediction problem i.e. calculating probabilities that the data point belongs to a certain class. 


* Two generative models that we will see in this class: multinomial model and Gaussian model. 


* We will use the example of generating the next word $w$ in a sequence of words, given a vocabulary $W$. Let us denote the probability of generating $w$ given the parameters of the model $\theta$ (which might include all the words in the sequence so far) by $P(w \lvert \theta) = \theta_w$. Because this is a probability, it has to respect the rules of $\theta_w \geq 0$ for all $w \in W$ and $\sum_{w \in W} \theta_w = 1$. 


* Let's say we want to generate a document $D$ that consists of the words $\{w_1, w_2, w_1, w_2\}$. Let's also say that we have two models with parameters that describe the probability of generating word $w_i$. Let these parameters be $\theta_{w_i}$ and $\theta'_{w_i}$ respectively. I can evaluate the probability that model 1 with parameters $\theta$ and model 2 with parameters $\theta'$  generated $D$. This is easily seen by calculating for each model
    $$
    P(D\lvert \theta) = \prod\limits_{w \in D}\theta_{w}^{\textrm{count}(w)}
    $$
    I can now compare $P(D\lvert \theta)$ and $P(D\lvert \theta')$ and use maximum likelihood estimation to decide which model actually generated the data. 
    
    
* Now, if I am given  a document as a training set and I want to find all the individual $\theta_{w_i}$, it can be easily shown with maximum likelihood estimation that 
    $$
    \theta_{w_i} = \dfrac{\textrm{count}(w_i)}{\sum_{w_i}\textrm{count}(w_i)}
    $$
    This is also the intuitive result of the multinomial model.
    
    
* Now, if we look at the problem of prediction, we could formulate the problem in the following way. We essentially are interested in determining given a particular document $D$ whether it was generated by the model with parameters $\theta^+$ or the model $\theta^-$. To answer this, we can consider the ratio 
    $$
        \log\left(\dfrac{P(D \lvert \theta^+)}{P(D \lvert \theta^-)}\right)
    $$
    If this quantity is bigger than zero, then the model with parameters $\theta^+$ generated the data. If this ratio is less than 0, then the model with parameters $\theta^-$ generated the data. Note that this looks somewhat like the likelihood ratio test, except that I haven't constrained any of the values of $\theta$ to be anything. I could now go ahead and write out expressions for the quantities $P(D \lvert \theta^+$ and $P(D \lvert \theta^-$ to finally arrive at the above logarithm to look like $\sum\limits_{w \in W}\textrm{count}(w)\log\dfrac{\theta^+_w}{\theta^-_w} \geq 0$ i.e. $\sum\limits_{w \in W}\textrm{count}(w)\log\tilde{\theta}_w \geq 0$ for the test example to be classified as positive, and $\sum\limits_{w \in W}\textrm{count}(w)\log\tilde{\theta}_w \leq 0$ for the test sample to be classified negative. This looks exactly like the condition for the linear classification problem if the sum were to be written as an inner product. So the generative model is just another way to approaching the linear classification problem, at least in the case of the multinomial model.
    
    
* If we include prior knowledge into this formulations using a Bayesian approach we will find that we get the criterion of a linear classifier with an offset. If we were apply Bayes rule to the quantity $P(D \lvert \theta^+)$ we get
    $$
    P(y=+ \lvert D) = \dfrac{P(D \lvert \theta^+)P(y=+)}{P(D)}
    $$
    Therefore,
    $$
    \log\dfrac{P(y=+ \lvert D)}{P(y=- \lvert D)} = \log\dfrac{P(D \lvert \theta^+)P(y=+)}{P(D \lvert \theta^-)P(y=-)}\\
    = \log\dfrac{P(D \lvert \theta^+)}{P(D \lvert \theta^-)} + \log\dfrac{P(y=+)}{P(y=-)}\\
    = \sum\limits_{w \in D}\textrm{count}(w)\tilde{\theta}_w + \tilde{\theta}_0
    $$
    which is the exactly the linear classification problem with an offset.
    
    
* The two most common generative models are the multinomial model and the Gaussian model. The approach here is basically maximum likelihood estimation. Given all the data, what are the parameters of the multinomial or the Gaussian that generated that data? Now given the parameters - solved through maximum likelihood - we can make predictions of future labels given the parameters. We just calculate the probability that each model (characterized by the set of parameters) generated that data point and then pick the label of the model that has the highest probability. Think intuitively in terms of fitting Gaussians or multinomials to each label and this should become more clear. 


* We can extend this idea of multinomial and Gaussian generative models by considering **mixture models**. Here, we assume that we have a set of $K$ Gaussians each with parameters $\theta_K = \{\mu_K, \sigma^2_K\}$ that could have generated the data, with the probability of each model having generated the data being given by multinomial probabilities $\pi_1, ..., \pi_K$. In other words, we can define the probability that a certain point was generated by a particular component $k$ of the mixture 
    $$
    P(y=k \lvert x^{(i)}, \theta_K) = \dfrac{ P(x^{(i)} \lvert y = k)P(y=k)}{\sum_k P(x^{(i)} \lvert y = k)P(y=k)}\\
     = \dfrac{\pi_k \mathcal{N}(x^{(i)}; \theta_k)}{\sum_k \pi_k \mathcal{N}(x^{(i)}; \theta_k)}
    $$
    
### The Expectation Maximization (EM) algorithm

* The EM algorithm is an iterative way to apply maxiumum likelihood estimation. The algorithm consists of two steps: The E step, where we assign points to each of the clusters by calculating the soft counts of each point given a cluster. i.e. we calculate 


                        **SECTION NEEDS TO BE FILLED IN DO AT THE END OF THE CLASS**


## Unit 5: Reinforcement Learning

* The idea in reinforcement learning is to have an agent that learns to maximize a reward given that it take many paths to achieve its goal. We don't necessarily provide a reward at each step of the path, but only provide it with a reward if it achieves it's goal. An example is a robot that has to navigate a grid of squares with a hazard in the middle. If it reaches the top right square, it gets a reward of +1 but if it reaches the square below, it receives a reward of -1. 


* We essentially need to understand three different things: Markov Decision Processes, Bellman Equations (which allow us to propagate the reward back to the paths through the various states), and the value iteration algorithm. 


* Terminology: **The Markov decision process** has the following terminology: State $s \in S$, action $a \in A$, transition $T(s, s', a) = P(s' \lvert S, a)$ i.e. the transition probability of ending up in state $s'$ given that you are in state $s$ and that you take action $a$ while in state $s$ and reward $R(s)$. In general, it can be $R(s, a, s')$. Note that $\sum_s' P(s' \lvert s, a) = 1$. Therefore, the Markov decision process is characterized by this set of $\langle S, A, T, R \rangle$. The utility function keeps track of the rewards that are bailed out as the agent goes through the different states.


* When we discuss rewards, we need to discuss how we pay these rewards out. Consider again the case where the rewards only depends on the current state we are in. If we keep revisiting this state, we will get infinite reward, and then it becomes impossible to compare paths based on these rewards because we will be comparing infinities. We therefore use a discounted utility function where rewards of early states are values more highly and then we value future rewards less and less. This is very much like how humans behave. i.e we say that the the utility function $U(\{s_0, s_1, ...\}) = R(s_0) + \gamma R(s_1) + ...$, where $0 \leq \gamma \leq 1$. Note that we can show quite easily that this utility function is now bounded and $(\{s_0, s_1, ...\}) \leq R_\max/(1 - \gamma)$. The other alternative is final Horizon based utility, where we say that only the rewards in the first $N$ steps matter. But this would insert artifacts in the actions if we are in step $N-1$ because the agent might then act greedily to get the immediate highest reward. On the other hand, with discounted utility, the agent can allow moving to areas of higher rewards while taking an immediately lower reward. 


* The final bit of terminology that we will need here is the notion of policy. The policy $\pi^*: s \mapsto a$ tries to maximize the expected utility by suggesting the best possible action $a$ given that I am in state $s$. Associated with this notion is also the notion of the value of a state. Rewards are not present in every state, but states close to the ones having rewards are also somehow valuable. So we need some mechanism to propagate the value of states across different states. This is what the Bellman equations do for us. 


* We now introduce some notation that becomes very important to understand to carry out reinforcement learning problems. The first is the notion of the value of the state $V^*(s)$, where the $*$ denotes that it is an optimal number. $V^*(s)$ is the expected reward starting from state $s$ and then acting optimally thereafter. Another piece of terminology we need is $Q^*(s,a)$, is the expected reward if we start at state $s$, perform action $a$ and then act optimally thereafter. Clearly it is easy to see that
    $$
        V^*(s) = \underset{a}{\max} Q^*(s,a)
    $$
    and
    $$
        \pi^*(s) = \underset{a}{\textrm{argmax}} Q^*(s,a)
    $$
    
    
* We can now use a recursive relation (similar to Markov chains) and the total expectation theorem to show that
    $$
    Q^*(s,a) = \sum_{s'}T(s, a, s')\left( R(s, a, s') + \gamma V^*(s') \right)
    $$
    In other words, it is the expected reward given that I go to state $s'$ from state $s$ on taking action $a$ multiplied by the probability that I will end up in state $s'$ starting at state $s$ under action $a$ and then summing over all such possible states $s'$. This can further be simplified from the relationship between $V^*$ and $Q^*$ to obtain
    $$
    Q^*(s,a) = \sum_{s'}T(s, a, s')\left( R(s, a, s') + \gamma \underset{a'}{\max}Q^*(s', a') \right)
    $$
    The above recursive relationship is called the **Bellman Equations**. 
    
    
* To actually solve this equation in practice, we use the value iteration algorithm (if we we use the expression involving $V^*$) or the $Q$ value iteration algorithm (if we are using the expression involving $Q^*$). Below we only discuss the Q value algorithm but the value iteration algorithm proceeds very similarly.

    1. We first initialize $Q^*(s,a) = 0\ \forall\ s, a$. 
    2. We iterate the following until convergence:
        $$
        Q^*_{k+1}(s,a) = \sum_{s'}T(s, a, s')\left( R(s, a, s') + \gamma \underset{a'}{\max}Q_k^*(s', a') \right)
        $$
        
* There are however some problems with this approach. It assumes that I know the reward structure for all possible states under all possible actions and the transition probabilities between all possible states under all possible actions. However this information is not always provided in the real world. So we have to resort to using our statistical hammer of replacing our expectation with the emperical mean using the weak law of large numbers. All I need to assume in this case is that I know the set of all states $S$ in my problem are, and the set of all actions. 


* Even with this there are problems because in a complex state space, there is no way I can collect enough information to calculate the reward structure and the transition probabilities because I will probably visit each transition quite rarely. This is essentially a statement about saying that my sample size is unlikely to be large enough for the weak law of large numbers to hold. Therefore my estimated transition probabilities and reward structure is likely to be very noisy. 


* We get over this by apply a Q value iteration algorithm for reinforcement learning, rather than for a Markov Decision Process. The key idea here is that I can write an average as
    $$
    \bar{x}_n = \dfrac{x_1 + \cdots +x_n}{n} = \dfrac{x_1 + \cdots +x_{n-1}}{n-1}\cdot \dfrac{n-1}{n} + \dfrac{x_n}{n}\\
    = \bar{x}_{n-1}(1 - 1/n) + (1/n)x_n \\
    = \alpha x_n + (1-\alpha)\bar{x}_{n-1}
    $$
    where $\alpha=1/n$ but in reality we just fix alpha to be some small number. This running average is called the exponential running average because if we unpack the the recursion above, it would be clear to see that there is an exponential type relationship occuring. Therefore, extending this idea, we can write
    $$
    Q^*_{i+1}(s,a) = \alpha \cdot \textrm{sample} + (1-\alpha)Q^*_i(s,a)
    $$
    
    
* With this in mind, we can now rewrite the Q-value iteration algorithm as follows:
    1. Initialize $Q^*(s,a) = 0 \ \forall\ s,a$
    2. Iterate until convergence:
        1. Collect sample $s, a, s', R(s, a, s')$. That is, starting from the current state $s$ perform action $a$, see which state $s'$ we ended up in, and measure how much reward we got while doing that step. 
        2. Calculate $Q^*_{i+1}(s, a) = \alpha \cdot \textrm{sample} + (1-\alpha)Q^*_i(s,a).$ Substituting for the sample, we obtain 
        $$
        Q^*_{i+1}(s, a) = \alpha (R(s, a, s') + \underset{a'}{\max}Q_i^*(s', a')) + (1-\alpha)Q^*_i(s,a)
        $$
        
        
* If you rerrange the above equation, by grouping all the alpha terms, you will see that this looks very much like the stochastic gradient descent update rule, with $\alpha$ playing the role of the learning rate!


* Remind yourself of $\varepsilon$-greedy. 


* Read about Q-learning with linear approximation from the resources that have been summarized here.


### Natural Langugage Processing

* Natural Language Processing (NLP) includes various sub-fields such as machine translation, human understanding, responding to questions, speech recognition, etc. 
        


## Other random pieces of information I have learned along the way

### `*args` and `**kwargs`
* `*args` unpacks a list and allows you to pass a list of arguments to a function. Consider the following example:
    ```
    def fun(*args):
        for arg in reversed(args):
            print(arg)
    fun('a', 'b', 'c')
    ```
    This produces the output
    ```
    c
    b
    a
    ```
    Therefore, we have been able to pass a random number of arguments and do use them in the function. However this is of limited use because it requires us to know the specific order in which the arguments were passed for us to be able to do something with this. This is where ``**kwargs`` comes handy. the ``**`` operator can unpack dictionaties. The nice thing about this is that we can now pass named arguments and we don't have to remember the order anymore. Consider the following example:

    ```
    def fun2(**kwargs):
        y = kwargs['y']
        x = kwargs['x']
        print(x**y)
    fun2(x=2, y=3, z=5)
    ```
    Note also that I didn't have to use all the arguments that I gave. I just used what I needed references with the label of the dictionary. The ``**kwargs`` keyword took the arguments passed to the function, converted it into a dictionary and then I could appropriately get their values without having to worry about what order they were passed in because I can find the values from the key.
    

* Identiy matrix in `numpy`: `numpy.identity(n)`


* The softmax function $\sigma: \mathbb{R}^d \mapsto \mathbb{R}^d$ is defined as:
    $$
    \sigma(\boldsymbol{z}) = \dfrac{1}{\sum\limits_i e^{z_i/\tau}}{e^{z_i/\tau}}
    $$
    The paramter $\tau$ is called the temperature parameter. It takes a vector $z$ and converts each component into a probability that lies between 0 and 1. Sometimes these exponentials can get really large. In this case, we can subtract an arbitrary constant $c$ from the argument of the exponental, where $c$ is usally chosen to be $\max {z_i/\tau}$.  


* `np.clip(a, low, high)` clips the values in a matrix `a`. Any value lower than `low` will be assigned `low` and similarly for high.


* Check out `scipy.sparse.coo_matric()` which allows me to build sparse matrices by passing a set of values to be filled in at particular coordinates that is also passed. 






## `pytorch` tutorial



* The goal of `pytorch` is to serve both as a deep learning platform as well as a substitute for `numpu` so that I can use GPU processing. This might potentially be very interesting. 


* A lot of the `numpy` constructors for matrices already work: for example `torch.zeros`, `torch.empty`, `torch.rand`. To convert a list to a tensor, I can use `torch.tensor([..])`. 


* To find the shape of a tensor `x`, I can either use `x.shape` or I can use `x.size()`. The returned value is in fact a tuple so it supports all the tuple operations.


* There are many operations in `torch` that can be done in place. For example, I can add two tensors by just doing `x + y` for example. Or I can also do `x.add(y)`. The neat thing now is that if I want to do in place addition, I can use `x.add_(y)`. Often, using a method with an underscore means that it does an inplace operation.


* To resize a tensor, use `torch.view()`. Something cool that even works in `np.reshape()`. If I were to use one of the dimensions as -1, it's shape is automatically inferred from the shaped of the matrix. For example, if I had a tensor of size 11 x 4 and I reshaped (view) to (2, -1), it would automatically caluculate that this tensor/array should have 22 columns.


* To convert a tensor into a numpy array, use `torch.numpy(a)`. Note that the resulting array and the original tensor share memory locations, so changing the value of one will change the value of the other. To convert from a numpy array to a tensor, use `torch.from_numpy(a)`

**Autograd differentiation**

* The `torch.Tensor` is the most fundamental class of the package. If the attribute `.requires_grad` is set as `True`, torch begins to track all the operations done on the tensor. Then at the end, by calling `.backward()`, all the gradients are computed automatically and accumulated into the `.grad()` attribute. To prevent this kind of tracking, which can blow up memory, the code block can be wrapped in `with torch.no_grad()`. This is particularly helpful for models which have many parameters with `.requires_grad=True` but for which we don't need the actual gradients. The other class that turns out to be very important is the `Function` class. `Tensor` and `Function` are interconnected and build up an acyclic graph that encodes a complete history of computation. Each tensor has  `.grad_fn` attribute that references a `Function` that has created the `Tensor` except for Tensors created by the user - their `grad_fn` is `None`.


* To calculate the gradient of a tensor, I just need to call `.backward()` on a tensor. If the tensor is not a scalar, I also need to provide a `gradient` argument that is a tensor of matching shape. Below are some examples


```python
import torch

x = torch.ones((2, 2), requires_grad=True)
y = x + 2
print(y)
```

    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)
    


```python
z = y*y*3
out = z.mean()
print(z, out)
```

    tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward1>)
    


```python
out.backward()
print(x.grad)
```

    tensor([[4.5000, 4.5000],
            [4.5000, 4.5000]])
    

This can be understood using the following:
$$o = (1/4)\sum_i z_i$$
$$ = (1/4)\sum_i (x_i + 2)^2$$
and hence
 
$$\dfrac{\partial o}{\partial x_i} = (3/2)(x_i+2)$$

`torch.autograd` essentially calculates the vector-Jacobian product. If the output is a vector/tensor, then I need to supply to `y.backward()` another vector of the same shape i.e I need `y.backward(v)`. Then I can continue on and use `x.grad` which acutally calculates the vector jacobian product $J^T \cdot v$, where the Jacobian matrix is $J_{ij} = \partial y_j/\partial x_i$. If $v$ is athe gradient of some scalar function, then by the chain rule, the quantity $J^T \cdot v$ is the gradient of the scalar function with respect to $x$. Work this out for yourself. 

### Other things to learn

1)      Random forests

2)      Multiple linear regression

3)      Bayesian Linear  Regression and how Gaussian Process Regression is the kernelized variant of Bayesian Linear Regression.

4)      Bootstrapping and bagging
5) Bayesian Neural networks (See the Neural Nets chapter in ESL). 

6) See [this MIT course, 6.S191](www.introtodeeplearning.com) that seems more advnaced and covers some additional topics compared to the EdX course. 

7) Both pytorch as well as tensorflow at least as applies to neural networks

### Resources

**General resources for additional material not covered in this course**

* [MIT 6.S191](https://www.youtube.com/watch?v=5v1JnYv_yWs&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1)- Intro to deep learning crash course. [Slides here](http://introtodeeplearning.com/)

* [Random forest resource 1](https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76)

* [Random forest resource 2](https://www.datasciencecentral.com/profiles/blogs/rand)

* [Naive Bayes classifier](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c)

**Recurrent Neural Networks**

* Intuitive introduction to reinforcement learning from [UC Berkeley](http://ai.berkeley.edu/lecture_videos.html) 

* Graduate lectures from [David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

* Encoding/Decoding resources from Nvidia. [Here](https://devblogs.nvidia.com/introduction-neural-machine-translation-with-gpus/). Also see part 2 and part 3.

* [Paper](https://arxiv.org/pdf/1406.1078.pdf) by Yoshua Benjio on recurrent neural networks

* [Chris Olah's](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) blog post

**Reinforcement Learning**

* [Textbook](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf ) on reinforcement learning

* [Tsitsikils lecture](https://www.youtube.com/watch?v=fbmAsxbLal0)


**Deep Learning, Nueral Networks, Convolutional Neural Networks**

* [Tutorial](https://github.com/kyunghyuncho/DeepLearningTutorials)

* Convolutional Neural Networks by [Andrej Karpathy](http://cs231n.stanford.edu/)


**Generative and discriminative models**

* [What is the difference between a generative and discriminative model](https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-a-discriminative-algorithm)

* [Theoretical difference between a generative and discriminative model](http://papers.nips.cc/paper/2020-on-discriminative-vs-generative-classifiers-a-comparison-of-logistic-regression-and-naive-bayes.pdf)

* [Linear Discriminant Analysis](https://towardsdatascience.com/classification-part-2-linear-discriminant-analysis-ea60c45b9ee5)





```python

```
