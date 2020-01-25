---
title: "Projects"
permalink: /projects/
author_profile: true
layout: single
classes: wide

---

<!-- {% assign tags =  site.projects | map: 'tags' | join: ','  | split: ',' | uniq %}
{% for tag in tags %}
  <h3>{{ tag }}</h3>
  <ul>
  {% for project in site.projects %}
    {% if project.tags contains tag %}
    <li><a href="{{ site.baseurl }}{{ project.url }}">{{ project.title }}</a></li>
    <i>{{ project.excerpt }}</i>
    {% endif %}
  {% endfor %}
  </ul>
{% endfor %} -->

This page contains a listing of some of the projects I have implemented on my self study journey. All of these projects were implemented by me from scratch. 

**Please click through the project links below to access full jupyter notebooks exported as `markdown` files.**

## Denoising text images

This project is currently in progress. Please check back soon!

## [The Higgs Boson Machine Learning Challenge](/projects/higgs/)

In this [Kaggle challenge](https://www.kaggle.com/c/higgs-boson/overview), we are given two types of entries in a numerical dataset; rows representing significant signals that has features of a tau-tau decay, and rows representing background signals that has no significance. The signal is usually deeply buried in the noise, so is difficult to identify. The goal is to use machine learning to help distinguish the significant traces that contain meaningful data from the background traces with nothing of significance. The (labeled) dataset provided had a number of missing entries, and these missing entries were arbitrarily provided a value of -999.0, so I had to implement data cleaning and normalizing steps to prevent these from unduly influencing the data. The details of my implementation can be found in the code. I implemented a deep neural network architecture with a cross entropy loss function to perform binary classification: does a particular data vector represent signal or noise? With my architecture, I was able to distinguish meaningful signals from background noise with an accuracy of over 99.5%. 

## [Solution of partial differential equations using neural networks](/projects/neuralpde/)

The solution of PDEs is of critical importance to a large variety of physical problems. However, except for the most elementary cases, there are no analytical closed form solution to most PDEs describing phenomena in practical applications. There are various numerical ways that can be used to solve these equations, some more computationally efficient, and accurate than others. In my own research I have spent a significant amount of time implementing finite difference schemes to solve partial differential equations. This can be very computationally intensive and can take very long, even on multi-core computing clusters. 

One alternative is to use artificial neural networks to solve the PDEs. The idea has been explored in some detail in the existing literature (for example see [here](https://ieeexplore.ieee.org/abstract/document/712178)). The key insight here is that we use a neural network as a function approximator. We write the solution as the sum of two terms; the first term satisfies the boundary conditions, while the second term is constructed in such a way as to not contribute to the boundaries. The neural network is then trained over the domain of the equation by minimizing the squared loss when the equation is written in the form LHS=0 (i.e.) we force the neural network to adjust it's parameters so that the constructed solution satisfies the equation. More details and notes on the implementation can be found in the markdown file.

## Solubility prediction of organic molecules from images of charge density

[Utils file](/projects/utils)<br>
[Main code](/projects/cnn-hydrophilicity-from-structures/)

In this project, we aim to make predictions of molecular solubility in water from the [AqSolDB](https://www.nature.com/articles/s41597-019-0151-1) dataset, which contains solubility data along with relevant features of nearly 10,000 molecules. As a proof of concept, we explore in this project if we can make solubility predictions based solely on images of polar charge density. This project was conceived with some domain knowledge in mind; water is a polar molecule, and is likely to dissolve other polar molecules, (i.e.) molecules that have significant polar charge density. The hypothesis is that if we draw maps of polar charge density of the molecules (using the `rdkit` library), we might be able to make predictions by using a convolutional neural network to extract these polar features from the image.

This project has two files: A [utils](/projects/utils/) file that contains code for generating images of the polar charge density and applying other data preprocessing, and a [main](/projects/cnn-hydrophilicity-from-structures/) file that contains the code that implements both a `torch.utils.Dataset.DataLoader` object and a CNN.

## [Scientific abstract automatic text generation](/projects/rnn-arxiv/)

Anyone who has submitted scientific abstracts to conferences knows it tends to slip through the cracks and we tend to make it in just before the submission deadline. Wouldn't it be nice to have an abstract automatically written? In this project, I implement a LSTM-feed forward neural network structure for automatic character level text generation based on a corpus of abstracts downloaded through Cornell University's arxiv [API](https://arxiv.org/help/api). The html files were parsed with `BeautifulSoup` and then fed into the network for character level automatic text generation.

I spent a lot of time trying to design the network architecture, and with limited compute resources, I found it was taking a significant time to iterate between different architecture-hyperparameter combinations. For this reason, I implemented the same architecture used [here](https://github.com/spro/char-rnn.pytorch) for automatic text generation based on Shakespeare dialogs. All other code was written from scratch.

## [Molecule SMILES string generator](/projects/rnn-smiles-generator/)

This is an example of a project that didn't work out as planned, and I include it here as a reminder to come back, troubleshoot and debug. The intention was to implement a character level SMILES string generator using a gated recurrent neural network, given a hydrophobic/hydrophilic tag (i.e.) generate appropriate SMILES strings given whether I want a water soluble or water insoluble molecular structure. However, the model did not seem to learn, probably for a combination of reasons: was my architecture appropriate (I also tried ungated networks)? Were my hyper-parameters well tuned? Did I feed in the inputs in the right manner? I think the problem lies in this last question - I thought I could feed the RNN character pairs without showing the network consistent history in an attempt to improve diversity. It seems to me that I need to feed longer strings rather than just character pairs so that it can actually learn SMILES text structure. I plan to work on this further to understand what is going on. Oh well, not all projects work out! 


<!-- ## [MNIST digit prediction](/projects/MNISTdigitprediction/)

In this project I implement a convolutional neural network based classifier on the famous MNIST dataset. This was a good early project to implement from scratch because it is a simple project that clarifies a lot of steps in the process of setting up a convolutional neural network architecture, seeing the influence of hyper-parameters, creating custom `Dataset` classes, and running both training and validation steps. This project really set the stage for all the other work that I followed up with it because it allows you to examine in detail how convolutional 
neural networks and feed forward neural networks work. 

## [Mushroom toxicity prediction](/projects/mushroomproject/)

In this basic project, I implement a Naive Bayes classifier to predict if a mushroom is likely to be toxic or not given all of its attributes. Here I do some significant data exploration, cleaning and feature selection. We then use the Naive Bayes Classifier to find out the most significant features that either cause a mushroom to be poisonous or be edible. More detailed notes are available within the project file. 

## [Titanic survivor prediction](/projects/Titanic/)

This was a simple project that implemented a k-Nearest Neighbors scheme using `scikit-learn` to make predictions on whether a passenger is likely to have survived or not. I also included a small loop to look at the influence of the number of `neighbors` hyper parameter to find the optimal number, and in the process examined the trade off between low training error but poorer validation error (arising from over-fitting). This was a good beginner's project to apply theory in practice (which I have learned are often very different things!). -->