---
title: "Projects"
permalink: /projects/
author_profile: true
layout: single
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

This page contains a listing of some of the projects I have implements on my self study journey. All of these projects were implemented by me from scratch. 

**Please click through the project links below to access full jupyter notebooks exported as `markdown` files.**

## Solubility prediction of organic molecules from images of charge density

In this project, we aim to make predictions of molecular solubility in water from the [AqSolDB](https://www.nature.com/articles/s41597-019-0151-1) dataset, which contains solubility data along with relevant features of nearly 10,000 molecules. As a proof of concept, we explore in this project if we can make solubility predictions based solely on images of polar charge density. This project was conceived with some domain knowledge in mind; water is a polar molecules, and is likely to dissolve other polar molecules, (i.e.) molecules that have significant polar charge density. The hypothesis is that if we draw maps of polar charge density of the molecules (using the `rdkit` library), we might be able to make predictions by using a convolutional neural network to extract these polar features from the image.

This project has two files: A [utils](/projects/utils/) file that contains code for generating images of the polar charge density and applying other data preprocessing, and a [main](/projects/cnn-hydrophilicity-from-structures/) file that contains the code that implements both a `torch.utils.Dataset.DataLoader` object and a CNN.

## [Scientific abstract automatic text generation](/projects/rnn-arxiv/)

Anyone who has submitted scientific abstracts to conferences knows it tends to slip through the cracks and we tend to make it in just before the submission deadline. Wouldn't it be nice to have an abstract automatically written? In this project, I implement a LSTM-feed forward neural network structure for automatic character level text generation based on a corpus of abstracts downloaded through Cornell University's arxiv [API](https://arxiv.org/help/api). The html files were parsed with `BeautifulSoup` and then fed into the network for character level automatic text generation.

I spent a lot of time trying to design the network architecture, and with limited compute resources, I found it was taking a significant time to iterate between different architecture-hyperparameter combinations. For this reason, I implemented the same architecture used [here](https://github.com/spro/char-rnn.pytorch) for automatic text generation based on  Shakespeare dialogs. All other code was written from scratch.

## [Molecule SMILES string generator](/projects/rnn-smiles-generator/)

This is an example of a project that didn't work out as planned, as a reminder to come back and troubleshoot and debug. The intention was to implement a character level SMILES string generator using a gated recurrent neural network, given a hydrophobic/hydrophilic tag (i.e.) generate appropriate SMILES strings given whether I want a water soluble or water insoluble molecular structure. However, the model did not seem to learn, probably for a combination of reasons: was my architecture appropriate (I also tried ungated networks)? Were my hyper-parameters well tuned? Did I feed in the inputs in the right manner? I think the problem lies in this last question - I thought I could feed the RNN character pairs without showing the network consistent history in an attempt to improve diversity. It seems to me that I need to feed longer strings rather than just character pairs so that it can actually learn text structure. I plan to work on this further to understand what is going on. Oh well, not all projects work out! 


## [MNIST digit prediction](/projects/MNISTdigitprediction/)

In this project I implement a convolutional neural network based classifier on the famous MNIST dataset. This was a good early project to implement from scratch because it is a simple project that clarifies a lot of steps in the process of setting up a convolutional neural network architecture, seeing the influence of hyper-parameters, creating custom `Dataset` classes, and running both training and validation steps. This project really set the stage for all the other work that I followed up with it because it allows you to examine in detail how convolutional 
neural networks and feed forward neural networks work. 

## [Mushroom toxicity prediction](/projects/mushroomproject/)

In this basic project, I implement a Naive Bayes classifier to predict if a mushroom is likely to be toxic or not given all of its attributes. Here I do some significant data exploration, cleaning and feature selection. We then use the Naive Bayes Classifier to find out the most significant features that either cause a mushroom to be poisonous or be edible. More detailed notes are available within the project file. 

## [Titanic Survivor Prediction](/projects/Titanic/)

This was simple project that implemented a k-Nearest Neighbors scheme using `scikit-learn` to make predictions on whether a passenger is likely to have survived or not. I also included a small loop to look at the influence of the number of neighbors hyper parameter to find the optimal number to look at the trade off between low training error but poorer validation error (arising from over-fitting). 