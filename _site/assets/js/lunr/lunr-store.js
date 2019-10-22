var store = [{
        "title": "Computational Fluid Dynamics Python (CFD) - Python",
        "excerpt":"Class notes for the CFD-Python course taught by Prof. Lorena Barba, Boston University. The course outlines a 12 step program, each of increasing difficulty, towards building mastery solving the Navier-Stokes equations through finite difference techniques. Preliminaries The Naiver-Stokes (N-S) equation represents the conservation of momentum of purely viscous, i.e. Newtonian...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/coursenotes/CFDpython/",
        "teaser":null},{
        "title": "Google Machine Learning Crash Course",
        "excerpt":"Definitions and Framing Supervised Machine Learning: Using known data to generate some useful predictions of on unseen data Label $y$: The target variable that we are trying to predict, for example ‘spam’ or ‘not spam’. Features: Something about the data that is used to represent the data, that is later...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/coursenotes/googlemachinelearning/",
        "teaser":null},{
        "title": "Machine Learning with Python - From Linear Models to Deep Learning",
        "excerpt":"Class Notes: 6.86x - Machine Learning and Deep Learning with Python Unit 1: Linear classifiers and generalizations Basics Definiton: Machine learning aims to design, understand and apply computer programs that learn from experience for the purposes of prediction, control, modeling of outcomes and systems. Supervised learning In supervised learning, we...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/coursenotes/machinelearning/",
        "teaser":null},{
        "title": "Kevin Markham: `pandas`",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "http://localhost:4000/coursenotes/markhampandas/",
        "teaser":null},{
        "title": "Kevin Markham: ML in Python with `sklearn`",
        "excerpt":" ","categories": [],
        "tags": [],
        "url": "http://localhost:4000/coursenotes/markhamsklearn/",
        "teaser":null},{
        "title": "Probability: The Science and Uncertainty of Data (6.431x)",
        "excerpt":"$\\require{\\cancel}$ Unit 1: Probability models and axioms Lecture 1: Probability models and axioms The basic concept in probability is that of a sample space. Probability laws help calculate probabilities of a particular event in the sample space from occurring. These laws have to respect axioms, for example, probabilities cannot be...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/coursenotes/probability/",
        "teaser":null},{
        "title": "Miscellaneous notes: `python`, `numpy`, `pandas`, `sklearn`",
        "excerpt":"python numpy pandas pandas.DataFrame.reindex(labels) allows you to reorder the index of a dataframe to the order dictated by labels. If the corresponding label existed in the original dataframe, it will slot that particular row containing that index there. If that index does not exist, it will insert NaN unless the...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/coursenotes/python_notes/",
        "teaser":null},{
        "title": "Fundamentals of Statistics(18.6501x)",
        "excerpt":"Unit 1: Introduction to Statistics Lecture 1: What is Statistics Statistics is at the core, is fundamental to machine learning and data science. Without a thorough grounding in statistics, machine learning becomes a black box. There is the computational view of data: where problems are solved with very large amounts...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/coursenotes/statistics/",
        "teaser":null},{
        "title": "MNIST digit classification (Kaggle)",
        "excerpt":"import numpy as np import matplotlib.pyplot as plt import pandas as pd from torch.utils.data import DataLoader, TensorDataset import torch import torch.nn as nn import torch.optim as optim device = torch.device('cuda:0') import os from tqdm import tqdm # cwd = os.path.dirname(os.path.realpath(__file__)) train_data = torch.tensor(pd.read_csv('./train.csv').values, dtype=torch.float) test_data = torch.tensor(pd.read_csv('./test.csv').values, dtype=torch.float) labels =...","categories": [],
        "tags": ["Naive-Bayes-Classifier"],
        "url": "http://localhost:4000/projects/MNISTdigitprediction/",
        "teaser":null},{
        "title": "SMILES string generation given hydrophobic/hydrophilic label",
        "excerpt":"In this file, we are trying to implement all of our learnings from the previous failed runs of simple recurrent neural network. Here we use a gated neural network. I am not going to one hot encode but instead I am going to label encode using a dictionary I am...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/projects/rnn-smiles-generator/",
        "teaser":null},{
        "title": "Titanic Survival Prediction",
        "excerpt":"import pandas as pd import numpy as np import matplotlib.pyplot as plt from tqdm import tqdm as tqdm from sklearn.neighbors import KNeighborsClassifier from sklearn.preprocessing import LabelEncoder from sklearn.preprocessing import MinMaxScaler from sklearn.metrics import accuracy_score from sklearn.model_selection import train_test_split The function below does all the preprocessing to remove the unnecessary columns...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/projects/Titanic/",
        "teaser":null},{
        "title": "Molecular solubility from polar charge density images: Main file",
        "excerpt":"Now that we have all the figures generated and corresponding numpy arrays for the figures, we can continue with the goal of implementing a convolutional neural network to make solubility predictions. First we do all the basic imports import numpy as np import pandas as pd import matplotlib.pyplot as plt...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/projects/cnn-hydrophilicity-from-structures/",
        "teaser":null},{
        "title": "Mushroom Toxicity Prediction ",
        "excerpt":"Mushrooms: to eat or not to eat? I love mushrooms. I also love hiking. Often, I see various types of mushrooms on strewn around the forest floor. The diversity of the different kinds of mushrooms I see is truly staggering. The type of mushroom depends on geography of course: which...","categories": [],
        "tags": ["Naive-Bayes-Classifier"],
        "url": "http://localhost:4000/projects/mushroomproject/",
        "teaser":null},{
        "title": "Project1: The mushrooom project",
        "excerpt":"Does this work well? What about math?   Equation    ","categories": [],
        "tags": ["Naive-Bayes-Classifier"],
        "url": "http://localhost:4000/projects/project1/",
        "teaser":null},{
        "title": "Automatic scientific abstract text generation",
        "excerpt":"Basic Imports import urllib import re from bs4 import BeautifulSoup import numpy as np from tqdm import tqdm import matplotlib.pyplot as plt # import itertools import string import random import torch import torch.nn as nn from torch.utils.data import Dataset, DataLoader import torch.optim as optim device = torch.device('cuda:0' if torch.cuda.is_available() else...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/projects/rnn-arxiv/",
        "teaser":null},{
        "title": "Molecular solubility from polar charge density images: utils file",
        "excerpt":"This is the utils file of the project that makes solubility predictions of molecules based solely on pictures of polar charge density of molecules. The dataset was downloaded from this paper: https://www.nature.com/articles/s41597-019-0151-1 import numpy as np import pandas as pd from rdkit.Chem import AllChem as Chem from rdkit.Chem import Draw...","categories": [],
        "tags": [],
        "url": "http://localhost:4000/projects/utils/",
        "teaser":null}]
