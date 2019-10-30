---
title: "Molecular solubility from polar charge density images: Main file"
layout: single
permalink: /projects/cnn-hydrophilicity-from-structures/
author_profile: true
---

Now that we have all the figures generated and corresponding numpy arrays for the figures, we can continue with the goal of implementing a convolutional neural network to make solubility predictions.

First we do all the basic imports


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
device = torch.device('cuda:0')

from tqdm import tqdm_notebook
```

Next, we import the exported numpy arrays consisting of all the exported figures. This is quite a large file so can take a while. Note that we will have to transpose the dataset because `torch` required the figure format to be `[N, C, H, W]` where `C` are the channels.


```python
images_arr = np.load('./im_tensor.npy')
labels_arr = np.load('./category_tensor.npy')

images_train, images_test, labels_train, labels_test = train_test_split(
    images_arr, labels_arr, test_size=0.2, random_state=25)

# torch requires images in the format (channels, height, width)
images_train = np.transpose(images_train, axes=(0, 3, 1, 2))
images_test = np.transpose(images_test, axes=(0, 3, 1, 2))
```

Because of the size of the dataset, we are better off not loading the whole dataset in memory, and instead use the `Dataset` class from `torch.utils.data`. We can use this to implement our own custom dataset in the form of an iterator. All we need to do is to implement the `__len__()` and `__getitem__()` methods.  


```python
class imagesDataset(Dataset):

    def __init__(self, images, categories):

        super(imagesDataset, self).__init__()

        self.images = images
        self.categories = categories

    def __len__(self):
        # len() specifies the upper bound for the index of the dataset
        return len(self.categories)

    def __getitem__(self, index):
        # The generator executes the getitem() method to generate a sample
        return self.images[index], self.categories[index]
```


```python
train_set = imagesDataset(images_train, labels_train)
test_set = imagesDataset(images_test, labels_test)

batch_size = 5
train_generator = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_generator = DataLoader(test_set, batch_size=batch_size, shuffle=True)
```

Let us load some structures to see what they look like:


```python
# Lets look at some of these structures.
images = next(iter(train_generator))[0]

for image in images:
    image = image.numpy()
    image = np.transpose(image, axes=(1, 2, 0))
    plt.imshow(image)
    plt.show()
    
```


![png](/assets/images/projects/cnn-solubility-prediction/output_8_0.png)



![png](/assets/images/projects/cnn-solubility-prediction/output_8_1.png)



![png](/assets/images/projects/cnn-solubility-prediction/output_8_2.png)



![png](/assets/images/projects/cnn-solubility-prediction/output_8_3.png)



![png](/assets/images/projects/cnn-solubility-prediction/output_8_4.png)


The images show that the polar charge density of the different molecules arising from the chemical nature of the different atoms inthe structure. The intensity of color represents how strong the charge, while the color itself represents whether the char ge is positive (red) or negative (blue). The goal of generating this figure is to use a CNN to check if we get reasonable results for solubility predictions based on the distribution of charge density alone. In reality, solubility is a complicated prediciton to make, with factors such as the 3D structure of the geometry and intermolecular interactions (i.e. how strongly the molecules want to bind to each other) playing a major role.

Next we define the architecture of the CNN we wish to use. We are going to use two convolutiona layers with 32 and 64 channels each with relu activation and max pooling at the end of each activation. We then pass these through two fully connected dense layers and decrease the dimensionality in a step wise direction, again with relu activate. For the very last layer we use a log softmax activation so that we use a negative log likelihood loss function later.


```python
# Next we define the CNN we want to use

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))

        self.fc1 = nn.Linear(104*104*64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = x.view(-1, 104*104*64)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.logsoftmax(x)

        return x
```


```python
batch_size = 25
NUM_EPOCHS = 50
cnn = CNN().to(device)

objective = nn.NLLLoss()
optimizer = optim.SGD(cnn.parameters(), lr=.001)

```


```python
total_loss = torch.zeros(NUM_EPOCHS)
for i in range(NUM_EPOCHS):
    for inp, labels in tqdm_notebook(train_generator):
        inp, label = inp.float().to(device), labels.float().to(device)
        out = cnn(inp)

        optimizer.zero_grad()
        # The reshaping is because pytorch needs these dimensions for 
        # NLLLoss to work properly. 
        loss = objective(out.squeeze(), label.long().view(-1))
        loss.backward()
        optimizer.step()
        total_loss[i] += loss.item()/(.8*len(train_generator)*batch_size)
    print('Loss for Epoch', i, '=', total_loss[i])

plt.plot(range(total_loss.shape[0]), total_loss)
f = './cnn_saved.pt'
torch.save(cnn.state_dict(), f)
```

Once we spend the time training, we save the model's `state_dict()` so that we dont have to spend time retraining the model, which takes about an hour despite having access to a RTX 2060 6 GiB NVIDIA GPU. Below we load this trained model.


```python
f = './cnn_saved.pt'
cnn = CNN()
cnn.load_state_dict(torch.load(f))
cnn.eval()
cnn.to(device)
```


```python
def validation():
    errors = 0
    for inp, label in tqdm_notebook(test_generator):
        preds = torch.argmax(cnn(inp.float().to(device)), dim=1)
        labels = label.to(device).view(-1)
        errors += torch.sum((preds==labels)*1).item()
    return errors*100/(len(test_generator)*(batch_size/5))
# The factor of 5 exists because the test_generator dataset object
# was created using a batch_size of 5, while the training set was
# created with a batch_size of 25
```


```python
accuracy = validation()
print('accuracy:', accuracy, '%')
```


    accuracy: 73.66834170854271 %

We see that just by looking at images of the polar charge density, we are able to get a prediction accuracy of about 75%. This is remarkable given how simplistic our model is. Random chance would dictate that we should hit 50%, so our model has learned something about the physics of solubility. Of course, if we were interested in solubility in other solvents instead of water, our predictions would not be great. 
    
