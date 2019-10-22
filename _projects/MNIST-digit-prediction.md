---
title: "MNIST digit classification (Kaggle)"
toc: true
layout: single
permalink: /projects/MNISTdigitprediction/
author_profile: true
tags: [Naive-Bayes-Classifier]
excerpt: "This is an excerpt"
read_time: true
---

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0')

import os
from tqdm import tqdm

```


```python
# cwd = os.path.dirname(os.path.realpath(__file__))

train_data = torch.tensor(pd.read_csv('./train.csv').values, dtype=torch.float)
test_data = torch.tensor(pd.read_csv('./test.csv').values, dtype=torch.float)
```


```python
labels = train_data[:, 0].view([-1])
train_tensor = train_data[:, 1:].view([-1, 28, 28])
test_tensor = test_data.view([-1, 28, 28])

# Creates an iterator over the dataset
dataset = TensorDataset(train_tensor, labels)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
dataset_test = TensorDataset(test_tensor)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

```


```python
# Defining the architecture of the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(5*5*64, 128)  #This is was calculated before hand, but can be made more general
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(1, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

net = Net()
net.to(device)

```




    Net(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (relu): ReLU()
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (max_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=1600, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=10, bias=True)
    )




```python
NUM_EPOCHS = 4
optimizer = optim.SGD(net.parameters(), lr=.001)
optimizer.zero_grad()
loss_fn = nn.CrossEntropyLoss()
loss_mat = []

for n in range(NUM_EPOCHS):
    total_loss = 0
    for data, target in tqdm(dataloader):
        output = net(data.to(device).view(1, -1, 28, 28))
        loss = loss_fn(output, target.to(device).long())
        total_loss += loss.item()
        loss.backward()        
        optimizer.step()
        optimizer.zero_grad()
    print('Loss for Epoch Number', n, '=', total_loss)
    loss_mat.append(total_loss)
```

    100%|██████████| 42000/42000 [03:57<00:00, 176.95it/s]
    

    Loss for Epoch Number 0 = 14897.444639444351
    

    100%|██████████| 42000/42000 [03:46<00:00, 185.40it/s]
    

    Loss for Epoch Number 1 = 2580.0293949842453
    

    100%|██████████| 42000/42000 [04:07<00:00, 169.99it/s]
    

    Loss for Epoch Number 2 = 1713.4369139671326
    

    100%|██████████| 42000/42000 [04:05<00:00, 206.20it/s]
    

    Loss for Epoch Number 3 = 1361.1752125024796
    


```python
plt.plot(np.arange(len(loss_mat)), loss_mat)
plt.show()
```


![png](/assets/images/projects/MNISTdigitprediction/output_5_0.png)



```python
def validation(data):
    with torch.no_grad():
        inp, target = data
        inp = inp.to(device)
        target = target.to(device)
        out = net(inp.view(1, -1, 28, 28))
        plt.imshow(inp.cpu().numpy()[0])
        plt.show()
        print('output:', torch.topk(out, k=1)[1][0].item())
        print('target:', target.long().item())

for _ in range(5):
    validation(next(iter(dataloader)))


```


![png](/assets/images/projects/MNISTdigitprediction/output_6_0.png)


    output: 0
    target: 0
    


![png](/assets/images/projects/MNISTdigitprediction/output_6_2.png)


    output: 6
    target: 6
    


![png](/assets/images/projects/MNISTdigitprediction/output_6_4.png)


    output: 1
    target: 1
    


![png](/assets/images/projects/MNISTdigitprediction/output_6_6.png)


    output: 7
    target: 7
    


![png](/assets/images/projects/MNISTdigitprediction/output_6_8.png)


    output: 2
    target: 2
    


```python
def prediction():
    res = np.empty((test_tensor.shape[0], 2), dtype=np.int)
    for i, testdata in tqdm(enumerate(dataloader_test)):
        # print(testdata)
        out = net(testdata[0].view(1, -1, 28, 28).to(device))
        pred = torch.topk(out, k=1)[1][0].item()
        res[i] = np.array([i+1, pred], dtype=np.int)
    res = res.astype(int)
    np.savetxt('results.csv', res, delimiter=',', newline='\n', fmt='%i', 
                header='ImageId,Label', comments='')

prediction()

```

This implementation of the CNN feeding a fully connected ANN achieves over 96% accuracy on the validation set on Kaggle.
