---
title: "SMILES string generation given hydrophobic/hydrophilic label"
layout: single
permalink: /projects/rnn-smiles-generator/
author_profile: true
---

 In this file, we are trying to implement all of our learnings from the previous failed runs of simple recurrent neural network. Here we use a gated neural network.
 
 * I am not going to one hot encode but instead I am going to label encode using a dictionary
 * I am going to use nn.CrossEntropyLoss() as a way to calculate label losses and maybe that will work better.
 * I will use a different structure of the rnn - try using pytorch structure and see what I am doing different.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import rdkit.Chem as Chem
from rdkit.Chem import Draw

import torch
import torch.nn as nn
import torch.optim as optim
```


```python
# Note that the solubility units are given in lopS (log of molar solubility). 
# This dataset is available as the AqSolDB data set here: https://www.nature.com/articles/s41597-019-0151-1
data = pd.read_csv('./curated-solubility-dataset.csv')

```


```python
# Let us visualize some of the molecules for kicks. The figure shows a sampling of four different molecules
rows = data.shape[0]
idxs = np.random.randint(low=0, high=rows, size=4)

ss = data.iloc[idxs]['SMILES']
ms = [Chem.MolFromSmiles(s) for s in ss.tolist()]

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.axis('off')
ax.imshow(Draw.MolsToGridImage(ms, molsPerRow=2))

```

```python
# We find the median value of logS, and then set everything below
# the median value is classified as 0, and everything above that is classified as 1. 
med = np.median(data['Solubility'])
data['Solubility'] = np.where(data['Solubility']<=med, 0, 1)
```


```python
# We define the RNN based on the name generation code by pytorch.
# I tried my own architectures without success. # so in my debugging 
# effort I thought I would have one less variable by trying a known 
# architecture.

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(1 + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(1 + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, category, inp, hidden):
        input_combined = torch.cat((category, inp, hidden), dim=1)
        hidden = self.sigmoid(self.i2h(input_combined))
        output = self.sigmoid(self.i2o(input_combined))
        output_combined = torch.cat((hidden, output), dim=1)
        output = self.sigmoid(self.o2o(output_combined))
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
```


```python
smiles_combined = ''.join(data['SMILES']) + '^' # <EOS> character
unique_characters = np.unique(list(smiles_combined)).reshape(-1)

# We now construct a dictionary by associating each character with a label

idxs = np.arange(len(unique_characters))
idx_char_arr = np.concatenate((unique_characters.reshape(-1, 1), idxs.reshape(-1, 1)), axis=1)


char_dict = {char: int(idx) for [char, idx] in idx_char_arr}
```


```python
def encodedSMILES(smiles):
    """
    This function takes a smiles string and returns a numpy array of numbers.
    Each number is the label for that character from the char_dict
    Note that on inspection we arbitrarily set the <EOS> character to be '^'
    """
    chars = list(smiles)
    return np.fromiter(map(lambda char: char_dict[char], smiles), dtype=np.int)

```


```python
def invert_encoded(encoded):
    enc = torch.argmax(encoded)
    return [key for key, val in char_dict.items() if val==int(enc)][0]

```


```python
def characterPairs(data_row):
    """
    This function returns the category followed by a torch tensor of size 2 x len(unique_chars).
    The output array contains the encoded input char and the encoded successive char for all 
    chars in the smiles string. The last char has no successive char so we use the EOS char '^'.
    """
    category = data_row['Solubility']
    smiles = data_row['SMILES']
    encoded_smiles = encodedSMILES(smiles)
    out = torch.empty((len(encoded_smiles), 2))
    for i, encoded in enumerate(encoded_smiles):
        a = encoded_smiles[i]
        try:
            b = encoded_smiles[i+1]
        except:
            b = char_dict['^'] # The <EOS> character 
            # if at the end of smiles string
        out[i] = torch.Tensor([a,b]).view(1, -1)
    return torch.Tensor([category]).view(1, -1), out

```


```python
INPUT_SIZE = unique_characters.shape[0]
OUTPUT_SIZE = INPUT_SIZE
HIDDEN_SIZE = 64
NUM_EPOCHS = 50

rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
optimizer = optim.SGD(rnn.parameters(), lr=1e-6) # maybe change to lr=0.05

criterion = nn.NLLLoss()
```


```python
def train(data):
    epoch_loss = np.empty(NUM_EPOCHS)
    for n in range(NUM_EPOCHS):
        for i in tqdm(range(0, data.shape[0])):
            hidden = rnn.initHidden()
            data_row = data.iloc[i]
            category, pairs = characterPairs(data_row)
            total_loss = 0
            for pair in pairs:
                idx = pair[0]
                inp = torch.zeros((1, INPUT_SIZE))
                inp[0, idx.long()] = 1
                target = pair[1].view(-1).long()

                output, hidden = rnn(category,inp, hidden)
                optimizer.zero_grad()
                loss = criterion(output, target)
                total_loss += loss
            total_loss.backward()
            optimizer.step()
        epoch_loss[n] = total_loss.item()
        print(total_loss.item())
    return hidden, epoch_loss

```


```python
hidden, epoch_loss_mat = train(data)
```


```python
plt.plot(np.arange(epoch_loss_mat.shape[0]), epoch_loss_mat)
plt.show()
```


```python
def generateSmiles():
    # We first generate a rondom character
    with torch.no_grad():
        seed = np.random.randint(0, data.shape[0])
        seed_smiles = data['SMILES'][seed]
        seed_character = seed_smiles[0]
        # seed_character = 'S'
        inp = torch.zeros((1, INPUT_SIZE))
        inp[0, char_dict[seed_character]] = 1
        smiles_generated = seed_character
        category = torch.Tensor([data['Solubility'][seed]]).view(1, -1)
        itera = 0
        out_char = seed_character
        psmiles = out_char
        while  psmiles[-1] != '^':  # while the generated character is not <EOS>
            # print(inp)
            combined = torch.cat((category, inp), dim=1).view(1,1,-1)
            out  = lstm(combined)
            out_char = invert_encoded(out)
            psmiles += out_char
            if itera == 20:
                break
            itera += 1
            next_char = seed_smiles[itera]
            inp = torch.zeros((1, INPUT_SIZE))
            inp[0, char_dict[next_char]] = 1
        # print(psmiles)

        return psmiles


```


```python
gensmiles = generateSmiles()
if len(gensmiles) <= 21:
    gensmiles = gensmiles[:-1]
print(gensmiles)
genmol= Chem.MolFromSmiles(gensmiles)
```


```python
# It seems like my predictions are still not great. I need a significantly more
# complex recurrent neural network. Perhaps my dataset is also found wanting. 
# I might need a bigger dataset with more consistency of structure in the 
# SMILES strings?
```
