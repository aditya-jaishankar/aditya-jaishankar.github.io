---
title: "Automatic scientific abstract text generation"
toc: true
layout: single
permalink: /projects/rnn-arxiv/
author_profile: true
excerpt: "This is an excerpt"
read_time: true
---

## Basic Imports


```python
import urllib
import re
from bs4 import BeautifulSoup
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

# import itertools

import string
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from sklearn.preprocessing import OneHotEncoder
```

The first step is to be able to extract all the data. For this we use `BeautifulSoup` to extract all the text between the `<summary>` and `</summary>` tags. We then have a list of all abstracts. We can then look at it from a character level perspective, train an LSTM (say we use Karpathy's architecture instead of fooling around with different architectures) and then have it output an abstract of say 500 words. This should be an interesting exercise!

## Downloading the corpus and extracting the abstracts


```python
def downloadDataCorpus(parsehtml=False, downloadfile=False):
    """
    Creates the corpus.
    Inputs:
        parsehtml: whether to run this function at all. I should normally have this
        corpus presaved as a text file
        downloadfile: whether to download the data or not. This seems to be the speed
        bottleneck. 
    """
    if not parsehtml:
        return None
    
    if downloadfile:
        url = 'http://export.arxiv.org/api/query?search_query=all:polymer&start=0&max_results=10000'
        f = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(f, 'html.parser')
        with open(r'./rawData.txt', 'w+', encoding='utf-8') as out:
            out.write(str(soup))
    soup = BeautifulSoup(open('./rawData.txt', encoding='utf-8'), 'html.parser')
    abstracts = []
    for summary in tqdm(soup.find_all('summary')):
        abstracts.append(summary.text.replace('\n', ' ').strip())
    abstracts = np.array(abstracts)
    np.save('./abstracts.npy', abstracts)

downloadDataCorpus(parsehtml=False, downloadfile=False)

corpus = np.load('./abstracts.npy')
all_text = ''.join(corpus)  # combine all abstracts into a single string
```

## Defining some utility functions

### Encoding the characters into numbers


```python
all_characters = string.printable
n_characters = len(all_characters)

def encoder(string):
    """
    This function takes a string and tokenizes it by assigning it a unique index
    """
    encoded = torch.zeros(len(string)).long()
    for i, char in enumerate(string):
        encoded[i] = all_characters.index(char)
    return encoded

def save():
    """
    To save any intermediate models while training in case I give a
    KeyboardInterrupt.
    """
    torch.save(lstm_model.state_dict(), './arxiv-generator2.pt')
    return None
```

### Defining a `Dataset` object to generate the data


```python
class trainingSet(Dataset):

    def __init__(self, all_text, str_length):
        super(trainingSet, self).__init__()
        self.all_text = all_text
        self.str_length = str_length

    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, idx):
        seed = np.random.randint(0, len(self.all_text) - self.str_length)
        sequence = self.all_text[seed:seed+str_length]
        inp = encoder(sequence[:-1])
        target = encoder(sequence[1:])
        return inp, target

str_length = 25
batch_size = 100
dataset = trainingSet(all_text, str_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
```

## Defining the structure of the LSTM

Next need to define my lstm class. The need to define the training loop. Then need to define an abstract generator loop. if count('.') > 10, then break (this would keep the abstract a reasonable length).


```python
class LSTM(nn.Module):


    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.output_size = OUTPUT_SIZE
        self.num_layers = num_layers

        self.encoder = nn.Embedding(INPUT_SIZE, HIDDEN_SIZE)
        self.lstm = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, num_layers=num_layers)
        self.decoder = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, inp, hidden):
        batch_size = inp.size(0) # Think about what exactly is meant by batch_size here
        encoded = self.encoder(inp)
        output, hidden = self.lstm(encoded.view(1,batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden
    
    # Probably lets me handle predictions very nicely
    def forward2(self, inp, hidden):
        encoded = self.encoder(inp.view(1, -1))
        output, hidden = self.lstm(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
```

## Defining the training loop


```python
def train(inp, target):
    hidden = lstm_model.initHidden(batch_size)

    optimizer.zero_grad()
    loss = 0

    for i in range(str_length-1):
        output, hidden = lstm_model(inp[:, i], hidden)
        loss += loss_fn(output.view(batch_size, -1), target[:, i])

    loss.backward()
    optimizer.step()

    return loss.item()/str_length
```


```python
NUM_EPOCHS = 1000000
INPUT_SIZE = n_characters
HIDDEN_SIZE = 128
OUTPUT_SIZE = n_characters
num_layers = 2
lstm_model = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_layers=num_layers)

optimizer = optim.Adam(lstm_model.parameters(), lr=.001)
loss_fn = nn.CrossEntropyLoss()
lstm_model.to(device)
```




    LSTM(
      (encoder): Embedding(100, 128)
      (lstm): LSTM(128, 128, num_layers=2)
      (decoder): Linear(in_features=128, out_features=100, bias=True)
    )




```python
all_losses = []
loss_avg = 0

try:
    for epoch in tqdm(range(NUM_EPOCHS)):
        inp, target = next(iter(dataloader))
        loss = train(inp.to(device), target.to(device))
        loss_avg += loss

        if (epoch + 1) % 10000 == 0:
            print('loss so far: ', loss)
            all_losses.append(loss_avg)
            loss_avg = 0

except KeyboardInterrupt:
        print('Saving model...')
        save()

save()
plt.plot(range(len(all_losses)), all_losses)
plt.show()
```

### Code to generate the text

Once the model has been trained, we can use it to generate text. Note that instead of using the character expected from the argmax of the highest probability, we actually sample from a multinomial distribution based on an 'activation' temperature. Higher the value of temperature, more noise we would expect to see, and a larger number os spelling and other errors. However, very low temperatures would give very little diversity and we end up generating text in a deterministic manner given the priming string. 


```python
def generate(decoder, prime_str='Here', predict_len=600, temperature=0.8, cuda=False):
    hidden_state, cell_state = decoder.initHidden(1)
    hidden = hidden_state.cpu(), cell_state.cpu()
    prime_input = encoder(prime_str).unsqueeze(0)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = encoder(predicted_char).unsqueeze(0)

    return predicted

lstm_model_pred = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_layers=num_layers).cpu()

lstm_model_pred.load_state_dict(torch.load('./arxiv-generator2.pt', map_location=torch.device('cpu')))
lstm_model_pred.eval()
generate(lstm_model_pred)
```




    'Here we study the self-avoiding walks leads to the semi-flexibon. This behavior with enhanced as displacement is influenced by the microstrating Fermi statistical models as the large size in large attractive model based on a potential formalism. The model is explored theory is defined as the T$_{0}$ concentrational statistics to show that the polymers in DNA structure can be extended to the mechanical jump above the constraint in turn on a polymer network and simulations where our approach to the beam constant exists a consecutive circuits of the electrically to unfold in polymer chains and electr'



We see that the abstract makes very good grammatical and lexical sense, and nearly reads like an accepted abstract!
