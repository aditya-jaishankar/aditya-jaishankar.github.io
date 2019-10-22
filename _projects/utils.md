---
title: "Molecular solubility from polar charge density images: utils file"
layout: single
permalink: /projects/utils/
author_profile: true
---

This is the `utils` file of the project that makes solubility predictions of molecules based solely on pictures of polar charge density of molecules. The dataset was downloaded from this paper: https://www.nature.com/articles/s41597-019-0151-1


```python
import numpy as np
import pandas as pd

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps

import glob
import imageio
from tqdm import tqdm

import matplotlib.pyplot as plt
# matplotlib.use('agg')

import torch

import os
```

We first binarize the solubility to create a categorical label. We find the median value of the solubility and set everything above that value to be soluble in water, and everything below that to be insoluble.


```python
data = pd.read_csv('./curated-solubility-dataset.csv')
med = np.median(data['Solubility'])
# If hydrophobic, then label 1, else label 0
data['Solubility'] = np.where(data['Solubility']<=med, 0, 1)
```

Here we give some general drawing parameters. The idea is to draw the molecules without any atomic labels and just focusing on partial charge density and the color of the bonds (because the atoms are colored). I would expect that molecules with high partial charge densities are soluble in water and low partial charge densities are not soluble in water. This comes from the domaim knowledge of solubility.

Solubility is a complicated problem that requires significant effort in choosing the right features to make the right predictions. The goal of this project is to see how far we can get with just pictures of charge density. As we will see, pretty far!


```python
n_mols = data.shape[0]
Draw.DrawingOptions.bondLineWidth = 2
Draw.DrawingOptions.atomLabelFontSize = 1
Draw.DrawingOptions.atomLabelMinFontSize = 1
trim_margin = 20
```

We next generate the figures of molecular structure and polar charge density using the `python` package `rdkit`. This is a large dataset so this takes a while. We only have to do this once and save the figures for future use. The code below takes one generated image to determine the image size and trim width.


```python
def shapeParameters():
    """
    This function calculates the appropriate shape of im_tensor so that we
    can appropriately concat all image matrices to it. We need to be able to 
    preassign the shape correctly
    """

    # margin and other parameters. I first generate and image and get the
    # parameters from there. The only user parameter required is the trim
    # width.
    i = 10
    smiles = data['SMILES'][i]
    category = data['Solubility'][i]

    mol = Chem.MolFromSmiles(smiles)
    Chem.ComputeGasteigerCharges(mol)
    contribs = ([float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) 
                for i in range(mol.GetNumAtoms())])
    filename = dir_path + '/figs/mol' + str(i) + '.png'
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs,
                                                    scale=150,
                                                    colorMap='bwr', 
                                                    contourLines=1,
                                                    size=(250, 250))
    fig.savefig(filename, bbox_inches='tight')
    
    im = imageio.imread(filename)
    height, width, channels = im.shape
    trimmed_height, trimmed_width = (height - 2*trim_margin, 
                                             width - 2*trim_margin)
    
    return trimmed_width, trimmed_height
```

Next, we convert all the SMILES strings for the entire dataset into partial charge density figures. After we have the figures exported, we convert all of them into torch tensor. Each figure is 3 dimensional of size `[H x W x RGBA]`


```python
def molsToImgsToTensor():
    """
    This function takes all the molecules in the dataset, generates images of
    these molecules and daves these as png images
    """
    # We need to preallocate the tensor in memory because append/concat is very
    # slow. We will have a bunch of elements at the end which we will delete
    # before returning the result by maintaining a counter.
    # %matplotlib agg
    trimmed_height, trimmed_width = shapeParameters()
    n_mols = data.shape[0]  
    im_tensor = torch.zeros((n_mols, trimmed_height, trimmed_width, 4),
                            dtype=torch.uint8)
    category_tensor = torch.zeros((n_mols, 1))
    counter = 0
    for i in tqdm(range(data.shape[0])):
        try:
            smiles = data['SMILES'][i]
            category = torch.Tensor([data['Solubility'][i]]).view(1,-1)
            mol = Chem.MolFromSmiles(smiles)
            Chem.ComputeGasteigerCharges(mol)
            contribs = ([float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) 
                        for i in range(mol.GetNumAtoms())])
            filename = dir_path + '/figs/mol' + str(i) + '.png'
            fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs,
                                                            scale=150,
                                                            colorMap='bwr', 
                                                            contourLines=1,
                                                            size=(250, 250))
            fig.savefig(filename, bbox_inches='tight')
            
            im = imageio.imread(filename)
            height, width, channels = im.shape
            
            im = im[trim_margin:-trim_margin, trim_margin:-trim_margin, :]
            im = torch.from_numpy(im).view(1, trimmed_width, trimmed_height, 4)
            im_tensor[counter] = im
            # im_tensor = torch.cat((im_tensor, im), dim=0)
            category_tensor[counter] = category
            # category_tensor = torch.cat((category_tensor, category),
                                        # dim=0)
            counter += 1
        except:
            pass
    return (counter, im_tensor.numpy()[:counter], 
                     category_tensor.int().numpy()[:counter])

```


```python
counter, im_tensor, category_tensor = molsToImgsToTensor()

np.save('./im_tensor', im_tensor)
np.save('./category_tensor', category_tensor)
```

 At this point we have a tensor that contains all the images. We then take this
 tensor and convert this into a data training pair in the main file.


```python

```
