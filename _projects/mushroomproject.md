---
title: "Mushroom Toxicity Prediction "
layout: single
permalink: /projects/mushroomproject/
author_profile: true
---

# Mushrooms: to eat or not to eat?

I love mushrooms. I also love hiking. Often, I see various types of mushrooms on strewn around the forest floor. The diversity of the different kinds of mushrooms I see is truly staggering. The type of mushroom depends on geography of course: which part of the country or the world am I in? It can also be seasonal. I have done the same hike in summer and winter, and notices that the types and numbers of mushrooms have changed. On the same hike in the same season, I also see different mushrooms depending on altitude. All this diversity notwithstanding, I am most interested in the question: Can I eat it? 

Here I answer this question.

**Note: This project was done for educational purposes. Before eating any kind of wild and unknown mushroom, please check if it is safe to eat with experts or sources more authoritative than this project (such as textbooks).**


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
```

## Feature name cleaning and wrangling

The data source contains a list of property types, property names, and symbols. As I will show below, this data will prove critical to understanding the results of the analysis. We therefore need to save and digest this data in a useable form. The properties were saved as a text file and loaded into the notebook using `pd.read_fwf()` into a `pandas` dataframe as shown below. 


```python
prop_legend = pd.read_fwf('props_legend.txt')
prop_legend
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attribute Information: (classes: edible=e, poisonous=p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>- cap-shape: bell=b,conical=c,convex=x,flat=f,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>- cap-surface: fibrous=f,grooves=g,scaly=y,smo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>- cap-color: brown=n,buff=b,cinnamon=c,gray=g,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>- bruises: bruises=t,no=f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>- odor: almond=a,anise=l,creosote=c,fishy=y,fo...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>- gill-attachment: attached=a,descending=d,fre...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>- gill-spacing: close=c,crowded=w,distant=d</td>
    </tr>
    <tr>
      <th>7</th>
      <td>- gill-size: broad=b,narrow=n</td>
    </tr>
    <tr>
      <th>8</th>
      <td>- gill-color: black=k,brown=n,buff=b,chocolate...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>- stalk-shape: enlarging=e,tapering=t</td>
    </tr>
    <tr>
      <th>10</th>
      <td>- stalk-root: bulbous=b,club=c,cup=u,equal=e,r...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>- stalk-surface-above-ring: fibrous=f,scaly=y,...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>- stalk-surface-below-ring: fibrous=f,scaly=y,...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>- stalk-color-above-ring: brown=n,buff=b,cinna...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>- stalk-color-below-ring: brown=n,buff=b,cinna...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>- veil-type: partial=p,universal=u</td>
    </tr>
    <tr>
      <th>16</th>
      <td>- veil-color: brown=n,orange=o,white=w,yellow=y</td>
    </tr>
    <tr>
      <th>17</th>
      <td>- ring-number: none=n,one=o,two=t</td>
    </tr>
    <tr>
      <th>18</th>
      <td>- ring-type: cobwebby=c,evanescent=e,flaring=f...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>- spore-print-color: black=k,brown=n,buff=b,ch...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>- population: abundant=a,clustered=c,numerous=...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>- habitat: grasses=g,leaves=l,meadows=m,paths=...</td>
    </tr>
  </tbody>
</table>
</div>



Clearly, this table needs a lot of work to digest and convert to a format that can be used in further analysis. The ultimate goal of importing this data is to create a 'look up dictionary' mapping the feature symbol to a feature name that would help with, for example plotting, and visualizing the data to add more meaning and understanding to the analysis. The following pieces of code clean digest the data


```python
# Cleaning and wrangling the text file

prop_legend = pd.read_fwf('props_legend.txt', header=None, skiprows=1).dropna()
prop_legend.set_index(np.arange(prop_legend.shape[0]), inplace=True)
```

We now want to move the property names over as the index names. Examples of the property names are cap-shape, odor, ring-type, veil-color, etc. For this we extract the property names using regular expressions (implemented in python using the `re` package.) Documentation available here: https://docs.python.org/3.4/library/re.html

We first extract these property names into a data series to set as index names in the data frame


```python
import re

def property_extractor(element):
    """input: takes in the mushroom property with all the keys.
       output: returns the property name only
       
       The function uses regular expressions to only select words that end in a colon 
    """
    re_matcher = re.compile('[a-z-]+:')
    m = re_matcher.match(element)
    return m.group()[:-1]

property_list = prop_legend[1].apply(property_extractor)
prop_legend.set_index(property_list, inplace=True)
prop_legend.index.name = 'property type'
```

Now we replace the property name with an empty string and do some further clean up


```python
def string_replacer(element):
    return re.sub('[a-z-]+:','', element)

prop_legend[1] = prop_legend[1].apply(string_replacer)
prop_legend.drop([0], axis=1, inplace=True)
prop_legend.set_axis(['property names'], axis=1, inplace=True)
prop_legend.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>property names</th>
    </tr>
    <tr>
      <th>property type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cap-shape</th>
      <td>bell=b,conical=c,convex=x,flat=f, knobbed=k,s...</td>
    </tr>
    <tr>
      <th>cap-surface</th>
      <td>fibrous=f,grooves=g,scaly=y,smooth=s</td>
    </tr>
    <tr>
      <th>cap-color</th>
      <td>brown=n,buff=b,cinnamon=c,gray=g,green=r,pink...</td>
    </tr>
    <tr>
      <th>bruises</th>
      <td>bruises=t,no=f</td>
    </tr>
    <tr>
      <th>odor</th>
      <td>almond=a,anise=l,creosote=c,fishy=y,foul=f,mu...</td>
    </tr>
  </tbody>
</table>
</div>



We now convert the property names into a dictionary so that we can refer to them using a key value format. Examples of property names are (bell, convex, flat, etc.) for the property type cap-shape.

We again use regular expressions for this. We write the function below:


```python
def dict_converter(ser):
    """ input: one row of the dataframe
        output: converts the row into a dictionary of the form {symbol:name} 
        for easy lookup later
    """
    
    re_matcher_names = re.compile('[a-z]+=')
    n = re_matcher_names.findall(ser[0])
    names = list(map(lambda s: s[:-1], n))
    
    re_matcher_symbols = re.compile('=[a-z]')
    s = re_matcher_symbols.findall(ser[0])
    symbols = list(map(lambda s: s[1], s))

    pairs = list(zip(symbols, names))
    return {symbol: name  for (symbol, name) in pairs}

prop_legend = prop_legend.apply(dict_converter, axis=1)
prop_legend = pd.DataFrame(prop_legend)
prop_legend.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
    <tr>
      <th>property type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cap-shape</th>
      <td>{'b': 'bell', 'c': 'conical', 'x': 'convex', '...</td>
    </tr>
    <tr>
      <th>cap-surface</th>
      <td>{'f': 'fibrous', 'g': 'grooves', 'y': 'scaly',...</td>
    </tr>
    <tr>
      <th>cap-color</th>
      <td>{'n': 'brown', 'b': 'buff', 'c': 'cinnamon', '...</td>
    </tr>
    <tr>
      <th>bruises</th>
      <td>{'t': 'bruises', 'f': 'no'}</td>
    </tr>
    <tr>
      <th>odor</th>
      <td>{'a': 'almond', 'l': 'anise', 'c': 'creosote',...</td>
    </tr>
  </tbody>
</table>
</div>



We now have a data frame of the property names with all values as dictionary which we can use for easy look up while visualizing the data and interpreting the results.

## Loading  and inspecting the data


```python
data = pd.read_csv('mushrooms.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



Let us first check if the dataset is of a reasonable size to be able to do adequaete statistical analysis. We describe the data set to get a feel for the dataset


```python
data.describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>class</th>
      <td>8124</td>
      <td>2</td>
      <td>e</td>
      <td>4208</td>
    </tr>
    <tr>
      <th>cap-shape</th>
      <td>8124</td>
      <td>6</td>
      <td>x</td>
      <td>3656</td>
    </tr>
    <tr>
      <th>cap-surface</th>
      <td>8124</td>
      <td>4</td>
      <td>y</td>
      <td>3244</td>
    </tr>
    <tr>
      <th>cap-color</th>
      <td>8124</td>
      <td>10</td>
      <td>n</td>
      <td>2284</td>
    </tr>
    <tr>
      <th>bruises</th>
      <td>8124</td>
      <td>2</td>
      <td>f</td>
      <td>4748</td>
    </tr>
    <tr>
      <th>odor</th>
      <td>8124</td>
      <td>9</td>
      <td>n</td>
      <td>3528</td>
    </tr>
    <tr>
      <th>gill-attachment</th>
      <td>8124</td>
      <td>2</td>
      <td>f</td>
      <td>7914</td>
    </tr>
    <tr>
      <th>gill-spacing</th>
      <td>8124</td>
      <td>2</td>
      <td>c</td>
      <td>6812</td>
    </tr>
    <tr>
      <th>gill-size</th>
      <td>8124</td>
      <td>2</td>
      <td>b</td>
      <td>5612</td>
    </tr>
    <tr>
      <th>gill-color</th>
      <td>8124</td>
      <td>12</td>
      <td>b</td>
      <td>1728</td>
    </tr>
    <tr>
      <th>stalk-shape</th>
      <td>8124</td>
      <td>2</td>
      <td>t</td>
      <td>4608</td>
    </tr>
    <tr>
      <th>stalk-root</th>
      <td>8124</td>
      <td>5</td>
      <td>b</td>
      <td>3776</td>
    </tr>
    <tr>
      <th>stalk-surface-above-ring</th>
      <td>8124</td>
      <td>4</td>
      <td>s</td>
      <td>5176</td>
    </tr>
    <tr>
      <th>stalk-surface-below-ring</th>
      <td>8124</td>
      <td>4</td>
      <td>s</td>
      <td>4936</td>
    </tr>
    <tr>
      <th>stalk-color-above-ring</th>
      <td>8124</td>
      <td>9</td>
      <td>w</td>
      <td>4464</td>
    </tr>
    <tr>
      <th>stalk-color-below-ring</th>
      <td>8124</td>
      <td>9</td>
      <td>w</td>
      <td>4384</td>
    </tr>
    <tr>
      <th>veil-type</th>
      <td>8124</td>
      <td>1</td>
      <td>p</td>
      <td>8124</td>
    </tr>
    <tr>
      <th>veil-color</th>
      <td>8124</td>
      <td>4</td>
      <td>w</td>
      <td>7924</td>
    </tr>
    <tr>
      <th>ring-number</th>
      <td>8124</td>
      <td>3</td>
      <td>o</td>
      <td>7488</td>
    </tr>
    <tr>
      <th>ring-type</th>
      <td>8124</td>
      <td>5</td>
      <td>p</td>
      <td>3968</td>
    </tr>
    <tr>
      <th>spore-print-color</th>
      <td>8124</td>
      <td>9</td>
      <td>w</td>
      <td>2388</td>
    </tr>
    <tr>
      <th>population</th>
      <td>8124</td>
      <td>6</td>
      <td>v</td>
      <td>4040</td>
    </tr>
    <tr>
      <th>habitat</th>
      <td>8124</td>
      <td>7</td>
      <td>d</td>
      <td>3148</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8124 entries, 0 to 8123
    Data columns (total 23 columns):
    class                       8124 non-null object
    cap-shape                   8124 non-null object
    cap-surface                 8124 non-null object
    cap-color                   8124 non-null object
    bruises                     8124 non-null object
    odor                        8124 non-null object
    gill-attachment             8124 non-null object
    gill-spacing                8124 non-null object
    gill-size                   8124 non-null object
    gill-color                  8124 non-null object
    stalk-shape                 8124 non-null object
    stalk-root                  8124 non-null object
    stalk-surface-above-ring    8124 non-null object
    stalk-surface-below-ring    8124 non-null object
    stalk-color-above-ring      8124 non-null object
    stalk-color-below-ring      8124 non-null object
    veil-type                   8124 non-null object
    veil-color                  8124 non-null object
    ring-number                 8124 non-null object
    ring-type                   8124 non-null object
    spore-print-color           8124 non-null object
    population                  8124 non-null object
    habitat                     8124 non-null object
    dtypes: object(23)
    memory usage: 1.4+ MB
    

Everything looks good so far. It seems that there are no null values in the data and the data is already relatively clean. However, While exploring the data, I noticed a problem with the stalk-root column. Instead of NaN, the data set contains '?' instead, which could be treated as a symbolic parameter in the rest of my analysis, and that would be a problem because it would confuse the machine learning algorithm.


```python
# Check to see if '?' appears in the stalk-root column
print('?' in data['stalk-root'].unique())

# let us see how many appearances of '?' exist to see how bad the problem is
from collections import Counter
dict(Counter(data['stalk-root'].tolist()))
```

    True
    




    {'e': 1120, 'c': 556, 'b': 3776, 'r': 192, '?': 2480}



There are a huge number of '?' symbols! It seems like instead of using NaNs, they have used a '?'. They might have used this in other columns and hence we will check to see if this is true.  


```python
qmark = [(column, dict(Counter(data[column]))) for column in data.columns]

# Check if '?' exists in the key of any of the symbol dictionaries
qmark_bool = map(lambda tup: (tup[0], '?' in tup[1]), qmark) 
print([tup for tup in qmark_bool if tup[1]]) # only print the tuple if '?' in keys
```

    [('stalk-root', True)]
    

From this we see that only the stalk-root column contains a '?' and given that we have many features in the data set, we will drop the stalk-root column and not use it in our analysis moving forward.


```python
data.drop('stalk-root', axis=1, inplace=True)
```

## Data visualization

To get a feel for the data and understand what we are looking at, plotting and visualizing data helps greatly. Because this data set is entirely categorical, we are limited in the types of plotting we can do. One useful outcome of the plotting exercise is to see which features differ the most between the edible mushrooms and poisonous mushrooms. For example, are many of the edible mushrooms pleasant smelling and many of the poisonous mushrooms foul smelling? Building this kind of intuition will help us along the way in our analysis, and also to see if the results of any predictive analysis makes sense.

Below we plot histograms of each property type and compare if there are any particular properties where the different between edible and poisonous mushrooms is very large.


```python
import seaborn as sns

plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=10)

fig, ax = (plt.subplots(nrows=7, ncols=3, sharey=True, figsize=(16,32), 
                        constrained_layout=True))
for (i, column) in enumerate(data.columns[1:]):
    row = i // 3
    col = i % 3
    ax_curr = ax[row, col]
    sns.countplot(x=data[column], hue='class', data=data, ax=ax_curr)
    ax_curr.legend(['poisonous', 'edible'], loc='upper right')
    if row == 4:
        ax_curr.set_ylabel('Counts')
    else:
        ax_curr.set_ylabel('')
    (ax_curr.set_xticklabels([prop_legend.loc[column].values[0][val] 
                              for val in data[column].unique()]))
```


![png](/assets/images/projects/mushroomproject/output_27_0.png)


A number of different features stand out from this analysis. We see that a lot of edible mushrooms have no odor while a lot of poisonous mushrooms have a foul odor. So we would expect this feature to play a major role in determining if a mushroom is edible or not. Similarly, other features that stand out are gill-size, gill-color, and ring-type. Armed with this intuition, we will begin a more detailed predictive analysis. 

## Naive Bayes Classifier

Our goal to be able to predict if a mushroom is edible or poisonous given it's properties. Because the data is categorical and symbolic, a Naive Bayes classifier is well suited for this task. We will use the Naive Bayes Classifier in the `nltk` package to make predictions on whether a mushroom is likely to be poisonous or edible. For this we will need to convert the data into the following format:

```
    class                           feature_dictionary
0     e     {'cap-shape': 'x', 'cap-surface': f, 'cap-color': 'n', ...}
1     p     {'cap-shape': 'x', 'cap-surface': y, 'cap-color': 'w', ...}
```

In this format, we can then supply these features to the Naive Bayes Classifier and make predictions. We will now write a function to do this.


```python
def feature_dictionary_creator(row):
    """
        input: takes in a row of the data dataframe
        returns: a series that has the format: 
            'class': 'x'
            'feature_dictionary': '{... ... ... ...}'
        the ultimate goal is to create a new dataframe with class as one
        column and feature dictionary as another column to feed to the 
        Naive Bayes Classifier
    """
    
    # We omit the class column. We will add it in
    # later as the second element of a tuple 
    prop_names = row.index[1:]  
    prop_values = row.values[1:]
    label = row.values[0]
    tuples = zip(prop_names, prop_values)
    return ({name: value for (name, value) in tuples}, label)
```

We now use a for loop to iterate through the dataframe and add a column called featureset that contains the output of `feature_dictionary_creator`


```python
featuresets = data.apply(feature_dictionary_creator, axis=1)
data['featuresets'] = featuresets
```


```python
all_features = data['featuresets'].values.flatten().tolist()
len(all_features)
```




    8124



We are now ready to use the mushroom features to input into the naive bayes classifier. We first split the dataset into a training set and a test set. We use 80% of the data as the train set and the remaining as the test set. Because the total size of the dataset is 8124 rows, we will use 6500 rows as the training set. We then proceed to implement the classifier


```python
from nltk import NaiveBayesClassifier as nbc
train_set, test_set = all_features[:6500], all_features[6500:]
```


```python
from nltk import classify
classifier = nbc.train(train_set)
classify.accuracy(classifier, test_set)
```




    0.9593596059113301



As we see above, we get an accuracy of 95%. However, let us see if we can improve the accuracy further. Let us play with the size of the test set and train set. Let us plot the model accuracy as a function of the ratio of the train set from 20% to 98%. We also shuffle the list to make sure we appropriately randomize the training.


```python
from random import shuffle

# Go from using 0.1% of the data to 99.9% of the data
ratios = np.linspace(.001, .999, 100)
accuracies = []

for ratio in ratios:
    shuffle(all_features)
    train_set, test_set = all_features[:int(8124*ratio)], all_features[int(8124*ratio):]
    classifier = nbc.train(train_set)
    accuracies.append(classify.accuracy(classifier, test_set))
```


```python
fig2, ax2 = plt.subplots(nrows=1, ncols=1)
ax2.set_xlabel('Number of rows')
ax2.set_ylabel('accuracy')
ax2.plot(np.array(ratios*8124, dtype=int), accuracies)
```




    [<matplotlib.lines.Line2D at 0x1a2106f208>]




![png](/assets/images/projects/mushroomproject/output_40_1.png)


We see that the accuracy of prediction is low when we have very few records, but increases quickly to a plateau. Interestingly, if we use too much of the data, the accuracy of the model again begins to drop, probably due to over-fitting effects. In effect, it is not necessary to train the model on more than a few hundred different types of mushrooms. 

We next look at the most informative features: if I am going hiking and I spot a mushroom in the wild, which features should I look at first to guess if the mushroom is likely to be edible or not? We go back to using about 6500 rows as the training set for this exercise.


```python
train_set, test_set = all_features[:6500], all_features[6500:]
classifier = nbc.train(train_set)
mif = classifier.most_informative_features()[:12]
classifier.show_most_informative_features(n=12)
```

    Most Informative Features
           spore-print-color = 'h'                 p : e      =     34.2 : 1.0
      stalk-color-above-ring = 'n'                 p : e      =     33.9 : 1.0
                        odor = 'n'                 e : p      =     25.7 : 1.0
    stalk-surface-above-ring = 'k'                 p : e      =     18.2 : 1.0
    stalk-surface-below-ring = 'k'                 p : e      =     16.9 : 1.0
                gill-spacing = 'w'                 e : p      =     10.0 : 1.0
                  gill-color = 'u'                 e : p      =      8.6 : 1.0
                     habitat = 'p'                 p : e      =      8.3 : 1.0
                   gill-size = 'n'                 p : e      =      8.0 : 1.0
             gill-attachment = 'a'                 e : p      =      8.0 : 1.0
                   cap-shape = 'b'                 e : p      =      7.9 : 1.0
                  gill-color = 'n'                 e : p      =      7.8 : 1.0
    


```python
# Converting symbols to actual values based on the props_legend dictionary 
# constructed earlier
x = [(tup[0],prop_legend.loc[tup[0]].values.tolist()[0][tup[1]]) for tup in mif]
x
```




    [('spore-print-color', 'chocolate'),
     ('stalk-color-above-ring', 'brown'),
     ('odor', 'none'),
     ('stalk-surface-above-ring', 'silky'),
     ('stalk-surface-below-ring', 'silky'),
     ('gill-spacing', 'crowded'),
     ('gill-color', 'purple'),
     ('habitat', 'paths'),
     ('gill-size', 'narrow'),
     ('gill-attachment', 'attached'),
     ('cap-shape', 'bell'),
     ('gill-color', 'brown')]



From this, we see conclude the following:

**A mushroom is likely to be poisonous if:**
* The spore print color is chocolate
* The stalk color above and below the ring is brown
* The stalk surface above and below the ring is silky
* The gill size is narrow
* It is found along paths

**A mushroom is likely to be edible if:**
* It has no odor
* It has an attached gill
* The gill spacing is crowded
* The gill color is purple
* The cap is bell shaped
