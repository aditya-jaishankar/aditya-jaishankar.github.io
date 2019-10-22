---
title: "Titanic Survival Prediction"
toc: true
layout: single
permalink: /projects/Titanic/
author_profile: true
excerpt: "This is an excerpt"
read_time: true
---


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

```

 The function below does all the preprocessing to remove the unnecessary columns and
 transforming all the categorical datatypes to encoded numerical. This is needed for the
 knn algorithm as we need distance measures


```python
def processor(frame):
    # In some cases, Age is also null so I am going to drop age.
    frame = frame.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name', 'Age', 'Fare'])
    le1 = LabelEncoder()
    frame['Embarked'] = le1.fit_transform(frame['Embarked'].tolist())

    le2 = LabelEncoder()
    frame['Sex'] = le2.fit_transform(frame['Sex'].tolist())
    # frame = frame.dropna()
    min_max_scaler = MinMaxScaler()
    # To cover both the train and test cases (where no survived option is available)
    try:
        return (min_max_scaler.fit_transform(frame.drop(columns=['Survived'])),
                frame['Survived'])
    except:
        return (min_max_scaler.fit_transform(frame))

```


```python
train_data = pd.read_csv('./train.csv')
prediction_data = pd.read_csv('./test.csv')

```

 We first describe the data and see how many rows are there. How many `nans`
 are there? Is there any one column with a very high number of `nan` that we
  can delete? For this we count how many `nan` entries exist for each column


```python
train_data.describe()  # 891 rows in total
train_data.isnull().sum()  # 687 rows of 891 total in cabin are nulls, so we will drop this column.
X, y = processor(train_data)

```


```python
# We first do the pre-processing by calling the `preprocessor` function we
# wrote above.
def fitter_predictor(n_neighbors=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, shuffle=True)

    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, y_train)
    prediction = knn_classifier.predict(X_test)

    score = accuracy_score(y_test, prediction, normalize=True)
    return score

```


```python
def hyperparameter_tuner():
    # This function tires to find the optimal number of nearest neighbors to use
    start = 5
    end = 75
    list_of_n = np.arange(start, end)
    scores = np.empty_like(list_of_n, dtype=np.float)

    for i, n in tqdm(enumerate(list_of_n)):
        total = 0
        niter = 100
        for _ in range(niter):  # Do each n_neighbor a hundred times for statistics
            score = fitter_predictor(n)
            total += score
        scores[i] = total/niter
        # print(*['Score for n =', n, ' is:', scores[i]], sep=' ')

    plt.plot(list_of_n, scores)
    print('Optimal number of neighbors =', np.argmax(scores)+start)
    return None

hyperparameter_tuner()
```

    70it [00:28,  2.19it/s]
    

    Optimal number of neighbors = 28
    


![png](/assets/images/projects/titanic/output_7_2.png)


 We find that the optimal number of neighbors is between 28 and 35. This is because there is a tendency to overfit the data if we have too many neighbors. For the prediction below, we use n = 28


```python
def output_generator():
    knn_classifier = KNeighborsClassifier(n_neighbors=28)
    knn_classifier.fit(X, y)
    X_prediction = processor(prediction_data)
    prediction = knn_classifier.predict(X_prediction)

    output = pd.concat([prediction_data['PassengerId'], pd.Series(prediction)], axis=1)
    output = output.rename(columns={0:'Survived'})
    output.to_csv('./prediction.csv', index=False)
    return output
# output_generator()
```


```python

```
