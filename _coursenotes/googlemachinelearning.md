---
title: "Google Machine Learning Crash Course"
categories:
toc: true
layout: single
classes: wide
permalink: /coursenotes/googlemachinelearning/
author_profile: true
read_time: true
---

## Definitions and Framing

* **Supervised Machine Learning:** 
    Using known data to generate some useful predictions of on unseen data

    Label $y$: The target variable that we are trying to predict, for example 'spam' or 'not spam'. 

    Features: Something about the data that is used to represent the data, that is later fed into a model. Complicated machine learning models can have multiple features $x_1, x_2, \ldots, x_n$. So this is a vector $\mathbf{x}$.

    Model: Maps unseen data to predictions of label $y$. It is defined by internal parameters that are learned using a training set of labeled data

    Labeled data can be represented as $(\mathbf{x}, y)$. In unlabeled data, $y$ is not known and is predicted by the model. 


* **Models**

    A model defines the relationship between the features and the label. There are two key phases in the life of a model: <br>

    Training: The phase where you the model is trained or learned. You show the model a number of examples of labeled data, and allow the model to learn the relationship between the features and the label. In other words, it is learning the values of the parameters in the model. These parameters in the model are often called hyperparameters. 

    Inference: The phase where the model is used to generate labels $y'$ given features $\mathbf{x}$. 
    
    
* **Regression vs. Classification**

    A regression predicts continuous values while a classification predicts discrete values. 

## Descending into ML

Topics covered: Linear Regression, Training and Loss

* The $L_2$ loss is defined as 

$$
\begin{align}
L_2 \textrm{ Loss } = \sum\limits_i (y_i - prediction_i(\mathbf{x}))^2 
\end{align}
$$

where we sum over all data points in the training set $i$. 

* The convention in machine learning is to represent the linear model as $y' = w_1 x_1 + b$ instead of the more traditional $y = mx + b$. We could easily generalize this regression from one feature to multiple features and the prediction would be given by $y' = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$, where all the $w_i$'s are weights and all the $x_i$'s are features. The process of fitting the model is called training and the process of using the trained model to make a prediction is called inference. 

* A commonly used loss function is the mean squared error (MSE), but this is neither the only loss function nor the best or most practical loss function for all cases.

## Reducing Loss

* One of the popular ways to find the minimum in the loss function is to use Gradient Descent. We calculate a gradient at each point and move in the direction of decreasing gradient. The step size that we use as we advance in the direction of decreasing gradient is called the learning rate. This has to be chosen carefully: in multi-dimensional problems, too large a learning rate can cause the code to become unstable. 

* We can also get trapped in local minima if the space is *not-convex*. There is a whole sprawling field of non-convex optimization. 

* There are two important flavors of gradient descent:

    **Stochastic Gradient Descent:**

    In theory, while using gradient descent, we need to calculate the gradient of the loss function considering all examples. However in practice, this is found to be excessive and computationally expensive. We therefore select only one example at random and calculate the gradient of the loss function considering only that one example. This is called stochastic gradient descent. Although this might require more steps to reach the optimum, overall there is usually less computation when dealing with very large data sets. Gradient calculations can be very expensive. 

    Mathematically, we want to calculate 

    $$
    \begin{align}
    w_{n+1} = w_n -\eta Q(w)
    \end{align}
    $$

    where $\eta$ is the learning rate and $Q(w) = \frac{1}{N}\sum_i Q_i(w)$ is the $L_2$ loss for the $i$-th example. What we do instead, in stochastic gradient descent is to first randomnly pick a particular example $k$ and calculate the loss function $Q_k(w)$ and then update $w$ using:

    $$
    \begin{align}
    w_{n+1} = w_n -\eta Q_k(w)
    \end{align}
    $$

    and perform this iteratively until the minimum criterion is reached. 

    When there are multiple parameters to optimize, the gradient is of course a vector, and so is $\mathbf{w} = (w_1, w_2, \ldots)$ and we proceed in the direction of steepest gradient in steps $\eta$ large. 

    **Mini-batch Gradient Descent**

    This is very similar to stochastic gradient descent except that instead of taking only one data point, we take batches of 10 or 100 of them.  Especially with datasets that contain duplicates, enormous datasets do not contain any further information than very large datasets. Mini-batch gradient descent exploits this fact and works very well in practice. Typical values of batch-size range between 10 and 1000. Stochastic gradient descent is an extreme example of mini-batch gradient descent with batch size 1.

* The algorithm to train the model in this case is iterative: we start with some intial guesses for the parameters, compute the loss function for those values, update the values of the parameters through some scheme (with the goal of moving in the direction of lower loss), calculate the loss for the updated values of the parameters, and proceed iteratively until we achieve convergence. Convergence is usually assumed when the loss becomes below some threshold value, or if the loss function starts changing extremely slowly. 

## First steps with TensorFlow

* TensnsorFlow is a computational framework that allows you to build machine learning models. We can use both lower level APIs by defining models based on a series of mathematical operations or we could use predefined higher level APIs such as `tf.estimator`. These architectures can be linear regressors or neural networks. 

* TensorFlow consists of the following two components:

    0. A graph protocol buffer
    0. A runtime that executes the distributed graph

* The **graph protocol buffer** or protobuf takes data structures written in a text file and then generates classes in Python (or other language) that allows you to load, save and interact with the data in a user friendly way. In this sense the protobuf and the runtime are akin to Python code and a Python interpreter. 

* Because TensorFlow is built using APIs of various levels of abstraction, we have a choice of level. In general, I should chose the layer that offers the highest level of abstraction. Of course, the highest layers are also less flexible, so for some special modeling cases if I need more flexibility, I can just drop one run lower in the level of API. 

### `tf.estimator` API

The `tf.estimator` API is one of the highest level APIs that has a lot of prepackaged tools of use. Below we show some code to exemplify its use. In the example below, we are going to estimate the median housing pricebased on just one input feature. The data is from the 1990 California housing census data. Available [here](https://developers.google.com/machine-learning/crash-course/california-housing-data-description). First we perform some imports:

### Step 0: Setup, imports, loading and inspecting the data

**Note:** `tensorflow` does not work with `python 3.7.x` yet. I had `python 3.7.2` loaded on my machine, so this was a good exercise in understanding `conda environments`. It is recommended that each project have ita own *environment*, which consists of a python ditribution (which can be versioned) and a series of packages (which can also be versioned). we first use 

`conda env --name <envname>`

`-n` can be used instead of `--name`. The environment can be created with all the required packages in one line as follows (recommended):

`conda env -n tensorflowenv python=3.6 scipy numpy matplotlib scikit-learn`

Here `tensorflowenv` is the environment name that I choose to give. I then go into the new environment using

`conda activate tensorflowenv`

Now `jupyter` doesn't yet recognize this new kernel, so I need to manually go and install it (for each environment I create) using the following (see [here](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook?rq=1) for more details:)

`python -m ipykernel install --name tensorflowenv --display-name "Python (tendorflowenv)"`

Now when I invoke `jupyter notebook` in the activate environment, I can use the dropdown menu to activate the kernel with the specific python version and packages that I desire. Also note that invoking `conda install pkgname` in the activated environment only installs the `pkgname` for that particular environment. 

For more information regarding environments, see the [docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)


```python
import math
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf

from tensorflow.python.data import Dataset
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
```

We next load the data


```python
california_housing_dataframe = pd.read_csv('https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv', sep=',')

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000
```


```python
california_housing_dataframe.head()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4689</th>
      <td>-118.1</td>
      <td>34.1</td>
      <td>28.0</td>
      <td>238.0</td>
      <td>58.0</td>
      <td>142.0</td>
      <td>31.0</td>
      <td>0.5</td>
      <td>500.0</td>
    </tr>
    <tr>
      <th>7444</th>
      <td>-118.3</td>
      <td>33.9</td>
      <td>13.0</td>
      <td>2887.0</td>
      <td>853.0</td>
      <td>2197.0</td>
      <td>800.0</td>
      <td>2.9</td>
      <td>207.9</td>
    </tr>
    <tr>
      <th>14389</th>
      <td>-122.1</td>
      <td>38.0</td>
      <td>42.0</td>
      <td>2225.0</td>
      <td>367.0</td>
      <td>864.0</td>
      <td>381.0</td>
      <td>4.1</td>
      <td>172.4</td>
    </tr>
    <tr>
      <th>9089</th>
      <td>-119.0</td>
      <td>36.1</td>
      <td>20.0</td>
      <td>1042.0</td>
      <td>183.0</td>
      <td>509.0</td>
      <td>175.0</td>
      <td>3.0</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>8388</th>
      <td>-118.5</td>
      <td>34.3</td>
      <td>33.0</td>
      <td>1549.0</td>
      <td>264.0</td>
      <td>881.0</td>
      <td>289.0</td>
      <td>5.1</td>
      <td>222.9</td>
    </tr>
  </tbody>
</table>
</div>



We are now going to use `total_rooms` as an input feature to predict our target (or label) `median_house_value`.  Note that the data is at the city block level, so the feature represents the total number of rooms in the block. We are going to use the `tf.estimator` API to implement a linear regressor to model the data. The API already implements a lot of the low-level nuts and bolts of the regression (or in general, other models) so we can focus on the training, evaluating, and visualizing aspects of the process. The specific class we will use is the `tf.estimator.LinearRegressor` class. 

### Step 1: Defining the features 

There are two main classes of features: 

* **categorical:** a feature that can take on discrete values, like spam/not spam, sunny/rainy/cloudy, etc.

* **neumerical:** a feature that can take on continuous or a large number of discrete values, like price, temperature, etc. It seems that these are features that we can do arithmetic on. 

In TensorFlow, we need to define what kind of feature we are working with. In our case, because we are working with `total_rooms` in this example, it is a numerical feature. This identification of the type of feature is done using `tf.feature_column` module, which has various functions built into it. See [here](https://www.tensorflow.org/guide/feature_columns) for documentation. Feature columns only contain a description of the data and not the data itself


```python
my_feature = california_housing_dataframe[['total_rooms']]
feature_columns = [tf.feature_column.numeric_column('total_rooms')]
```

### Step 2: Defining the target

We want to predict the meadian house value given the number of rooms, so this is our target


```python
targets = california_housing_dataframe['median_house_value']
```

### Step 3: Configure the LinearRegressor

We now configure a linear regression model using `LinearRegressor`. The optimization itself is carried out using a built-in implementation of mini-batch stochastic gradient descent. We also use the `tf.contrib.estimator.clip_gradients_by_norm` functions to impose a cut off on the step size so that the gradient descent doesn't become unstable. 


```python
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
```


```python
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,
                                                optimizer=my_optimizer)
```

### Step 4: Define the input function

We now need to define an input function which tells tensorflow how to preprocess the data as well as how to batch, shuffle, and repeat it during model training. This function is a bit of a jump from what we have seen so far, so we will explore it step by step.


```python
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model with one feature
    Inputs: 
        features: a pandas dataframe of features
        targets: a pandas dataframe of targets
        batch_size: size of batches to be passed to the model
        shuffle: Boolean, whether to shuffle the training data or not
        num_epochs: number of epochs for which the data should be repeated. None means repeat indefinitely
    Returns:
        Tuple of (features, label) for the next data batch
    """
    # convert pandas data into a dict of numpy arrays
    features = {key:np.array(value) for (key, value) in dict(features).items()} # line 13
    
    # Construct a dataset and configure batching and repeating
    ds = Dataset.from_tensor_slices((features,targets)) # line 16
    ds = ds.batch(batch_size).repeat(num_epochs) # line 17
    
    # Shuffle the data if needed
    if shuffle:
        ds = ds.shuffle(buffer_size=10000) # line 21
        
    # return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next() # line 24
    return (features, labels)
```

Here is an explanation by line number of the code above. 

* **line 13:**
    `features` is a pandas dictionary, and `dict(features)` converts it into a dictionary with the column name as key and all the values in the column as pandas series of values. Therefore, the items generates an iterator of type `(feature_name, values)`, which we use to generate our feature dictionary. `values` is of type `pandas.Series`.

* **lines 16:**

    This line makes use of the `tf.data` API, which lets you take in distributed data from various sources and transform it in different ways to generate datasets for machine learning models. There are two cheif abstractions that `tf.data` introduces, one of which is the `tf.data.Dataset` module (the other is the `tf.data.Iterator` module which is used in line 24 and discussed next). Using `tf.data.Dataset`, there are two distinct ways to create a dataset:    

    1. Creating a source (for example `Dataset.from_tensor_slices()`) which constructs a dataset from one or more `tf.Tensor` objects. 
    2. Applying a transformations (for example `Dataset.batch()`) which constructs a dataset from one or more `tf.data.Datasets` objects. 
    
    A tensor object is in principle a matrix of n-dimensions, except that its value is not evaluated until called for using a `tf.run()` command. The objects can be passed into operations to yield other tensor objects, but again, actual evaluation of the numerical value of the tensor is held off until later. This enables the *flow* of a tensor through various operations without numerical evaluation. With this context, what line 16 is doing is the following: 
    
    `Dataset.from_tensor_slices()` returns a dataset for each row of the input tensor. For example the code snippet below would return `[1, 2], [3, 4]` for the variable `ds`. In our case, we provide a tuple of the form `(features, targets)`, where `features` is a dictionary, and `targets` is a number, `from_tensor_slices` would essentially unpack this tuple, to provide multiple datasets, each of the form `({key: values}, target)`. In other words, for this dataset, if I were to print the Dataset object (with an appropriate `tf.Session.run()` call) I would see something like `({lattitide: -118, longitude: 30, ... , households: 600, ...}, <median_house_value>)`. It takes a row, converts it into a dictionary of the form `{column name: value}` an then constructs a iteratable and transformable `tf.Dataset` object which has a dictionary of features associated with the `target`. The code snippet in the next cell below should make things clear regarding the output format of the `tf.Dataset` call in this case. 
    
```python
    t = tf.constant([[1, 2], [3, 4]])
    ds = tf.data.Dataset.from_tensor_slices(t)   # [1, 2], [3, 4]
```


```python
df_test = pd.DataFrame()
df_test['col1'] = [0, 1]
df_test['col2'] = [2, 3]
t = [100, 200]
x = {key: value for (key, value) in dict(df_test).items()}
print(x)
ds_test = Dataset.from_tensor_slices((x, t))
tensor_test = ds_test.make_one_shot_iterator().get_next()
with tf.Session() as session:
    print(session.run(tensor_test))
    print(session.run(tensor_test))
```

    {'col1': 0    0
    1    1
    Name: col1, dtype: int64, 'col2': 0    2
    1    3
    Name: col2, dtype: int64}
    ({'col1': 0, 'col2': 2}, 100)
    ({'col1': 1, 'col2': 3}, 200)


### Step 5: Train the model

We now call `train()` on `linear_regressor` to train the model. 


```python
_ = linear_regressor.train(
    input_fn=lambda: my_input_fn(my_feature, targets),
    steps=100)
```

The use of the `lambda` function is somewhat unclear to me. It seems that it ensures that the input function is truly randomized? Or that datasets are correctly paired to models?

### Evaluate the model

We now just make predictions on the training data to see how well the model fit it during training. 


```python
# Create an input function for predictions
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict on the linear_regressor to make predictions
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

#Format predictions as a numpy array so we can calculate error metrics
predictions = np.array([item['predictions'][0] for item in predictions])

#Calculate the mean squared error and the root mean squared error
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
```

    Mean Squared Error (on training data): 56367.025
    Root Mean Squared Error (on training data): 237.417


To know if this mse and rmse is any good, we compare the rmse to the difference between the min and max of our targets


```python
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)
```

    Min. Median House Value: 14.999
    Max. Median House Value: 500.001
    Difference between Min. and Max.: 485.002
    Root Mean Squared Error: 237.417


This seems really large - the rmse is nearly 50% of the max difference. Let us first look at some summary statistics on how well the model is doing.


```python
calibration_data = pd.DataFrame()
calibration_data['predictions'] = pd.Series(predictions)
calibration_data['targets'] = pd.Series(targets)
calibration_data.describe()
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
      <th>predictions</th>
      <th>targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17000.0</td>
      <td>17000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.1</td>
      <td>207.3</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.1</td>
      <td>116.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.1</td>
      <td>119.4</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.1</td>
      <td>180.4</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.2</td>
      <td>265.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.9</td>
      <td>500.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting the data

sample = california_housing_dataframe.sample(n=300)

# Get the min and max total_rooms values.
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r') # Plotting the line using just 2 points

# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()
```


![png](/assets/images/coursenotes/GoogleML/output_27_0.png){: width="75%" .align-center}


The rest of the exercise goes into a discussion on tweaking the hyperparameters to overcome this por fit. With a suitable choice of `learning_rate`, `steps` and `batch_size` we can get a reasonably good fit. The parameter `steps` is just the number of times the mini-batch gradient descent attempts to go towards the minimum, and `batch_size` is the number of data points the gradient descent optimizer is optimizing over per step, so `steps * batch_size` is just the total number of data points involved in the optimization (not necessary all unique because the data points are randomly sampled)

### Aside: Synthetics features and outliers

In this set of tasks, we are going to create a synthetic feature that is the ratio of two other features and we are going to use this new feature to train the linear regression model. We are also going to look at how we can remove outliers to improve the effectiveness of the model. We are going to use the same `my_input_fn` defined above to generate datasets. However let us define a new function `train_model` to train the model


```python
def train_model(learning_rate, steps, batch_size, input_feature):
    """
    Args:
        learning_rate: A `float`, the learning rate.
        steps: A non-zero `int`, the total number of training steps. A training step
               consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        input_feature: A `string` specifying a column from `california_housing_dataframe`
                       to use as input feature.
      
    Returns:
        A Pandas `DataFrame` containing targets and the corresponding predictions done
        after training the model.
  """
    periods = 10 # line 15
    steps_per_period = steps / periods
    
    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]].astype('float32')
    my_label = 'median_house_value'
    targets = california_housing_dataframe[my_label].astype('float32')
    
    # Create input functions
    training_input_fn = lambda: my_input_fn(my_feature_data,
                                            targets,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_epochs=None)
    predict_training_input_fn = lambda: my_input_fn(my_feature_data, # line 29
                                                   targets,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_epochs=1)
    
    # Create feature columns
    feature_columns = [tf.feature_column.numeric_column(my_feature)] # line36
    
    # Create a LinearRegressor object
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,
                                                   optimizer=my_optimizer)
    
    # Plotting commands to plot our model's fit after each period
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
    
    # We now train the model inside a loop so that we can periodically
    # assess loss metrics
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(periods):
        linear_regressor.train(input_fn=training_input_fn,
                              steps=steps_per_period)
        # Compute predictions
        predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions]) # line 64
        
        # Compute mean squared error
        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))
        
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        
        # Finally, track the weights and biases over time.
        
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[my_label].max()])
        
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0] # line 76
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights') # line 77
    
        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period]) 
    print("Model training finished.")
    
    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # Create a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    calibration_data.describe()

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
  
    return calibration_data
```

Again, we study all the lines that introduce something new.

* **line 29:** 

    Here, we define a `predict_training_input_fn`. This is used when we call `linear_regressor.predict()` later, which needs an `input_fn` of its own. This is because we need a function that constructs the features that is fed into the `linear_regressor.predict()` in order to generate a prediction for the label. The `LinearRegressor` object is of course blind to the targets, and makes a prediction based on the outcome of the training phase. Therefore, the output of the `predict_training_input_fn` has to be a `tf.data.Dataset` object. However, this function is different from `training_input_function` in a few ways.
    
    1. `batch_size=1` because I create a prediction of a set of features one at a time
    2. `shuffle=False`because there is no need toa shuffle a batch of size 1.
    3. `num_epochs=1` is important. `tf.data.Dataset.repeat(count)` will throw an out of range error (`tf.errors.OutOfRangeError`) when the number of times the dataset is repeated exceeds `count`. In our case, `num_epochs` is used in place of count. `tf.LinearRegressor.predict()` continues to predict until an out of range error is thrown. So in the prediction case, we use this to ensure that the prediction is made 1 set of features at a time.

* **line 36:**

    Defining a feature column which is like a bridge, or intermediary between raw data and estimators. It transforms input data into formats that estimators can understand, giving a lot of richness to have input data speak with estimators. See the [docs](https://www.tensorflow.org/guide/feature_columns) for the different kind of feature columns, of which `numeric_column` is one.
    
* **line 64:**

    Some thoughts on `linear_regressor.predict()`. The output of this function returns a generator which iterators over dictionaries. Each dictionary is of the form, for example, `{'predictions': array([15.93631], dtype=float32)}`. For each row of the form `(features, label)`, there is an entry in the generator object.
    
* **line 66, 67:**

    `tf.estimator.get_variable_names()` gives me the names of the variables in the `tf.estimator` object. In this case, our estimator happens to be a `LinearRegressor` object. For example, calling `linear_regressor.get_variable_names()` outputs:
    
    ```
    ['global_step',
     'linear/linear_model/bias_weights',
     'linear/linear_model/total_rooms/weights']
    ```
    
    We can now get the values of these variables by calling the `get_variable_value(name)` method where `name` is one of the names returned by `get_variabale_names()`. For example,
    ```python
    linear_regressor.get_variable_value('linear/linear_model/{}/
                                        weights'.format('total_rooms'))
    ```
    returns `array([[4.999998e-05]], dtype=float32)` as the output.



#### Task 1: Creating my own synthetic feature

The idea here is that `total_rooms` and `population` are both reported on a per city block basis. But some city blocks might have a lot of people for the same number of rooms, so the number of rooms per person is in a sense a better indicator of house value. So perhaps that would make a better feature.


```python
california_housing_dataframe["rooms_per_person"] = (california_housing_dataframe['total_rooms'] / 
                                                   california_housing_dataframe['population'])

calibration_data = train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person"
)
```

    Training model...
    RMSE (on training data):
      period 00 : 212.73
      period 01 : 189.62
      period 02 : 171.05
      period 03 : 153.35
      period 04 : 141.30
      period 05 : 135.36
      period 06 : 131.91
      period 07 : 130.58
      period 08 : 130.18
      period 09 : 130.61
    Model training finished.
    Final RMSE (on training data): 130.61



![png](/assets/images/coursenotes/GoogleML/output_33_1.png){: width="100%" .align-center}


#### Task 2: Identify outliers

When we plot the predictions against the actual `median_house_value`, ideally we should see a straight line with slope 1. Do we really see that, and does that help us identify outliers?


```python
plt.scatter(calibration_data['targets'], calibration_data['predictions'])
```




    <matplotlib.collections.PathCollection at 0x1a3a855cc0>




![png](/assets/images/coursenotes/GoogleML/output_35_1.png){: width="75%" .align-center}


A histogram might help to see how the data is distributed


```python
california_housing_dataframe['rooms_per_person'].hist(bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a3a8324a8>




![png](/assets/images/coursenotes/GoogleML/output_37_1.png){: width="75%" .align-center}


It looks like there are some small set of data points for `rooms_per_person` that is very large and might be throwing off the fit, so we will clip those


```python
criterion = california_housing_dataframe['rooms_per_person'] < 4
california_housing_dataframe = california_housing_dataframe[criterion]
california_housing_dataframe.hist('rooms_per_person')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x1a3ce3d320>]],
          dtype=object)




![png](/assets/images/coursenotes/GoogleML/output_39_1.png){: width="75%" .align-center}


This looks much more reasonable. We now follow the fitting procedure defined above by calling `train_model`


```python
calibration_data = train_model(learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person"
)
```

    Training model...
    RMSE (on training data):
      period 00 : 213.27
      period 01 : 189.60
      period 02 : 167.16
      period 03 : 147.02
      period 04 : 130.93
      period 05 : 118.40
      period 06 : 111.94
      period 07 : 109.36
      period 08 : 107.38
      period 09 : 105.99
    Model training finished.
    Final RMSE (on training data): 105.99



![png](/assets/images/coursenotes/GoogleML/output_41_1.png){: width="100%" .align-center}



```python
# Replot the historgram and the parity plot to see if the plot looks any better now

fig, axs = plt.subplots(1, 2, figsize=(10,4))
axs[0].hist(california_housing_dataframe['rooms_per_person'])
axs[1].scatter(calibration_data['targets'], calibration_data['predictions'])
```




    <matplotlib.collections.PathCollection at 0x1a3fcfad68>




![png](/assets/images/coursenotes/GoogleML/output_42_1.png){: width="100%" .align-center}

