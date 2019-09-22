# Tensorflow

When solving a business problem, you may need many machine learning models. Avoid trap of creating single monolithic ML model for a large problem. Break down the problem into smaller chunks and create machine learning model for each of them. For example, a product of reading an image through camera and translating that into different languages might need following models.
1. Model to read Images
2. OCR character recognition
3. Identify the language in the photo
4. Translate the sign
5. Superimpose the translated text

## Intro
Deep neural networks are all about networks of neurons, with each neuron learning to do its own operation as part of a larger picture. Tensors are the standard way of representing data in deep learning. Tensors are multi-dimensional arrays. In Tensorflow, computation is approached as a dataflow graph. In graph, nodes represent operations (addition, multiplication,etc.) and edges represent data (tensors) flowing around the system. It is a software framework for numerical computations based on dataflow graphs. It can run on cloud or single machine. The core is written in C++ and have high-level frontend languages and interfaces for executing graphs. In deep learning, networks are trained with a feedback process called **backpropagation** based on gradient descent optimization. Tensorflow supports many optimiation algorithms. To monitor, debug and visualize the training, it comes with TensorBoard, a visualization tool that runs in the browser. There are some abstraction libraries like Keras and TF-slim which allows for complex machine learning tasks to be trivial. Tensorflow comes with many features for boosting scalability including asynchronous computation, efficient IO and data formats, etc.

To install Tensorflow, execute `pip install tensorflow`. Better option is to install this on virtual environment.

```shell
pip install virtualenv
cd ~
mkdir envs
virtualenv ~/envs/tensorflow
source ~/envs//bin/activate # enable created virtual environment
pip install tensorflow
import tensorflow as tf
print(tf.__version__)
deactivate # deactivate virtual environment
alias tensorflow="source ~/envs/tensorflow/bin/activate"
```

[Hello world](examples/hello_world.py)

In Tensorflow, the evalutions are lazy, so if we don't run the execution in the context of a session, it will print Tensor and not the final result. `Session` object acts as an interface to the external Tensorflow computation mechanism and allows to run parts of computation graph we have defined. `sess.run(hw)` actually computes the value of `hw`.

## Digit recognition system

With this simple example, we will not use the spatial information in the pixel layout in the image, but in reality, they are very important.

[Digit recognition simple example](examples/softmax_regression.py)

A large class of machine learning tasks is that we would like to learn a function from data examples to their known labels. This is called *supervised learning*. The measure of similarlity between true and predicted values in classification is called *cross entropy*. We try to minimize the loss function.

When testing performance on a test set, if all the test examples (here data.test.images) are not able to fit into memory in the system, we will get memory error at this point. The easy way around this is to split the test procedure into batches, just like training.

## Tensorflow Basics

Tensorflow is a numerical computation library using dataflow graphs.

### Computation Graphs

Tensorflow allows us to implement machine learning by creating and computing operations that interact with one another. These interactions form a **computation graph**. In dataflow graph, the edges allow data to flow from one node to another in a direct manner. In Tensorflow, each nodes represent operation (including all kinds of functions). Tensorflow (TF) optimizes its computations based on graph's connectivity. Being able to locate depenedencies between units of model allwos to both distribute computations across available resources and avoid performing redundant computations irrelevant subsets, resulting in a faster and efficient way of computing things.

#### Creating a graph

Once we import Tensorflow, a specific empty default graph is formed. All nodes we create are automatically associated with  that default graph. Using `tf.<operator>`, we create nodes. For some arithmetic and logical operati0ns,   we can use shortcuts instead of having to `tf.<opeartor>` syntax. For example, we can use */+/- instead of `tf.multiply()/tf.add()/tf.subtract()`.

| TF operator | Shortcut | Description |
|:-----------|:----------:|:-------------|
| tf.add() | a + b | Adds a and b, element wise |
| tf.multiply() | a * b | multiplies a and b, element-wise |
| tf.subtract() | a - b | subtracts a from b, element-wise |
| tf.divide() | a / b | computes python-style division of a by  b |
| tf.pow() | a ** b | returns the results fo raising each element in a to its corresponding element b |
| tf.mod() | a % b | returns element-wise modulo. |
| tf.logical_and() | a & b | returns the truth table of a & b, element-wise. dtype must e tf.bool |
| tf.greater() | a > b | returns the truth table of a > b, element-wise |
| tf.greater_equal() | a >= b | retursn a >= b, element-wise |
| tf.less_equal() | a <= b | returns the truth table of a <= b |
| tf.less() | a < b | returns a < b |
| tf.negative() | -a | returns negative value of each element in a |
| tf.logical_not() | ~a | returns logical NOT of each element in a. Only compatible with Tensor objects with dtype tf.bool. |
| tf.abs() | abs(a) | returns absolute value of a |
| tf.logical_or() | `a | b` | returns logical or operation of a and b. |

Once we have described our computation graph, to run the graph, we need to create and run a session.

```python
sess = tf.Session()
outs = sess.run(f)
sess.close()
print("outs = {}".format(outs))
```

When computation is completed, it is good to close the session to release the resources. Default graph is automatically created and if we want to create additional graphs with some given operations we can use `tf.Graph()` to create it. As the new graph is not assigned as default graph, any operations we create will not be associated with it, but rather with the default graph.

```python
import tensorflow as tf
print(tf.get_default_graph())
g = tf.Graph()
print(g)
a = tf.constant(5)
print(a.graph is g) # print which graph is associated with a, False
print(a.graph is tf.get_default_graph()) # True
```

The `with` statement in Python is used to execute some setup code and then always tearing it down at the end regardless of success or failure. We don't need to call `close()` method on session explicitly with this.

```python
g1 = tf.get_default_graph()
g2 = tf.Graph()
print(g1 is tf.get_default_graph()) # True
with g2.as_default():
  print(g1 is tf.get_default_graph()) # False
print(g1 is tf.get_default_graph()) # True
```

We request one specific node using `session.run()` method. Whatever we assign to this method is called **fetches**, corresponding to the elements of the graph we wish to compute. We can ask multiple nodes' outputs by inputting a list of requested nodes.

```python
with tf.Session() as sess:
  fetches = [a,b,c,d,e]
  outs = sess.run(fetches)
print("outs: {}".format(outs))
print(type(outs[0])) # <type 'numpy.int32'>
```

Tensorflow computes only the essential nodes according to the set of dependencies.

### Flowing Tensors

When we construct a node in the graph, we are actually creating an operation instance. These operations reference their to-be-computed result as a handle that can be passed on - flow - to another node. These handles, which are edges in graph, are referred to as Tensor objects. This is how its named 'Tensorflow'. Tensor objects have methods and attributes that control their behavior and can be defined upon creation. `tf.<operator>` function could be thought of as a constructor, but this is actually not a constructor, but a factory method that sometimes does quite a bit more than just creating the operator objects. Each Tensor object has attributes such as name, shape and dtype that help identify and set characteristics of that object. 

The basic units of data that pass through a graph are numerical, Boolean or string elements. We can explicitly specify the parameter when we create tensor objects using operators.

```python
c = tf.constant(4.0, dtype=tf.float64)
print(c)
print(c.dtype)
```

Data types are important because operation with two non-matching data types will result in an exception. We can cast datatype using `tf.cast()` operation. Data types include `tf.float32`, `tf.uint8` (unsigned), `tf.complex64` (complex number made from 32 bit floating points), `tf.qint32` (32-bit signed integer used in quantized ops), etc.

```python
x = tf.constant([1,2,3], name='x', dtype=tf.float32)
print(x.dtype)
x = tf.cast(x, tf.int64)
print(x.dtype)
```

*1 x 1* tensor is a scala, a *1 x n* tensor is a vector, *n x n* tensor is a matrix. We can use numpy arrays to initialize high dimensional tensors.

```python
import numpy as np
c = tf.constant([1,2,3],
                [4,5,6])
# get_shape() method returns the shape of the tensor as a tuple of integers
print("Python list input: {}".format(c.get_shape())) # (2,3)
c = tf.constant(np.array([
  [[1,2,3],
  [4,5,6]],
  [[1,2,1],
  [2,2,2]]
]))
print("3d numpy array input: {}".format(c.get_shape())) # (2,2,3)
```

We can generate random numbers from a normal distribution using `tf.random.normal()`, passing the shape, mean and standard deviation as arguments. Sequence generator `tf.linspace(a,b,n)` creates n evenly spaced values from a to b. If we want to explore data content of an object, we can use `tf.InteractiveSession()` and the `.eval()` method allows us to check the contents of some tensor.

```python
sess = tf.InteractiveSession()
c = tf.linspace(0.0, 4.0, 5)
print("The contents of C:\n {}\n".format(c.eval()))
sess.close()
```

| Tensorflow operation | Description |
|:------------------|:--------------|
| tf.constant(value) | Creates a tensor populated with the value specified in argument. |
| tf.fill(shape, value) | Creates a tensor with shape and fills in with values. |
| tf.zeros(shape) | returns a tensor of given `shape` with all elements 0. |
| tf.zeros_like(tensor) | returns tensor of same type and shape with elements set to 0. |
| tf.ones(shape) | returns tensor of given shape with elements 1. |
| tf.ones_like(tensor) | returns a tensor of the same type and shape as `tensor` with all elements 1 |
| tf.random_normal(shape, mean, stddev) | outputs random values from a normal distribution |
| tf.truncated_normal(shape, mean, stddev) | outputs random values from a truncated normal distribution ( values whose magnitude is more than two standard devications from the mean are dropped and re-picked) |
| tf.random_uniform(shape, minval, maxval) | generates values rom a unifrom distribution in the range [minval, maxval) |
| tf.random_shuffle(tensor) | randomly shuffles a tensor along its first dimension |
| tf.random_crop(value, size, seed=None, name=None) | crop out a contiguous block of shape from the tensor. |

Matrix multiplication is performed via the `tf.matmul(A,B)` for two Tensor objects A and B.

```python
A = tf.constant([1,2,3],
                [4,5,6])
print(a.get_shape()) # (2,3)

x = tf.constant([1,0,1])
print(x.get_shape()) # (3,)
```

We can add another dimension by passing `tf.expand_dims()` with the position of the added dimension as the second argument.

```python
x = tf.expand_dims(x,1)
print(x.get_shape()) # (3,1)
b = tf.matmul(A,x)

sess = tf.InteractiveSession()
print('matmul result:\n{}'.format(b.eval()))
sess.close()
```

If we want to flip an array, we can use `tf.transpose()` function. We can use `.name` attribute to see the name of the object. Objects within the same graph cannot have the same name. It will automatically add underscore and number to distinguish the two.

```python
with tf.Graph().as_default():
  c1 = tf.constant(4, dtype=tf.float64, name='c')
  c2 = tf.constant(4, dtype=tf.int32, name='c')
print(c1.name) # c
print(c2.name) # c_1
```

If we want to create node grouping to make it easier for complicated graph, we can group nodes together by name using `tf.name_scope("prefix")`. Prefixes are useful when we would like to divide a graph into subgraphs with some meaning.

```python
with tf.Graph().as_default():
  c1 = tf.constant(4, dtype=tf.float64, name='c')
  with tf.name_scope("prefix_name"):
    c2 = tf.constant(4, dtype=tf.int32, name='c')
    c3 = tf.constant(4, dtype=tf.float64, name='c')
print(c1.name) # c
print(c2.name) # prefix_name/c
print(c3.name) # prefix_name/c_1
```

Tensorflow uses variables to optimize the process (tune the parameters). We can create variable using `tf.Variable()` function and then have to explicitly initialize using `tf.global_variables_initializer()` method which allocates memory for the variable and sets its initial values. Variables are computed only when the model runs.

```python
init_val = tf.random_normal((1,5), 0, 1)
var = tf.Variable(init_val, name='var')
print("pre run: \n{}".format(var))  # Tensor("var/read:0", shape(1,5), dtype=float32)
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  post_var = sess.run(var)
print("\npost run: \n{}".format(post_var)) 
```

If we want to reuse the same variable, we can use the `tf.get_variables()` function. TF has built-in structures for feeding input values called **placeholders**. Placeholders are empty Variables that will be filled with data later on. If `shape` argument is not specified, placeholder can be fed with data of any size. Whenever we define a placeholder, we must feed it with some input values or else an exception will be thrown. The input data is passed to the `session.run()` method as a dictionary.

```python
ph = tf.placeholder(tf.float32, shape=(None, 10))
sess.run(s, feed_dict={x: X_data, w: w_data})
```

```python
x_data = np.random.randn(5, 10)
w_data = np.random.randn(10, 1)
with tf.Graph().as_default():
  x = tf.placeholder(tf.float32, shape(5,10))
  w = tf.palceholder(tf.float32, shape(10,1))
  b = tf.fill((5,1), -1.)
  xw = tf.matmul(x,w)
  xwb = xw + b
  # s gets the maximum value of that vector by using reduce_max operation.
  s = tf.reduce_max(xwb)
  with tf.Session() as sess:
    outs = sess.run(s, feed_fact={x: x_data, w: w_data})
print("outs = {}".format(outs))
```

Let's say we have target variable y and we want to explain using feature vector x. We will need to create placeholders for our input and output data and variables for our weights and intercept.

```python
x = tf.placeholder(tf.float32, shape=[None, 3])
y_true = tf.placeholder(tf.float32, shape=None)
w = tf.Variable([[0,0,0]], dtype=tf.float32, name='weights')
b = tf.Variable(0, dtype=tf.float32, name='bias')
y_pred = tf.matmul(w, tf.transpose(x)) + b # matrix multiplications of input container x and weights w plus a bias b
```

To measure the discrepancy between our model's predictions and the observed targets, we need to measure distance referred to as **loss function** and need to optimize the parameters (weights and  bias) that minimize it. The loss is measured as MSE (mean square error) which is found by taking difference of *y_true* and *y_pred* and squaring the value.

```python
loss = tf.reduce_mean(tf.square(y_true - y_pred))
```

Another common loss functions is *cross entropy* which is a measures of similarity between two distributions and used mostly in classification. We can compare the true class with the probabilities of each class given by the model. The more similar the two distributions, the smaller our cross entropy will be.

Next, we need to figure out how to minimize the loss function. Mostly, we need to use optimizers to update the set of weights iteratively in a way that decreases the loss over time. The most common is **gradient descent**. The steepest direction of decrease of gradient is obtained by moving from a point in the direction of the negative gradient. It is suitable for a wide variety of problems. Convergence to global minimum is guaranteed for convex functions, for nonconvex problems they can get stuck in local minima.

Commonly, the stochastic graident descent (SGD) is used where instead of feeding entire dataset to the algorithm for the computation, a subset of the data is sample sequentially at each step. The number of samples range from one sample at a time to a few hundred. Using smaller batches usually works faster. Using a relatively smaller batch size is effectively the preferred approach. Tensorflow makes it easy to use gradient descent. Tensorflow automatically computes the gradients on its own, deriving them from the operations and structure of the graph. An important parameter is *learning rate*, determining how aggressive each update iteration will be. If this is large, we will overshoot the target and never reach minima.

```python
# We create optimizer by using GradientDescentOptimizer() function with learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# create train operation that updates our variables by calling optimizer.minimize() passing loss as argument.
train = optimizer.minimize(loss)
# train operation is executed when it is fed to `sess.run()` method.
```

```python
# Linear Regression example
import numpy as np
# create data for simulation
x_data = np.random.randn(2000, 3)
w_real = =[0.3, 0.5, 0.1]
b_real = -0.2
noise = np.random.randn(1,2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise
# Estimate the bias b and weights w by optimizing the model.
NUM_STEPS = 10
g = tf.Graph()
wb_ = []
with g.as_default():
  x = tf.placeholder(tf.float32, shape=[None, 3])
  y_true = tf.placeholder(tf.float32, shape=None)

  with tf.name_scope('inference') as scope:
    # Initialize both variables with zero values.
    w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
    b = tf.Variable(0, dtype=tf.float32, name='bias')
    y_pred = tf.matmul(w, tf.transpose(x)) + b
  
  with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.square(y_true - y_pred))

  with tf.name_scope('train') as scope:
    learning_rate = 0.5
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
  
  # Before starting, initialize variables.
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
    for step in range(NUM_STEPS):
      sess.run(train, {x: x_data, y_true: y_data})
      if(step % 5 == 0):
        print(step, sess.run([w, b]))
        wb_.append(sess.run([w,b]))
    print(10, sess.run([w,b]))
```

```python
# Logistic Regression example
N = 20000

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
# create data to simulate results
x_data = np.random.randn(N, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2
wxb = np.matmul(w_real, x_data.T) + b_real
y_data_pre_noise = sigmoid(wxb)
y_data = np.random.binomial(1, y_data_pre_noise)
y_pred = tf.sigmoid(y_pred)
loss = y_true * tf.log(y_pred) - (1 - y_true) * tf.log(1 - y_pred)
loss = tf.reduce_mean(loss)

NUM_STEPS = 50
with tf.name_scope('loss') as scope:
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
  loss = tf.reduce_mean(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  for step in range(NUM_STEPS):
    sess.run(train, {x: x_data, y_true: y_data})
    if(step % 5 == 0):
      print(step, sess.run([w,b]))
      wb_.append(sess.run([w,b]))
  print(50, sess.run([w,b]))
```

## Convolutional Neural Networks

The difference between fully connected and convolutional neural networks is the pattern of connections between consecutive layers. In fully connected case, each unit is connected to all of the units in the previous layer. In a convolutional layer, each unit is connected to a number of nearby units in the previous layer. All units aer connected to the previous layer in the same way, with exact same weights and structure. This leads to an operation known as convolution. In convolutional neural networks, each layer looks at an increasingly larger part of the image as we go deeper into the network. Convolutional structure can be seen as a regularization mechanism. **Regularization** is used to refer to the restriction of an optimization problem by imposing a penalty on the complexity of solution, in the attempt to prevent overfitting to the given examples.

Convolution is the fundamental means by which layers are connected in convolutional neural networks. We build it using `conv2d()` method.

```python
tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') # x is data
```

We stack convolutional layers hierarchically, and *feature map* is a common term referring to the output of each such layer. Suppose we have unknown number of images, each 28x28 pixels with one color channel (grayscale images). In images that have multiple color channels (RGB), we regard each images as a three dimensional tensor of RGB values, but in one channel data, they are just two dimensional and convolutional filters are applied to two-dimensinoal regions. Setting padding to `SAME` means that the borders of x are padded such that the size of the result of the operation is the same as the size of x. The strides value here means that the filter is applied to the input in one-pixel intervals in each dimension, corresponding to full convolution.

Following linear layers, whether convolutional or fully connected, it is common to apply nonlinear activation functions. One aspect of activation functions is thata consecutive linear operations can be replaced by a single one and thus depth doesn't contribute to the expressiveness of the model unless we use nonlinear activations between the linear layers.

### Pooling

Pooling means reducing the size of the data with some local aggregation function, typically written within each feature map. Pooling reduces the size of the data to be processed downstream and can reduce the number of parameters in the model, especially if we use fully connected layers after convolutional ones. Also, we would like our computed features not to care about small changes in position in an image. For instance, feature looking for eyes in the top-right part should not change too much if we move the camera a bit to the right when taking picture. Aggregating eye-detector feature spatially allows the model to overcome such spatial variability between images.

`tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')`

Max pooling outputs the maximum of the input in each region of a predeined size. *ksize* controls the size of pooling and the *strides* controls by how much we slide the pooling grids across x.

### Dropout

Drop is a regularization trick used to force the network to distribute the learned representation across all neurons. It "turns off" a random preset fraction of the units in a layer by setting their values to zero during training. These dropped-out neurons are random, forcing the network to learn a representation that will work even after the dropout. This process is often thought of as training an "ensemble" of multiple networks, thereby increasing generalization. During using network as a classifier at test time (inference), there is no dropout and full network is used as is.

`tf.nn.dropout(layer, keep_prob=keep_prob)` where *keep_prob* is fraction of the neurons to keep working at each step.

57