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

36
