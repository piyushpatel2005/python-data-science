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

21