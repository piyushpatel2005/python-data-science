# Intro to Tensorflow on GCP

```shell
rm -rf earthquake.csv
# download latest data from USGS
wget http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.csv -O earthquakes.csv
gsutil cp earthquakes.* gs://<YOUR-BUCKET>/earthquakes/
# publish cloud files to the web
gsutil acl ch -u AllUsers:R gs://<YOUR-BUCKET>/earthquakes/*
# get compute zones names
gcloud compute zones list
datalab create mydatalabvm --zone <ZONE> # create datalab instance in the zone of your data
# This takes some time, once ready you'll see Web preview enabled in Cloud shell
# If cloud shell is closed for some reason as they are ephermeral. We can reconnect to the same datalab instance using
datalab connect mydatalabvm
# Enable Cloud Source API
```

[Reading CSV files](earthquakes.ipynb)

Check [flights notebook](flights.ipynb) to see invocation of BigQuery and getting results in Pandas dataframes.

[Invoking already trained machine learning API](mlapis.ipynb) on GCP is easy.

Creating repeatable dataset is an important skill for ML engineers. You should divide the dataset into three parts, development and training, validation and test dataset. This datasets should be independent of bias and if we split the dataset based on one of the features, that feature will have lose its predictability. Before dividing the data into such training and testing datasets, it is important that the data is cleaned.

Check [Datasets creation](repeatable_splitting.ipynb)

Also, benchmarking is an important consideration to stop overfitting of the model. Overfitted model may not generalize beyond training datasets.

[Benchmarking](create_datasets.ipynb)

Tensorflow works in lazy evaluation mode. Tensorflow is an open-source, high performance library for numerical computation that uses directed graphs. We create DAG to represent computation. The nodes represent mathematical opeartion. The edges between the nodes represent arrays of data. 3-D array is called 3-D tensor and 4D tensor, etc.

DAG is a language and hardware independent representation of model. Tensorflow is written in C++ but we can write code in Python.

```python
c = tf.add(a, b) # doesn't execute, it builds the DAG
# a,b,c are tensor and add is opeartion
session = tf.Session()
numpy_c = session.run(c)

a = tf.constant([5,3,8])
b = tf.constant([3,-1,2])
c = tf.add(a, b)
print c
with tf.Session() as sess:
  result = sess.run(c)
  print result
```

`tf.eager` allows you to execute operations eagerly. Eager mode can be useful for development purposes.

```python
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

tfe.enable_eager_execution()
x = tf.constant([3,5,7])
y = tf.constant([1,2,3])
print (x-y) # tf.Tensor([2,3,4], shape=(3,), dtype=int32)
```

DAG consists of tensors and operations. Tensorflow can optimize code if the evaluation is lazy. Session object represents connection between python code and lower level C++ api. Instead of `sess.run(z)`, we can also call `z.eval()` which will evaluate `z` tensor in the context of default session. We can also pass list of tensors to evaluate.

```python
import tensorflow as tf
x = tf.constant([3,5,7])
y = tf.constant([1,2,3])
z1 = tf.add(x, y)
z2 = x * y # short cut for common arithmetic operations on tensors
z3 = z2 - z1

with t.Session() as sess:
  a1, a3 = session.run([z1, z2]) # evaluate two tensors to be evaluate, it can accept a list
  print a1
  print a3
```

If you want to visualize the graph, we can use `tf.summary.FileWriter`.


```python
import tensorflow as tf
x = tf.constant([3,5,7], name="x") # name tensors and the operations
y = tf.constant([1,2,3], name="y")
z1 = tf.add(x,y, name="z1")
z2 = x * y
z3 = z2 - z1
with tf.Session() as sess:
  with tf.summary.FileWriter('summaries', sess.graph) as writer: # write the session graph to a summary dictionary
    a1,a3 = sess.run([z1,z3])
```

To visualize the graph, we use tensorboard.


```python
from google.datalab.ml import TensorBoard
TensorBoard().start('./summaries')
# open the url specified and switch to graphs section and you can see graph
```

```shell
tensorboard --port 8080 --logdir gs://${BUCKET}/${SUMMARY_DIR}
```

Tensor is n dimension array of data.

```python
tf.constant(3) # Shape(), Rank 0
tf.constant([3,4,5]) # Shape(3,), Vector, Rank 1
tf.constant([3,5,7], [4,6,8]) # (2,3), Matrix, Rank 2
tf.constant([[[3,4,7], [4,6,8]],
             [[1,2,3], [4,5,6]]
            ])  # (2,2,3), 3D Tensor, Rank 3
# We can stack tensors programmatically
x1 = tf.constant([2,3,4]) # (3,)
x2 = tf.stack([x1, x1]) # (2,3)
s3 = tf.stack([x2, x2, x2, x2]) (4,2,3)

# Tensors can be sliced
import tensorflow as tf
x = tf.constant([[3,5,7], 
                 [4,6,8]])
y = x[:,1]
with tf.Session() as sess:
  print y.eval() # [5,6]
```

We can reshape a tensor

```python
x  =tf.constant([[3,5,7],
                 [4,6,8]])
y = tf.reshape(x, [3,2])
with tf.Session() as sess:
  print y.eval() # prints [[3,5],[7,4],[6,8]]

y = tf.reshape(x,[3,2])[1,:]
with tf.Session() as sess:
  print y.eval() # [7,4]
```

A variable is a tensor whose value is initialized and then changed as the program runs.

```python
def forward_pass(w,x):
  return tf.matmul(w,x)

def train_loop(x, niter=5):
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE): # reuse the variable
    w = tf.get_variable("weights",  # create variable, specify how to init and whether it can be tuned
                        shape=(1,2),
                        initializer=tf.truncated_normal_initializer(),
                        trainable=True)
  preds = []
  for k in xrange(niter): # training loop of 5 updates to weights
    preds.append(forward_pass(w,x))
    w = w + 0.1 # gradient update
  return preds

with tf.Session() as sess:
  preds = train_loop(tf.constant([[3.2,5.1, 7.2], [4.3, 6.2, 8.3])) # 2 x 3 matrix
  tf.global_variables_initializer().run() # initialize all variables
  for i in xrange(len(preds)):
    print "{}:{}".format(i, preds[i].eval()) # prints (1,3) matrix at each 5 iterations
```

To feed different values, we can use placeholder which can read file and  use that value

```python
import tensorflow as tf

a = tf.placeholder("float", None)
b = x * 4
print a
with tf.Session() as session:
  print (session.run(b, feed_dic={a:[1,2,3]}))
```

[Tensorflow lazy and eager evaluation](a_tfstart.ipynb)

The shapes of various tensors can be changed using some of the methods. `squeeze` squeezes one dimension, `expand_dims` expands the dimension, `reshape` reshapes. We can use `tf.cast(t, tf.float32)` to convert the datatype of a tensor.
`tf.print()` is a function to print the value of a tensor when some condition is met. `tfdbg` is used to debug a tensorflow session. TensorBoard can also be used for various tensorflow debugging. We can change the logging level from WARN to INFO using `tf.logging.set_verbosity(tf.logging.INFO)`. Tensorflow also comes with dynamic debugger that can be used from command line `tf_debug`.

```python
import tensorflow as tf
from tensorflow.python import debug as tf_debug

def some_method(a,b):
  b = tf.cast(b, tf.float32)
  s = (a / b)
  return tf.sqrt(tf.matmul(s, tf.transpose(s)))

with tf.Session() a sess:
  fake_a = tf.constant([[5.0, 3.0, 7.1], [2.3, 5.1, 4.8]])
  fake_b = tf.constant([[2, 0, 5], [2,8,7]])
  sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
  print sess.run(some_method(fake_a, fake_b))
```

This can be run using `python xyz.py --debug`.

Check [debugger notebook](debug_demo.ipynb)