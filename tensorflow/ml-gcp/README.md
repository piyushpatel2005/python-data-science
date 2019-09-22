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

[Reading CSV files](notebooks/earthquakes.ipynb)

Check [flights notebook](notebooks/flights.ipynb) to see invocation of BigQuery and getting results in Pandas dataframes.

[Invoking already trained machine learning API](notebooks/mlapis.ipynb) on GCP is easy.

Creating repeatable dataset is an important skill for ML engineers. You should divide the dataset into three parts, development and training, validation and test dataset. This datasets should be independent of bias and if we split the dataset based on one of the features, that feature will have lose its predictability. Before dividing the data into such training and testing datasets, it is important that the data is cleaned.

Check [Datasets creation](notebooks/repeatable_splitting.ipynb)

Also, benchmarking is an important consideration to stop overfitting of the model. Overfitted model may not generalize beyond training datasets.

[Benchmarking](notebooks/create_datasets.ipynb)

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

[Tensorflow lazy and eager evaluation](notebooks/a_tfstart.ipynb)

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

Check [debugger notebook](notebooks/debug_demo.ipynb)

## Estimator API

Estimator API provides high level API for creating machine learning applications. They are interchangeable and allows for quick modelling. It also provides checkpointing. It allows for out-of-memory datasets. It will train, evaluate and monitor. It will allows for distributed training. Everything is managed by estimator API. The base class `tf.estimator.Estimator` can wrap our code. Tensorflow has ready estimators like LinearRegressor, DNNRegressor, LinearClassifier, DNNClassifier, etc. which we can try out.

To predict the property value based on features (sq_footage, type), we can do something like this. Categorical values will be one-hot encoded.

There are many different feature columns to choose from like bucketized_column, embedding_column, crossed_column, categorical_column_with_hash_bucket, etc. To train the model, we write input function that will return features as named in the `featcols`.

```python
import tensorflow as tf
featcols=[
  tf.feature_column_numeric_column("sq_footage"),
  tf.feature_column.categorial_column_with_vocabulary_list("type", ["house", "apt"])
]
model = tf.estimator.LinearRegressor(featcols) # instantiate LinearRegressor

def train_input_fn():
  features = {"sq_footage": [1000, 2000, 3000, 4000, 1000, 2000],
              "type":       ["house", "house", "apt", "house", "apt", "house"]}
  labels = [500, 1000, 1500, 2000, 300, 900]
  return features, labels

# call train function
model.train(train_input_fn, steps=100) # repeat 100 times

# use the model for prediction
def predict_input_fn():
  features = {"sq_footage": [1500, 1800],
              "type":       ["house", "apt"]}
  return features
predictions = model.predict(predict_input_fn)
```

To use different pre-made estimator, just change the class name and supply appropriate parameters. 

When we train large models, checkpointing is important as it will allow us to start training from where we left off.

```python
model = tf.estimator.LinearRegressor(featcols, './model_trained') # specify checkpointing directory while instantiating estimator class
# To restore from the checkpointing directory, we instantiate using that directory
trained_model = tf.estimator.LinearRegressor(featcols, './model_trained')
trained_model.train(train_input_fn, steps=100) # continue training from last checkpoint if you think few more steps are required
predictions = trained_model.predict(pred_input_fn) # predict from checkpointed training
# To restart from scratch, delete this folder
```

For feeding numpy array or pandas dataframe, estimator has easy api

```python
def numpy_train_input_fn(sqft, prop_type, price):
  return tf.estimator.inputs.numpy_input_fn(
    x = {"sq_footage": sqft, "type": prop_type}, # feature dictionary using x
    y = price, # labels using y named parameter
    batch_size = 128, # how many datasets to use 
    num_epochs = 10, # how many times to repeat the dataset
    shuffle = True, # shuffling training data is important
    queue_capacity = 1000 # size of the shuffle queue
  )
def pandas_train_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df, # sq_footage and type selected automatically because of feature columns definition
    y = df['price']
    batch_size = 128,
    num_epochs = 10,
    shuffle = True,
    queue_capacity = 1000
  )

model.train(pandas_train_input_fn(df), steps=1000) # override the steps in the defined function
# one step corresponds to one batch of data
model.train(pandas_train_input_fn(df), max_steps=1000) # might be nothing if checkpoint already there
```

[Example of running training using estimator API](notebooks/b_estimator.ipynb)

For real world model, we can supply data using datasets. Large datasets might be sharded into different files.

```python
# load data from CSV files
def decode_line(row):
  cols = tf.decode_csv(row, record_defaults=[[0], ['house'], [0]])
  features = {'sq_footage': cols[0], 'type': cols[1]}
  label = cols[2] # price
  return features, label

dataset = tf.data.TextLineDataset("train_1.csv").map(decode_line)

dataset = dataset.shuffle(1000)
                  .repeat(15)  # no. of epoch
                  .batch(128)

def input_fn():
  features, label = dataset.make_one_shot_iterator().get_next()
  return features, label # input function actually returns Tensorflow node and not the actual data.
model.train(input_fn)
```

If we want to load large dataset with multiple sharded files, we have to change the following

```python
dataset = tf.data.Dataset.list_files("train.csv")
            .flat_map(tf.data.TextLineDataset)
            .map(decode_line)
```

[Handling large data](notebooks/c_dataset.ipynb)

The function `tf.estimator.train_and_evaluate()` implements distributed training. For using this api, we choose estimator, provide run config and provide train and evaluation specs. EvalSpec is where we provide our test dataset.

To visualize the training use `tensorboard --logdir output_dir` when the model is created using `tf.estimator.RunConfig(model_dir="output_dir")`.

Once we deploy our model in production, it can handle json input and produces json output. Gcloud command `gcloud ml-engine predict --model <model_name> --json-instances data.json` allows to receive output from our model. `gcloud ml-engine local predict --model-dir output/export/pricing/12325353 --json-instances data.json` lets us get output from exported model from local disk. When working with images, we send images as compressed but the model expects images to be uncompressed. Here is the function that handles that.

```python
def serving_input_fn():
  json = { 'jpeg_bytes': tf.placeholder(tf.string, [None])}

  def decode(jpeg):
    pixels = tf.image.decode_jpeg(jpeg, channels=3)
    return pixels
  pics = tf.map_fn(decode, json['jpeg_bytes'], dtype=tf.uint8)
  features = {'pics': pics}
  return tf.estimator.export.ServingInputReceiver(features, json)
```

[Distributed training and using Tensorboard](notebooks/d_traineval.ipynb)

Google cloud AI platform allows to scale machine learning beyond single machine's capabilities. Large datasets can be handled on cloud. We can have micro service to serve our model with scalable cloud platform.

When running machine learning job on Google cloud, we separate our job into `task.py` to parse command-line parameters and send to `train_and_evaluate`. This file will invoke `model.py`. For ML training, single region bucket gives much better performance.

Once submitted the job, we can get details on the job using `gcloud ml-engine jobs describe job_name`. To get the latest logs `gcloud ml-engine jobs stream-jobs job_name`.

Filter jobs based on creation time or name

```shell
gcloud ml-engine jobs list --filter='createTime>2017-01-15T19:00'
gcloud ml-engine jobs list --filter='jobId:census*' --limit=3
```

[Running Tensorflow on cloud](notebooks/e_ai_platform.ipynb)

## Feature Engineering

Feature engineering takes about 50-70% of the time. Good features have to be related to the objective. It should be known at prediction time, should be numeric and have enough examples. If we have too many features, it will result in data-draggery. Different problems in the same domain may require different features. We should at least have 5 examples of each data (categorical data) in order to consider it as good feature.

[First notebook feature engineering](notebooks/a_features.ipynb)

Sometimes, during preprocessing, we may need to re-scale values. To re-scale, we need minimum and maximum values of dataset.

```python
features['scaled_price'] = (features['price'] - min_price) / (max_price - min_price)
```

To use one-hot encoding, we must have specific set of values for categorical values.

**Apache Beam** is data processing system. It creates a data processing pipeline. The pipeline must have a source, series of steps (transform) which works on `PCollection` which is the data processing unit. The output goes to a sink. To run a pipeline, we need runner. Runners are platform specific. In Python, `|` operator is overloaded to mean `apply` method which specifies operation on the data. To run the pipeline, we need to call `run()` method. 

If you want to parallelize transformation and scale it across many nodes in the cluster, we should use Apache Beam's `ParDo` class. It acts on one item at a time (like Map in MapReduce). There are few methods in Beam that makes use of this class. For example `Map`, `FlatMap`, etc.

```python
# Simple apache beam pipeline to check the import statement in java files
import apache_beam as beam
import sys

def my_grep(line, term):
   if line.startswith(term):
      yield line

if __name__ == '__main__':
   p = beam.Pipeline(argv=sys.argv)
   input = '../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/*.java'
   output_prefix = '/tmp/output'
   searchTerm = 'import'

   # find all lines that contain the searchTerm
   (p
      | 'GetJava' >> beam.io.ReadFromText(input)
      | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm) )
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run().wait_until_finish()
   # To execute, `python grep.py`
```

```shell
gsutil cp ../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/*.java gs://<YOUR-BUCKET-NAME>/javahelp
```

```python
# Run on Cloud DataFlow

import apache_beam as beam

def my_grep(line, term):
   if line.startswith(term):
      yield line

PROJECT='cloud-training-demos'
BUCKET='cloud-training-demos'

def run():
   argv = [
      '--project={0}'.format(PROJECT),
      '--job_name=examplejob2',
      '--save_main_session',
      '--staging_location=gs://{0}/staging/'.format(BUCKET),
      '--temp_location=gs://{0}/staging/'.format(BUCKET),
      '--runner=DataflowRunner'
   ]

   p = beam.Pipeline(argv=argv)
   input = 'gs://{0}/javahelp/*.java'.format(BUCKET)
   output_prefix = 'gs://{0}/javahelp/output'.format(BUCKET)
   searchTerm = 'import'

   # find all lines that contain the searchTerm
   (p
      | 'GetJava' >> beam.io.ReadFromText(input)
      | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm) )
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run()

if __name__ == '__main__':
   run()
# Execute using `python grepc.py`
```

This executes Cloud Dataflow job, which you can also monitor in the Dataflow services.

```shell
gsutil cat gs://<YOUR-BUCKET-NAME>/javahelp/output-*
```

MapReduce job using Apache Beam

```python
import apache_beam as beam
import argparse

def startsWith(line, term):
   if line.startswith(term):
      yield line

def splitPackageName(packageName):
   """e.g. given com.example.appname.library.widgetname
           returns com
	           com.example
                   com.example.appname
      etc.
   """
   result = []
   end = packageName.find('.')
   while end > 0:
      result.append(packageName[0:end])
      end = packageName.find('.', end+1)
   result.append(packageName)
   return result

def getPackages(line, keyword):
   start = line.find(keyword) + len(keyword)
   end = line.find(';', start)
   if start < end:
      packageName = line[start:end].strip()
      return splitPackageName(packageName)
   return []

def packageUse(line, keyword):
   packages = getPackages(line, keyword)
   for p in packages:
      yield (p, 1)

def by_value(kv1, kv2):
   key1, value1 = kv1
   key2, value2 = kv2
   return value1 < value2

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Find the most used Java packages')
   parser.add_argument('--output_prefix', default='/tmp/output', help='Output prefix')
   parser.add_argument('--input', default='../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/', help='Input directory')

   options, pipeline_args = parser.parse_known_args()
   p = beam.Pipeline(argv=pipeline_args)

   input = '{0}*.java'.format(options.input)
   output_prefix = options.output_prefix
   keyword = 'import'

   # find most used packages
   (p
      | 'GetJava' >> beam.io.ReadFromText(input)
      | 'GetImports' >> beam.FlatMap(lambda line: startsWith(line, keyword))
      | 'PackageUse' >> beam.FlatMap(lambda line: packageUse(line, keyword))
      | 'TotalUse' >> beam.CombinePerKey(sum)
      | 'Top_5' >> beam.transforms.combiners.Top.Of(5, by_value)
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run().wait_until_finish()
```

```shell
python ./is_popular.py --output_prefix=/tmp/myoutput
ls -lrt /tmp/myoutput*
```

Cloud **DataPrep** allows to visualize and preprocess our data for feature engineering. It is fully managed service that allows for data processing using visual graphical UI. We can also create data visualizations to get statistical summaries. Dataprep has wranglers which can be used in Dataflow to convert them to pipeline.


### Feature Cross

Some problem even though they don't appear as linear model, we can have feature cross which makes the model linear with new feature. It lets you combine features.

For any model, we can discretize the data which will create various number of squares. Memorization works when you have lots of data.

To create a feature-cross use the method `crossed_column`.

```python
day_hr = tf.feature_column.crossed_column(
  [dayofweek, hourofday],
  24 * 7 # total number of hash buckets
) # These columns have to be categorical otherwise, bucketize the column and then use crossed_column
```

The number of hash buckets manages sparsity and collision. Using a large value of hash buckets result in spark representation of feature cross. To use the embedding, we can use one of the method from tensorflow. Instead of one-hot encoding, we pass it through dense layer. This dense layer creates **embeddings**. To create embeddings, use `embedding_column`.

```python
day_hr_em = tf.feature_column.embedding_column(
  day_hr,
  2,
  ckpt_to_load_from='london/ckpt-1000*',
  tensor_name_in_ckpt='dayhr_embed',
  trainable=False
)
```

There are three possible places where we can do feature engineering.

```python
# inside train_input_fn function
featcols = [
  fc.numeric_column('sq_footage'),
  fc.categorical_column_with_vocabulary_list(
    "type", ["house", "apt"]
  )
]
featcols[0] = (
  fc.bucketized_column(features[0],
          [500, 1000, 2500])
)
model = tf.estimator.LinearRegressor(featcols)

# creating new fature columns with embeddings
latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()

b_lat = fc.bucketized_column(house_lat, latbuckets)

loc = fc.crossed_column([b_lat, b_lon], nbuckets * nbuckets)

eloc = fc.embedding_column(loc, nbuckets//4)
```

Features can be created in Tensorflow.

```python
def add_engineered(features):
  lat1 = features['lat']
  .. ... ...
  dist = tf.sqrt(latdiff*latdiff + longdiff * longdiff)
  features['euclidean'] = dist
  return features
```

Call the `add_engineered` method from all input function.

```python
def train_input_fn():
  ... ...
  return add_engineered(features), label

def serving_input_fn():
  .. ....
  return ServingInputReceiver(
    add_engineered(features), json_features_ph)
```

The other option is to create feature engineering in Cloud Dataflow. The third option is to use `tf.transform`.

[Feature Engineering on Dataflow](notebooks/feateng.ipynb)

With **Tensorflow transform**, we are limited to tensorflow transform but also get efficiency of Tensorflow. It is hybrid of Apache Beam and Tensorflow. For on the fly preprocessing, use Tensorflow. In this case, analysis is carried out in Apache Beam and transformations are done in Tensorflow. `tf.transform` provides two `PTransform`s. `AnalyzeAndTransformDataset` is executed in Beam to create the training dataset. `TransformDataset` is executed in Beam to create the evaluation dataset. The transformation code is executed in Tensorflow at prediction time. There are two phases: Analysis phase executed in Beam, Transform phase executed in Tensorflow during prediction.

First, we inform Beam, which kind of data is expected using schema of training dataset.

```python
raw_data_schema = {
  colname: dataset_schema.ColumnSchema(tf.String, ...)
  for colname in 'dayOfweek,key',split(",")
}

raw_data_schema.update({
  colname: dataset_schema.ColumnSchema(tf.float32, ...)
  for colname in 'fare_amount,pickuplon,...,dropofflat'.split(',')
})
# Use schema to create metadata template
raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema(raw_data_schema))

# Run analyze-and-transform PTransform on training dataset to get back preprocessed training data and transform function
raw_data = (p
  # read data from BigQuery
  | beam.io.Read(beam.io.BigQuerySource(query=myquery, use_standard_sql=True))
  # Filter the data that is valid
  | beam.Filter(is_valid))
  # Transformations are saved in transform_fn
transformed_dataset, transform_fn = ((raw_data, raw_data_metadata)
# pass raw data and metadata template to AnalyzeAndTransformDataset
| beam_impl.AnalyzeAndTransformDataset(preprocess))

# Write out the preprocessed training data into TFRecords, the most efficient format for Tensorflow
transformed_data |
  tfrecordio.WriteToTFRecord(
    os.path.join(OUTPUT_DIR, 'train'),
    coder=ExampleProtoCoder(transformed_metadata.schema)
  )

# Preproecssing function is restricted to Tensorflow functions
# The things in preprocess() will get added to the Tensorflow graph and be executed in Tensorflow during serving.
# Input to preprocess is a dictionary of tensors
def preprocess(inputs):
  results = {} # Create features from the input tensors and put into "results" dict.
  result['fare_amount'] = inputs['fare_amount']
  result['dayofweek'] = tft.string_to_int(inputs['dayofweek']) # convert Sunday to number 1
  ... ...
  result['dropofflat'] = (tft.scale_to_0_1(inputs['dropofflat']))
  return result
```

While writing out the evaluation dataset, we reuse the transform function computed from the training. The reason `preprocess` needs only Tensorflow methods and not any Python method is because they are part of prediction graph. This way user can give raw data and model can do necessary transformations.

[TF Transform example](notebooks/tftransform.ipynb)

## Regularization

When the loss on test data is increasing compared to training data. That graph is a sign of overfitting with large number of iterations.

Learning rate controls the size of the step in weight space. If too small, training will take a long time. If too large, training will bounce around. Default learning rate in Estimator's LinearRegressor is smaller of 0.2 or 1/sqrt(num_features). Similarly, batch size controls the number of samples that gradient is calculated on. If too small, training will bounce around. If too large, training will take a long time. Usually batch size of 40-100 is a good number.
Regularization provides a way to define model complexity based on the values of the weights.

Optimizing is a technique of minimizing or maximizing function. There are many functions for optimization.
- Gradient Descent --> Traditional approach, typically implemented.
- Momentum --> Reduces learning rate when gradient values are small
- AdaGrad --> Gives frequently occurring features low learning rates.
- AdaDelta --> Improves AdaGrad by avoiding reducing LR (learning rate) to zero.
- Adam --> AdaGrad with a bunch of fixes
- Ftrl --> "Follow the regularized leader" works well on wide models

The last two are good defaults for DNN and Linear models.

[Hand Tuning of Hyperparameters](notebooks/a_handtuning.ipynb)

### Hyperparameter Tuning

Hyperparameter is a parameter set before training which doesn't change during training. Google Vizier allows us to automatically tune hyperparameters. Cloud ML takes this burden of hyperparameter tuning away. For such auto tuning, we need to do following three things.
1. Make hyperparameters as command line arguments
2. Make sure that outputs of different steps don't clobber each other by probably adding suffix.
3. Supply hyperparameters to training job.

[Hyperparameter tuning automatic](notebooks/b_hyperparam.ipynb)