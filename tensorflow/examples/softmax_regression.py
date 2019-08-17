import tensorflow as tf
# import input data
from tensorflow.examples.tutorials.mnist import input_data

# Define program constants
DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100
# The `read_data_sets()` from MNIST downloads the dataset and saves it locally
# DATA_DIR is the location where we wish the data to be saved
# second argument tells how the data to be labeled
data = input_data.read_data_sets(DATA_DIR, one_shot=True)

# A variable is an element manipulated by the computation
# A placeholder has to be supplied when triggering it. Here, image is a placeholder because it will be supplied when running computation graph
# The image is a 784 (28 x 28) pixels unrolled into a single vector. None means we are not sure how many of images we will use at once.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

# y_true is the true label
# y_pred is the predicted label
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

# The measure of similarity for this model is known as cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  logits=y_pred, labels=y_true
))

# This is called loss function. We try to minize this using gradient descent optimization.
# 0.5 is the learning rate
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# the procedure to evaluate the accuracy of the model
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# create session to run training and test
with tf.Session() as sess:
  # Train
  # initialize all variables
  sess.run(tf.global_variables_initializer())
  # NUM_STEPS is the number of steps we will make for gradient descent
  for _ in range(NUM_STEPS):
    batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
    # feed_dict argument  will feed the data from current batch
    sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    # calculate accuracy using test datasets
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})
# print the results in percentage values
print "Accuracy: {:.4}%".format(ans*100)