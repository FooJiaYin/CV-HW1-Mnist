from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## Import Mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Number of samples in dataset
#n_train = mnist.train.num_examples  # 55,000
#n_validation = mnist.validation.num_examples  # 5000
#n_test = mnist.test.num_examples  # 10,000

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None,10])

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

## Initialize weight
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

## Input layer, layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## Layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## Fully connected
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## Softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

epoch = 10000
loss_list = []
train_accuracy = []
epoch_count = []

## Train model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(epoch):
    batch_x, batch_y = mnist.train.next_batch(50)
    if i%100 == 0:
        epoch_count.append(i);
        train_accuracy.append(sess.run(accuracy, feed_dict={
            x:batch_x, y_: batch_y, keep_prob: 1.0}))
        print("step %d, training accuracy %g"%(i, train_accuracy[int(i/100)]))
    loss = sess.run((train_step, cross_entropy), feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    loss_list.append(loss[1])

print("test accuracy", sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
plt.plot(loss_list)
plt.show()
plt.ylim(0, 1)
plt.xlim(0,epoch)
plt.plot(epoch_count, train_accuracy)
plt.show()