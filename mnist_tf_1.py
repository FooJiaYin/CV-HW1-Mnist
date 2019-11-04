from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import Mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Number of samples in dataset
#n_train = mnist.train.num_examples  # 55,000
#n_validation = mnist.validation.num_examples  # 5000
#n_test = mnist.test.num_examples  # 10,000

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.random_uniform([784,10], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([10], -1.0, 1.0))
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])

epoch = 10000
loss_list = np.empty([epoch])
train_accuracy = np.empty([int(epoch/100)])
step_id = np.empty([int(epoch/100)])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(50)
    if i%100 == 0:
        step_id[int(i/100)] = i;
        train_accuracy[int(i/100)] = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("step %d, training accuracy %g"%(i, train_accuracy[int(i/100)]))
    loss = sess.run((train_step, cross_entropy), feed_dict={x: batch_x, y_: batch_y})
    #print(loss[1])
    loss_list[i] = loss[1]

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
plt.plot(loss_list)
plt.show()
plt.ylim(0, 1)
plt.xlim(0, epoch)
plt.plot(step_id, train_accuracy)
plt.show()