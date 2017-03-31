from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import math
import csv

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

NUM_CLASSES = 10
NUM_PIXELS = 28 * 28
TRAIN_STEPS = 20000
BATCH_SIZE = 100
LEARNING_RATE = 0.1
BETA_INIT = 0.0000
PROB = 0.5

sess = None
def ResetSession():
	tf.reset_default_graph()
	global sess
	if sess is not None: sess.close()
	sess = tf.InteractiveSession()

ResetSession()
x = tf.placeholder(tf.float32,[None, NUM_PIXELS], name='pixels')
y = tf.placeholder(tf.float32,[None, NUM_CLASSES], name = 'labels')
beta = tf.placeholder(tf.float32, name='beta')

def weight_variable(shape, name):
	# Random small values
	initial = tf.truncated_normal(shape=shape, stddev=1.0)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def logits(data, weights, biases):
	return tf.matmul(data, weights) + biases

x_image = tf.reshape(x, [-1, 28, 28, 1])

# First conv layer + pool
W_conv1 = weight_variable([5,5,1,32], 'c_weights1')
b_conv1 = bias_variable([32], 'c_bias1')

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Seconde conv layer + pool
W_conv2 = weight_variable([5,5,32,64], 'c_weights2')
b_conv2 = bias_variable([64], 'c_bias2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# FC layer
W_fc1 = weight_variable([7 * 7 * 64, 1024], 'fc_weights1')
b_fc1 = bias_variable([1024], 'fc_bias1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(logits(h_pool2_flat, W_fc1, b_fc1))

# Dropout (randomly drop some connections to reduce overfitting)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Write a summary of the graph (before we add the loss and optimizer)
# Which will add a bunch of nodes automatically
# sw = tf.summary.FileWriter('summaries/single_hidden', graph=tf.get_default_graph())
# sw.close()

# Readout layer
W_fc2 = weight_variable([1024, 10], 'fc_weights2')
b_fc2 = weight_variable([10], 'fc_bias2')

h = logits(h_fc1_drop, W_fc2, b_fc2)
# h = logits(h_fc1_drop, W_fc2, b_fc2) 

# Loss function using L2 Regularization
# regularizer = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights3)
# regularizer = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3)

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h) + beta * regularizer)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
prediction = tf.equal(tf.argmax(y, 1), tf.argmax(h, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

init = tf.initialize_all_variables()
sess.run(init)

loss_lst = []
step_lst = []
for i in range(TRAIN_STEPS):
	btach_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
	_, i_loss = sess.run([train_step, loss], feed_dict={x: btach_xs, y: batch_ys, keep_prob: PROB})
	if i % 10:
		loss_lst.append(i_loss)
		step_lst.append(i)
	if i % 200 == 0 :
		print("Train accuracy %f" % sess.run(accuracy, feed_dict={x: btach_xs, y: batch_ys, keep_prob: 2*PROB}))
		# print("loss %f" % i_loss)
with open("fc_loss.csv",'wb') as resultFile:
	wr = csv.writer(resultFile)
	wr.writerows([step_lst, loss_lst])
plt.plot(step_lst, loss_lst)
plt.show()
print("Test accuracy %f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 2*PROB}))
