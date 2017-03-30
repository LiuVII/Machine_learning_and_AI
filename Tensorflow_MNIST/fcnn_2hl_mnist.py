from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import math

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

NUM_CLASSES = 10
NUM_PIXELS = 28 * 28
TRAIN_STEPS = 2000
BATCH_SIZE = 100
LEARNING_RATE = 0.5
HIDDEN_UNITS = 128
HIDDEN_UNITS2 = 64
# BETA_INIT = 0.00015

sess = None
def ResetSession():
	tf.reset_default_graph()
	global sess
	if sess is not None: sess.close()
	sess = tf.InteractiveSession()

ResetSession()
x = tf.placeholder(tf.float32,[None, NUM_PIXELS], name='pixels')
y = tf.placeholder(tf.float32,[None, NUM_CLASSES], name = 'labels')

def weight_variable(inputs, outputs, name):
	# Normalized
	initial = tf.truncated_normal(shape=[inputs, outputs], stddev=1.0 / math.sqrt(float(inputs)))
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.constant(0.0, shape=[shape])
	return tf.Variable(initial, name=name)

def logits(data, weights, biases):
	return tf.matmul(data, weights) + biases

weights1 = weight_variable(NUM_PIXELS, HIDDEN_UNITS, 'weights1')
biases1 = bias_variable(HIDDEN_UNITS, 'biases1')
# weights2 = weight_variable(HIDDEN_UNITS, HIDDEN_UNITS2, 'weights2')
# biases2 = bias_variable(HIDDEN_UNITS2, 'biases2')
# weights3 = weight_variable(HIDDEN_UNITS2, NUM_CLASSES, 'weights3')
# biases3 = bias_variable(NUM_CLASSES, 'biases3')

weights3 = weight_variable(HIDDEN_UNITS, NUM_CLASSES, 'weights3')
biases3 = bias_variable(NUM_CLASSES, 'biases3')

hidden1 = tf.nn.relu(logits(x, weights1, biases1))
# hidden2 = tf.nn.relu(logits(hidden1, weights2, biases2))
# h = logits(hidden2, weights3, biases3)
h = logits(hidden1, weights3, biases3)

# Write a summary of the graph (before we add the loss and optimizer)
# Which will add a bunch of nodes automatically
sw = tf.summary.FileWriter('summaries/single_hidden', graph=tf.get_default_graph())
sw.close()

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

init = tf.initialize_all_variables()
sess.run(init)

for i in range(TRAIN_STEPS):
	btach_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
	_, i_loss = sess.run([train_step, loss], feed_dict={x: btach_xs, y: batch_ys})
	if i % 200 == 0 :
		print("loss %f" % i_loss)

prediction = tf.equal(tf.argmax(y, 1), tf.argmax(h, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print("Accuracy %f" % sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels}))