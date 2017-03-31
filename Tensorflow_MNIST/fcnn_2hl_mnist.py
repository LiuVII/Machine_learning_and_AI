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
TRAIN_STEPS = 2000
BATCH_SIZE = 100
LEARNING_RATE = 0.5
HIDDEN_UNITS = 128
HIDDEN_UNITS2 = 64
BETA_INIT = 0.0000

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

def weight_variable(inputs, outputs, name):
	# Random small values
	initial = tf.truncated_normal(shape=[inputs, outputs], stddev=1.0 / math.sqrt(float(inputs)))
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.constant(0.0, shape=[shape])
	return tf.Variable(initial, name=name)

def logits(data, weights, biases):
	return tf.matmul(data, weights) + biases

weights1 = weight_variable(NUM_PIXELS, HIDDEN_UNITS, 'weights1')
biases1 = bias_variable(HIDDEN_UNITS, 'biases1')
# weights3 = weight_variable(HIDDEN_UNITS, NUM_CLASSES, 'weights3')
# biases3 = bias_variable(NUM_CLASSES, 'biases3')
weights2 = weight_variable(HIDDEN_UNITS, HIDDEN_UNITS2, 'weights2')
biases2 = bias_variable(HIDDEN_UNITS2, 'biases2')
weights3 = weight_variable(HIDDEN_UNITS2, NUM_CLASSES, 'weights3')
biases3 = bias_variable(NUM_CLASSES, 'biases3')

hidden1 = tf.nn.relu(logits(x, weights1, biases1))
# h = logits(hidden1, weights3, biases3)
hidden2 = tf.nn.relu(logits(hidden1, weights2, biases2))
h = logits(hidden2, weights3, biases3)


# Write a summary of the graph (before we add the loss and optimizer)
# Which will add a bunch of nodes automatically
sw = tf.summary.FileWriter('summaries/single_hidden', graph=tf.get_default_graph())
sw.close()


# Loss function using L2 Regularization
# regularizer = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights3)
regularizer = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h) + beta * regularizer)
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

init = tf.initialize_all_variables()
sess.run(init)

loss_lst = []
step_lst = []
for i in range(TRAIN_STEPS):
	btach_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
	_, i_loss = sess.run([train_step, loss], feed_dict={x: btach_xs, y: batch_ys, beta: BETA_INIT})
	loss_lst.append(i_loss)
	step_lst.append(i)
	if i % 200 == 0 :
		print("loss %f" % i_loss)
with open("fc_loss.csv",'wb') as resultFile:
	wr = csv.writer(resultFile)
	wr.writerows([step_lst, loss_lst])
plt.plot(step_lst, loss_lst)
plt.show()
prediction = tf.equal(tf.argmax(y, 1), tf.argmax(h, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print("Accuracy %f" % sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels}))