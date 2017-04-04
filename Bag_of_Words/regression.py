from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math

import matplotlib.pyplot as plt

MAX_FEAT = 5000
NUM_CLASSES = 1
LEARNING_RATE = 0.1
TRAIN_STEPS = 10
TRAIN_SIZE = 20000
TEMP_SIZE = 5000

train_data = pd.read_csv("Clean_train.csv", names=['id', 'reviews', 'sentiment'], quoting=3)
test_data = pd.read_csv("Clean_test.csv", names=['id', 'reviews', 'sentiment'], quoting=3)

#print(test_data)
vectorizer = CountVectorizer(analyzer = "word",  \
							tokenizer = None,    \
							preprocessor = None, \
							stop_words = None,   \
							max_features = MAX_FEAT)

x_train = vectorizer.fit_transform(train_data['reviews']).toarray()
y_train = np.reshape(train_data['sentiment'].values, (len(train_data['sentiment']), 1))
# print(y_train)
vocab = vectorizer.get_feature_names()
x_test = vectorizer.transform(test_data['reviews']).toarray()
y_test = np.reshape(test_data['sentiment'].values, (len(test_data['sentiment']), 1))
# print(y_test)
# test_data_feats = vectorizer.fit_transform(test_data['reviews']).toarray()
#print train_data_features.shape

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

sess = None
def ResetSession():
	tf.reset_default_graph()
	global sess
	if sess is not None: sess.close()
	sess = tf.InteractiveSession()

ResetSession()
x = tf.placeholder(tf.float32, [None, MAX_FEAT], name='x')
y = tf.placeholder(tf.float32, [None, 1], name='y_label')

def weight_variable(inputs, outputs, name):
	# Random small values
	initial = tf.truncated_normal(shape=[inputs, outputs], stddev=1.0 / math.sqrt(float(inputs)))
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.constant(0.0, shape=[shape])
	return tf.Variable(initial, name=name)

W = weight_variable(MAX_FEAT, 1, 'weights')
b = bias_variable(1, name='bias')
# a = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9], shape=[6, 3])
# b = tf.constant([7, 8, 9,7, 8, 9], shape=[3, 2])
# print(tf.matmul(a, b))
h = tf.matmul(x, W) + b
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

prediction = tf.equal(y, h)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)

loss_lst = []
i_lst = []
for i in range(TRAIN_STEPS):
	i_lst.append(i)
	i_loss, _ = sess.run([loss, train_step], feed_dict={x: x_train, y: y_train})
	loss_lst.append(i_loss)
	if i % 10 == 0:
		print (round(i / TRAIN_STEPS * 100), i_loss)
print("Train accuracy %f" % sess.run(accuracy, feed_dict={x: x_train, y: y_train}))
print("Test accuracy %f" % sess.run(accuracy, feed_dict={x: x_test, y: y_test}))

# plt.plot(i_lst, loss_lst)
# plt.show()

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
# output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
# output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )