from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import math
import itertools
from collections import Counter
from sklearn.metrics import auc, roc_auc_score

import matplotlib.pyplot as plt

MAX_FEAT = 5000
NUM_CLASSES = 1
LEARNING_RATE = 2
TRAIN_STEPS = 100
BATCH_SIZE = 2000
TEMP_SIZE = 5000
NGRAM = 2

MAX_FEAT *= NGRAM

train_data = pd.read_csv("Clean_train.csv", names=['id', 'reviews', 'sentiment'], quoting=3)
test_data = pd.read_csv("Clean_test.csv", names=['id', 'reviews', 'sentiment'], quoting=3)

TRAIN_SIZE = len(train_data['sentiment'])
TEST_SIZE = len(test_data['sentiment'])

#print(test_data)
# vectorizer = CountVectorizer(analyzer = "word",  \
# 							tokenizer = None,    \
# 							preprocessor = None, \
# 							stop_words = None,   \
# 							ngram_range=(1, NGRAM),  \
# 							max_features = MAX_FEAT)

vectorizer = TfidfVectorizer(analyzer = "word",  \
							tokenizer = None,    \
							preprocessor = None, \
							stop_words = None,   \
							ngram_range=(1, NGRAM),  \
							max_features = MAX_FEAT)

#transformer = TfidfTransformer(smooth_idf=False)

x_train_raw = vectorizer.fit_transform(train_data['reviews'])
#x_train_raw = transformer.fit_transform(x_train_raw)
x_train = x_train_raw.toarray()
# train_data['asentiment'] = 1 - train_data['sentiment']
y_train = np.reshape(train_data['sentiment'].values, (TRAIN_SIZE, NUM_CLASSES))
# y_train = np.reshape(train_data[['sentiment', 'asentiment']].values, (TRAIN_SIZE, NUM_CLASSES))
vocab = vectorizer.get_feature_names()
x_test_raw = vectorizer.transform(test_data['reviews'])
#x_test_raw = transformer.fit_transform(x_test_raw)
x_test = x_test_raw.toarray()
# test_data['asentiment'] = 1 - test_data['sentiment']
y_test = np.reshape(test_data['sentiment'].values, (TEST_SIZE, NUM_CLASSES))
# y_test = np.reshape(test_data[['sentiment', 'asentiment']].values, (TEST_SIZE, NUM_CLASSES))

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
y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='y_label')

def weight_variable(inputs, outputs, name):
	# Random small values
	initial = tf.truncated_normal(shape=[inputs, outputs], stddev=1.0 / math.sqrt(float(inputs)))
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.constant(0.0, shape=[shape])
	return tf.Variable(initial, name=name)

def batch_iter(data, batch_size, num_epochs, shuffle=True):

    # Generates a batch iterator for a dataset.
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

W = weight_variable(MAX_FEAT, NUM_CLASSES, 'weights')
b = bias_variable(NUM_CLASSES, name='bias')

h = tf.matmul(x, W) + b
h_sig = tf.sigmoid(h)
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=y))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=y))
# train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
train_step = tf.train.AdamOptimizer(4e-3).minimize(loss)

prediction = tf.equal((h_sig > 0.5), tf.cast(y, tf.bool))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

init_g = tf.global_variables_initializer()
sess.run(init_g)

tr_loss_lst = []
ts_loss_lst = []
#i_lst = []
i = 0
avg_loss = 0
batches = batch_iter(zip(x_train, y_train), BATCH_SIZE, TRAIN_STEPS)
for batch in batches:
	x_batch, y_batch = zip(*batch)
	#i_lst.append(i)
	i_loss, _ = sess.run([loss, train_step], feed_dict={x: x_batch, y: y_batch})
	
	avg_loss += i_loss
	if i % 10 == 0:
		print(int((i * 10) / TRAIN_STEPS), i_loss)
		tr_loss_lst.append(avg_loss/10)
		avg_loss = 0
		ts_loss_lst.append(sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
	if i % 100 == 0:
		print("Train accuracy %f" % sess.run(accuracy, feed_dict={x: x_train, y: y_train}))
		# sess.run(init_l)
		exh, exy = sess.run([h_sig, y], feed_dict={x: x_batch, y: y_batch})
		print("Train AUC %f" % roc_auc_score(exy[:,0],exh[:,0]))
	i += 1
print("Train accuracy %f" % sess.run(accuracy, feed_dict={x: x_train, y: y_train}))
exh, exy = sess.run([h_sig, y], feed_dict={x: x_train, y: y_train})
print("Train AUC %f" % roc_auc_score(exy[:,0],exh[:,0]))
print("Test accuracy %f" % sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
exh, exy = sess.run([h_sig, y], feed_dict={x: x_test, y: y_test})
print("Test AUC %f" % roc_auc_score(exy[:,0],exh[:,0]))

plt.plot(tr_loss_lst, 'b')
plt.plot(ts_loss_lst, 'g')
plt.show()

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
# output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
# output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )