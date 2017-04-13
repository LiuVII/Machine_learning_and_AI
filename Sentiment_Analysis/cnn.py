# Code based on source: https://github.com/dennybritz/cnn-text-classification-tf
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import math
import time
import os
import datetime

from sklearn.metrics import auc, roc_auc_score

import matplotlib.pyplot as plt

NUM_CLASSES = 2
TEST_SIZE = 0.1

# Hyper Params
EMBEDDING_SIZE = 128
DROPOUT_INIT = 0.5
FILTER_SIZES = [3, 4, 5]
## Numbero of filters per filter_size
NUM_FILTERS = 128
CHANNELS_NUM = 1
## Without padding
## ? why
PADDING = "VALID"
STRIDES = [1,1,1,1]

# Training Params
ADAM_PARAM = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 128
EVALUATE = 100  
CHECKPOINT = 100
BETA_INIT = 1e-5
BETA_STEPS = 1

# Misc
allow_soft_placement = True
log_device_placement = False

# Load data
# x_text, y =

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(TEST_SIZE * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

sequence_length = x_train.shape[1]
vocab_size = len(vocabulary)

# Model vars
x = tf.placeholder(tf.float32, [None, sequence_length], name='x')
y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='y_label')
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# Creating top-level node for TensorBoard
with tf.name_scope("embedding"):
	Wem = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), name="W")
	# Results in Tensor [None, sequence_length, EMBEDDING_SIZE]
	embedded_chars = tf.nn.embedding_lookup(Wem, x)
	# Insert dimension of one to the end of tensor
	embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

pooled_outputs = []
for i, filter_size in enumerate(FILTER_SIZES):
	# Another top-level node for each filter size:
	with tf.name_scope("conv_maxpool-%s" % filter_size):
		# Convlolution layer with 1 channel
		filter_shape = [filter_size, EMBEDDING_SIZE, CHANNELS_NUM, NUM_FILTERS]
		W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
		b = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]), name='b')
		conv = tf.nn.conv2d(
			embedded_chars_expanded,
			W,
			strides=STRIDES,
			padding=PADDING,
			name="conv")
		# Apply nonlinearity
		h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
		# Max-pooling over the outputs
		# kszie - output shape as without padding
		pooled = tf.nn.max_pool(
			h,
			kszie=[1, sequence_length - filter_size + 1, 1, 1],
			strides=STRIDES,
			padding=PADDING,
			name="pool")
		pooled_outputs.append(pooled)

# Combine all pooled features
# ? not sure
num_filters_total = NUM_FILTERS * len(FILTER_SIZES)
h_pool = tf.concat(3, pooled_outputs)
# Flatten dimension when possible
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# Add dropout
# 1 - during evaluation (to disable), ~0.5 during training to prevent extensive co-adapting
with tf.name_scope("dropout"):
	h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

with tf.name_scope("output"):
	Wout = tf.Variable(tf.truncated_normal([num_filters_total, NUM_CLASSES], stddev=0.1),
		name="Wout")
	bout = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b")
	# Same as matmul(x, W) + b in 2D
	scores = tf.nn.xw_plus_b(h_drop, Wout, bout, name="scores")
	predictions = tf.argmax(scores, 1, name="predictions")

# Calc loss
with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, y))

# Training
train_step = tf.train.AdamOptimizer(ADAM_PARAM).minimize(loss)

# Calc accuracy:
with tf.name_scope("accuracy"):
	correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

# ? Calc AUC
# with tf.name_scope("AUC_ROC"):
with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=allow_soft_placement,
		log_device_placement=log_device_placement)
    # sess = tf.Session(config=session_conf)
    sess = tf.InteractiveSession(config=session_conf)
    # with sess.as_default():

# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
 
# Summaries for loss and accuracy
loss_summary = tf.scalar_summary("loss", loss)
acc_summary = tf.scalar_summary("accuracy", accuracy)
 
# Train Summaries
train_summary_op = tf.merge_summary([loss_summary, acc_summary])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)
 
# Dev summaries
dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

# Checkpointing
# checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
# checkpoint_prefix = os.path.join(checkpoint_dir, "model")
# # Tensorflow assumes this directory already exists so we need to create it
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
# saver = tf.train.Saver(tf.global_variables())

# Initialize all variables
sess.run(tf.global_variables_initializer())

def step(x_batch, y_batch, writer=None, train=True)

	if train:
		dropout = DROPOUT_INIT = 0.5
	else:
		dropout = 1

	feed_dict = { x: x_batch, y: y_batch, dropout_keep_prob: dropout }

	if train:
		sess.run([train_step], feed_dict=feed_dict)

	summaries, loss, accuracy = sess.run(
		[train_summary_op, loss, accuracy], feed_dict=feed_dict)

	time_str = datetime.datetime.now().isoformat()
	print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
	
	if train:
		train_summary_writer.add_summary(summaries, step)
	elif writer:
	        writer.add_summary(summaries, step)

# Generate batches
batches = data_helpers.batch_iter(
    zip(x_train, y_train), BATCH_SIZE, NUM_EPOCHS)
# Training loop. For each batch...
step_num = 0
for batch in batches:
	step_num += 1
	x_batch, y_batch = zip(*batch)
	step(x_batch, y_batch)
	if step_num % EVALUATE == 0:
		print("\nEvaluation:")
		step(x_dev, y_dev, writer=dev_summary_writer, train=False)
		print("")
	# if step_num % CHECKPOINT == 0:
	# 	path = saver.save(sess, checkpoint_prefix, global_step=current_step)
	# 	print("Saved model checkpoint to {}\n".format(path))
