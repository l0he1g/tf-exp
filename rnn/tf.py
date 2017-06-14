# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

batch_size = 200
num_classes = 2
state_size = 16
num_steps = 10
learning_rate = 0.1


def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i - 3] == 1:
            threshold += 0.5
        if X[i - 8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


# graph definition
cell = rnn.BasicRNNCell(state_size)
xs = tf.placeholder(tf.int32, [None, num_steps], name="xs")
ys = tf.placeholder(tf.int32, [None, num_steps], name="ys")
init_state = tf.placeholder(tf.float32, [None, state_size], name="init_state")
inputs = tf.one_hot(xs, num_classes)

outputs, states = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)
with tf.variable_scope("softmax") as scope:
    W = tf.get_variable("W", [state_size, num_classes])
    b = tf.get_variable("b", [num_classes], initializer=tf.constant_initializer(0.0))

logits = tf.reshape(tf.matmul(tf.reshape(outputs, [-1, state_size]), W) + b,
    [batch_size, num_steps, num_classes])

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys, logits=logits)

total_loss = tf.reduce_mean(losses)

trainer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

# training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_losses = []
for iter in range(10):
    train_loss = 0.0
    state = np.zeros((batch_size, state_size))
    for b, (xs_, ys_) in enumerate(gen_batch(gen_data(), batch_size, num_steps)):
        # batch processing
        t_loss, state, _ = sess.run([total_loss, states, trainer],
                                    feed_dict={xs: xs_, ys: ys_, init_state: state})
        train_loss += t_loss
        if b % 100 == 0 and b > 0:
            print("step %d, avg loss/100 steps=%f" % (b, train_loss / 100))
            train_losses.append(train_loss / 100)
            train_loss = 0.0

plt.plot(train_losses)
plt.show()