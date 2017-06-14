# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
from rnn.rnnUtil import mask_weights

class RNN:
  def __init__(self, voca_size,
               state_size=5,
               kept_prob=1,
               batch_size=2):
    """build the RNN Graph"""
    self.xs = tf.placeholder(tf.int32, [batch_size, None])
    self.lengths = tf.placeholder(tf.int32, [batch_size])
    self.ys = tf.placeholder(tf.int32, [batch_size, None])

    num_steps = tf.shape(self.xs)[1]
    weights = mask_weights(num_steps, self.lengths)
    # embedding

    W = tf.get_variable("embedding", [voca_size, state_size], dtype=tf.float32)
    embed_xs = tf.nn.embedding_lookup(W, self.xs)

    # RNN
    cell = tf.contrib.rnn.LSTMCell(state_size)
    outputs, states = tf.nn.dynamic_rnn(cell,
                                        inputs=embed_xs,
                                        sequence_length=self.lengths,
                                        dtype=tf.float32)

    softmax_W = tf.get_variable("softmax_W", [state_size, voca_size+1], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [voca_size+1], dtype=tf.float32)
    flat_outputs = tf.reshape(outputs, [batch_size * num_steps, state_size])
    flat_logits = tf.matmul(flat_outputs, softmax_W) + softmax_b
    logits = tf.reshape(flat_logits, [batch_size, num_steps, voca_size + 1])
    loss = tf.contrib.seq2seq.sequence_loss(logits, self.ys, weights)
    self._loss = tf.reduce_sum(loss) / batch_size
    # compute accuracy
    predicts = tf.cast(tf.argmax(logits, 2), tf.int32)
    self._accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, self.ys), tf.float32))
    # optimize
    self._train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

  def loss(self):
    return self._loss

  def accuracy(self):
    return self._accuracy

  def train_step(self):
    return self._train_step



if __name__ == '__main__':
  xs = tf.placeholder(tf.float32, [None, None])
  xs_ = np.random.rand(2, 4)
  print(xs_)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print()
    print(sess.run(xs, feed_dict={xs: xs_}))
