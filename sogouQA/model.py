# -*- coding: utf-8 -*- 
import tensorflow as tf
from rnn.rnnUtil import gen_mask


class RNN:
  def __init__(self, voca_size, state_size, xs, ys, lengths):
    """build the RNN Graph for training"""
    batch_size = tf.shape(xs)[0]
    num_steps = tf.shape(xs)[1]
    weights = gen_mask(num_steps, lengths)
    # embedding
    with tf.name_scope("model"):
      W = tf.get_variable("embedding", [voca_size, state_size], dtype=tf.float32)
      embed_xs = tf.nn.embedding_lookup(W, xs)

      cell = tf.contrib.rnn.LSTMCell(state_size)
      initial_state = cell.zero_state(batch_size, tf.float32)
      # output shape: [batch_size, num_steps, output_size]
      # last_states: [batch_size, state_size]
      outputs, last_states = tf.nn.dynamic_rnn(cell,
                                          inputs=embed_xs,
                                          sequence_length=lengths,
                                          dtype=tf.float32,
                                          initial_state=initial_state)

      softmax_W = tf.get_variable("softmax_W", [state_size, voca_size], dtype=tf.float32)
      softmax_b = tf.get_variable("softmax_b", [voca_size], dtype=tf.float32)

    with tf.name_scope("loss"):
      # 将3-dim tensor转换为Matrix，方便矩阵运算
      flat_outputs = tf.reshape(outputs, [-1, state_size])
      flat_logits = tf.matmul(flat_outputs, softmax_W) + softmax_b
      logits = tf.reshape(flat_logits, [batch_size, num_steps, voca_size])
      self.loss = tf.contrib.seq2seq.sequence_loss(logits, ys, weights)

    # compute accuracy
    self.predicts = tf.cast(tf.argmax(logits, 2), tf.int32)
    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicts, ys), tf.float32))

    with tf.name_scope("optimize"):
      # optimize
      self.train_op = tf.train.AdamOptimizer(0.01).minimize(loss)


class RNNInfer:
  def __init__(self, voca_size, state_size):
    """build the RNN Graph for inference"""
    self.xs = tf.placeholder(tf.int32, [1, None], "input")
    num_steps = tf.shape(self.xs)[1]
    with tf.name_scope("model"):
      W = tf.get_variable("embedding", [voca_size, state_size], dtype=tf.float32)
      embed_xs = tf.nn.embedding_lookup(W, self.xs)

      cell = tf.contrib.rnn.LSTMCell(state_size)
      self.initial_state = cell.zero_state(1, tf.float32)
      # output shape: [batch_size, num_steps, output_size]
      # last_states: [batch_size, state_size]
      outputs, last_states = tf.nn.dynamic_rnn(cell,
                                          inputs=embed_xs,
                                          sequence_length=[num_steps],
                                          dtype=tf.float32,
                                          initial_state=self.initial_state)

      softmax_W = tf.get_variable("softmax_W", [state_size, voca_size], dtype=tf.float32)
      softmax_b = tf.get_variable("softmax_b", [voca_size], dtype=tf.float32)

    # generate a predict
    flat_outputs = tf.reshape(outputs, [-1, state_size])
    flat_logits = tf.matmul(flat_outputs, softmax_W) + softmax_b
    self.logits = tf.reshape(flat_logits, [1, num_steps, voca_size])
    self.predicts = tf.cast(tf.argmax(self.logits, 2), tf.int32)
    self.last_states = last_states
