#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.python.layers.core as layers_core
from rnn.rnnUtil import gen_mask

tgt_sos_id = 0
tgt_eos_id = 1

class TrainGraph:
  def __init__(self, src_voca_size,
               dest_voca_size,
               embedding_size,
               state_size,
               dataset,
               learning_rate,
               max_gradient_norm):
    iterator = dataset.make_one_shot_iterator()
    src_inputs, src_lengths, dest_inputs, dest_lengths = iterator.get_next()
    num_steps = tf.shape(dest_inputs)[1]
    weights = gen_mask(num_steps, dest_lengths)

    with tf.name_space("model"):
      # Embedding
      src_embedding = tf.get_variable(
        "embedding_encoder", [src_voca_size, embedding_size])
      dest_embedding = tf.get_variable(
        "embedding_decoder", [dest_voca_size, embedding_size])

      src_emb_inputs = tf.embedding_lookup(src_embedding, src_inputs)
      dest_emb_inputs = tf.embedding_lookup(dest_embedding, dest_inputs)
      src_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
      src_outputs, src_state = tf.nn.dynamic_rnn(
        src_cell, src_emb_inputs, sequence_length=src_lengths)

      dest_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)

      helper = tf.contrib.seq2seq.TrainingHelper(dest_emb_inputs, dest_lengths)
      projection_layer = layers_core.Dense(dest_voca_size, use_bias=False)

      # Decoder
      decoder = tf.contrib.seq2seq.BasicDecoder(
        dest_cell, helper, src_state,
        output_layer=projection_layer)
      # Dynamic decoding
      outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
      logits = outputs.rnn_output

      # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      #   labels=dest_inputs, logits=logits)
      # train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)
      self.loss = tf.contrib.seq2seq.sequence_loss(logits, dest_inputs, weights)

    with tf.name_space("optimize"):
      params = tf.trainable_variables()
      gradients = tf.gradients(self.loss, params)
      clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params))


class InferGraph:
  def __init__(self, src_voca_size,
               dest_voca_size,
               embedding_size,
               state_size,
               dataset,
               max_iter):
    iterator = dataset.make_one_shot_iterator()
    src_inputs, src_lengths = iterator.get_next()
    batch_size = tf.shape(src_inputs)[0]

    with tf.name_space("model"):
      # Embedding
      src_embedding = tf.get_variable(
        "embedding_encoder", [src_voca_size, embedding_size])
      dest_embedding = tf.get_variable(
        "embedding_decoder", [dest_voca_size, embedding_size])

      src_emb_inputs = tf.embedding_lookup(src_embedding, src_inputs)

      src_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
      _, src_state = tf.nn.dynamic_rnn(
        src_cell, src_emb_inputs, sequence_length=src_lengths)

      dest_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)

      helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        dest_embedding,
        tf.fill([batch_size], tgt_sos_id), tgt_eos_id)
      projection_layer = layers_core.Dense(dest_voca_size, use_bias=False)

      # Decoder
      decoder = tf.contrib.seq2seq.BasicDecoder(
        dest_cell, helper, src_state, output_layer=projection_layer)
      # Dynamic decoding
      outputs, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder, maximum_iterations=max_iter)
      self.outputs = outputs.sample_id


def train():
  pass


def infer():
  pass

