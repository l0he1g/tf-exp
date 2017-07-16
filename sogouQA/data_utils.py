# -*- coding: utf-8 -*-
import tensorflow as tf


def create_lm_dataset(dwid_pt, batch_size=3):
  """
  读取文档集合，转换为language model training Dataset
  :return: (xs, ys, lengths)
  """
  data = tf.contrib.data.TextLineDataset([dwid_pt])
  data = data.map(lambda x: tf.string_to_number(tf.string_split([x]).values, tf.int32))
  data = data.map(lambda ws: (tf.slice(ws, [0], [tf.size(ws) - 1]),
                              tf.slice(ws, [1], [tf.size(ws) - 1]),
                              tf.size(ws) - 1))
  padded_data = data.padded_batch(batch_size,
                                  padded_shapes=(tf.TensorShape([None]),
                                                 tf.TensorShape([None]),
                                                 tf.TensorShape([])))
  return padded_data.repeat()
