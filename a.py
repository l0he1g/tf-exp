# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from rnn.rnnUtil import  mask_weights

if __name__ == '__main__':
  sess = tf.Session()
  a = tf.random_normal((2,3,4,5))
  b = tf.arg_max(a, 0)
  _a, _b = sess.run([a, b])
  print(a.shape)
  print(b.shape)

