# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

sess = tf.Session()
sess.run(tf.global_variables_initializer())
q = tf.PaddingFIFOQueue(capacity=10, dtypes=tf.int32, shapes=[[None]])

zs = tf.convert_to_tensor([1]).set_shape([None])
print(zs)

eq1 = q.enqueue(zs)
eq2 = q.enqueue(tf.convert_to_tensor([2,3]).set_shape([None]))
eq3 = q.enqueue(tf.convert_to_tensor([4,5,6]))
dq = q.dequeue()
sess.run([eq1, eq2, eq3])
print(sess.run(dq))  # [4 5 6]
print(sess.run(dq))  # [2 3]
print(sess.run(dq))  # [1]
