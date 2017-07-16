# -*- coding: utf-8 -*-
import tensorflow as tf

a = tf.range(0, 24)
b = tf.expand_dims(a, 0)
print(a.get_shape())
print(tf.shape(a))

sess = tf.Session()


