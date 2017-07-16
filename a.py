# -*- coding: utf-8 -*-
import tensorflow as tf


pts = ["/Users/yxh/tmp/a.txt"]
dataset = tf.contrib.data.TextLineDataset(pts)
dataset = dataset.map(lambda x: tf.string_to_number(tf.string_split([x]).values, tf.int32))
dataset = dataset.padded_batch(3, tf.TensorShape([None]))
print(dataset)

iterator = dataset.make_initializable_iterator()
e = iterator.get_next()
print(e)
sess = tf.Session()
sess.run(iterator.initializer)
print(sess.run(e))

