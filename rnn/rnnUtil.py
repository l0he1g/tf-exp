# -*- coding: utf-8 -*- 
import tensorflow as tf

def mask_weights(num_steps, lengths):
  batch_size = tf.size(lengths)
  nums = tf.range(num_steps)
  num_mat = tf.tile([nums], [batch_size, 1])
  len_mat = tf.tile(tf.expand_dims(lengths, 1), [1, num_steps])
  return tf.cast(tf.less(num_mat, len_mat), tf.float32)

if __name__ == "__main__":
  a = mask_weights(5, [3, 4])
  sess = tf.Session()
  print(sess.run(a))