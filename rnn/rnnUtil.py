# -*- coding: utf-8 -*- 
import tensorflow as tf

def gen_mask(num_steps, lengths):
  """
  generate a mask matrix for RNN padding
  :param num_steps: padding后的step数目
  :param lengths: 实际的每个batch中step数目
  :return: mask matrix, padding为0，其余为1
  """
  batch_size = tf.size(lengths)
  nums = tf.range(num_steps)
  num_mat = tf.tile([nums], [batch_size, 1])
  len_mat = tf.tile(tf.expand_dims(lengths, 1), [1, num_steps])
  return tf.cast(tf.less(num_mat, len_mat), tf.float32)

if __name__ == "__main__":
  a = gen_mask(5, [3, 4])
  sess = tf.Session()
  print(sess.run(a))