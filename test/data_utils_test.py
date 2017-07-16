# -*- coding: utf-8 -*-
import tensorflow as tf

from sogouQA.data_utils import create_lm_dataset


def verify_create_dataset():
  pt = '/Users/yxh/mp/tf-exp/resources/test/dwid.txt'
  data = create_lm_dataset(pt)
  print(data)
  iterator = data.make_initializable_iterator()
  with tf.Session() as sess:
    sess.run(iterator.initializer)
    batch = iterator.get_next()
    print(batch)
    print(sess.run(batch))


if __name__ == '__main__':
  verify_create_dataset()
