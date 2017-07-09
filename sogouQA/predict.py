# -*- coding: utf-8 -*- 

import tensorflow as tf
from common.dataset import PaddedDataset
from sogouQA.sogou import batch_size
from sogouQA.model import RNN

def run():
  voca_size = 950
  rnn = RNN(voca_size, state_size=100, batch_size=batch_size)

  saver = tf.train.Saver()
  sess = tf.Session()
  save_pt = "D:/mp/tf-exp/sogouQA/myModel/model.ckpt"
  saver.restore(sess, save_pt)

if __name__ == '__main__': 
  run()