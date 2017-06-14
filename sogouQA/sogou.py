# -*- coding: utf-8 -*- 
from common.dataset import PaddedDataset
from sogouQA.model import RNN
import json
import tensorflow as tf

n_epochs = 100
batch_size = 64

def run():
  data_pt = "D:/mp/tf-exp/resources/test/train.1.json"
  docs = read_docs(data_pt)
  print("n(doc)=%d" % len(docs))
  data = PaddedDataset(docs, batch_size=batch_size, filter_freq=5, min_doc_len=3, max_doc_len=10)
  rnn = RNN(data.voca_size, state_size=100, batch_size=batch_size)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_batch = data.batch_nums()
    for epoch in range(n_epochs):
      loss, accuracy = (0.0, 0.0)
      for i in range(n_batch):
        xs, ys, lengths = data.next_batch()
        feeds = {rnn.xs: xs, rnn.ys: ys, rnn.lengths: lengths}
        i_loss, i_accuracy, _ = sess.run([rnn.loss(), rnn.accuracy(), rnn.train_step()],
                                   feed_dict=feeds)
        loss += i_loss
        accuracy += i_accuracy

      print("epoch %d, loss=%f, accuracy=%f" % (epoch, loss/n_batch, accuracy/n_batch))

def read_docs(data_pt):
  rf = open(data_pt, encoding="utf-8")
  return  [json.loads(line)["query"] for line in rf]


if __name__ == '__main__':
  run()
