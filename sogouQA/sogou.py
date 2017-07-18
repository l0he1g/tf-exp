# -*- coding: utf-8 -*-
from sogouQA.model import RNN, RNNInfer
from sogouQA import voca, data_utils
import tensorflow as tf

data_dir = "/Users/yxh/mp/tf-exp/resources/test/"
train_pt = data_dir + "dwid.txt"
voca_pt = data_dir + "voca.py.txt"
model_pt = data_dir + "model/model.ckpt"

w2id = voca.load_w2id(voca_pt)
id2w = voca.load_id2w(voca_pt)
voca_size = len(w2id)
batch_size = 64
state_size = 100
max_iter = 3000


def train():
  data = data_utils.create_lm_dataset(train_pt, batch_size)
  iterator = data.make_one_shot_iterator()
  xs, ys, lengths = iterator.get_next()

  rnn = RNN(voca_size, state_size, xs, ys, lengths)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    loss, accuracy = (0.0, 0.0)
    for iter in range(max_iter):
      i_loss, i_accuracy, _ = sess.run([rnn.loss, rnn.accuracy, rnn.train_op])
      loss += i_loss
      accuracy += i_accuracy

      if iter % 100 == 99:
        saver.save(sess, model_pt)
        print("save model:" + model_pt)
        print("iter %d, loss=%f, accuracy=%f" % (iter, loss / 100, accuracy / 100))
        loss, accuracy = (0.0, 0.0)


def infer(start):
  print("infer:\n" + start, flush=True)
  ws = [w2id[w] for w in voca.cn_tokenizer(start)]

  rnn = RNNInfer(voca_size, state_size)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    print("model_pt:" + model_pt)
    # save_pt = tf.train.latest_checkpoint(model_pt)
    # print("restore:" + save_pt)
    saver.restore(sess, model_pt)
    logits, ys, last_states = sess.run([rnn.logits, rnn.predicts, rnn.last_states], feed_dict={rnn.xs: [ws]})
    print("logits:", end=" ")
    print(logits)
    print("ys:", end=" ")
    print(ys)
    print("predict seq=" + " ".join([id2w[y] for y in ys[0]]), flush=True)
    print("衍生预测开始")
    for i in range(10):
      feeds = {rnn.xs: [ws]}
      ys, last_states = sess.run([rnn.predicts, rnn.last_states], feed_dict=feeds)
      ws = ws + [ys[0][-1]]
      print("input:" + " ".join([id2w[y] for y in ws]), flush=True)
      print("output:" + " ".join([id2w[y] for y in ys[0]]), flush=True)


if __name__ == '__main__':
  #train()
  print("please input a start:")
  #start = sys.stdin.readline()
  start = "qq"
  #while start:
  infer(start)
#    start = sys.stdin.readline()
