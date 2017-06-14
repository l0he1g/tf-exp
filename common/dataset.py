# -*- coding: utf-8 -*-
from collections import Counter
from itertools import chain
import numpy as np
from random import shuffle

class PaddedDataset:
  def __init__(self, lines, char=True, batch_size=64,
               filter_freq=0, min_doc_len=1, max_doc_len=10):
    """
    :param lines: 每行一个doc, doc由words表示
    :param seq: word分隔符，默认是char
    """
    self.batch_size = batch_size

    if char:
      lines = [list(line.strip()) for line in lines if line]
    else:
      lines = [line.strip().split() for line in lines if line]

    word_freqs = Counter(chain.from_iterable(lines))
    self.words = ["<UNK>"] + [w for w, f in word_freqs.items() if f > filter_freq]
    self.voca_size = len(self.words)
    self.w2ids = dict(zip(self.words, range(self.voca_size)))
    print("n(word)=%d" % self.voca_size)

    # translate words to ids
    unk = self.w2ids["<UNK>"]
    self.docs = [[self.w2ids.get(w, unk) for w in line] for line in lines]

    # filter short and long documents
    self.docs = [doc for doc in self.docs if min_doc_len <= len(doc) <= max_doc_len]
    # 游标
    self.i = 0

  def next_batch(self):
    """
    读取一个text batch，并同时返回batch中每个term seq的真实长度
    :return: (text_batch, lengths)
    """
    if self.i + self.batch_size > len(self.docs):
      self.i = 0
      #shuffle(self.docs)

    end = self.i + self.batch_size
    xs = self.docs[self.i: end]
    lengths = [len(doc) for doc in xs]
    # targets多了个"end" term
    ts = [doc[1:] + [len(self.words)] for doc in self.docs]

    self.i = end

    max_len = max(lengths)
    padded_xs = self.padding(xs, max_len)
    padded_ts = self.padding(ts, max_len)
    return padded_xs, padded_ts, lengths

  def padding(self, batch, max_len):
    padded_batch = np.zeros([self.batch_size, max_len], np.float32)
    for i in range(self.batch_size):
      for j in range(len(batch[i])):
        padded_batch[i, j] = batch[i][j]

    return padded_batch

  def batch_nums(self):
    return len(self.docs) // self.batch_size

def run():
  pt = "../resources/test/test.txt"
  lines = open(pt, encoding="utf-8")
  data = PaddedDataset(lines, False, min_doc_len=3, batch_size=2)
  batch = data.next_batch()
  print(batch)


if __name__ == '__main__':
  run()
