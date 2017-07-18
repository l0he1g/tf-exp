# -*- coding: utf-8 -*-
from collections import defaultdict, Counter
import csv

class Voca:

  def __init__(self, default_word="<UNK>"):
    """默认default word的wid=0"""
    self.word_freq = defaultdict(int)
    self.default_word = default_word
    self.w2id = {default_word: 0}
    self.id2w = {0: default_word}

  def size(self):
    return len(self.w2id)

  def fromCounter(self, cnt, freq_threshold=2):
    for word, freq in cnt.items():
      if freq < freq_threshold:
        word  = self.default_word
      self.update(word, freq)

    return self

  def get_word(self, wid):
    return self.id2w.get(wid, self.default_word)

  def get_wid(self, word):
    return self.w2id.get(word, self.default_id)

  def update(self, word, freq=1):
    if word not in self.w2id:
      self.addWord(word, freq)
    else:
      self.word_freq[word] += freq

    return self.w2id[word]

  def addWord(self, word, freq=1):
    wid = len(self.w2id)
    self.w2id[word] = wid
    self.id2w[wid] = word
    self.word_freq = freq

  def setDefaultWordFreq(self, freq):
    self.word_freq[self.default_word] = freq

  def save(self, pt):
    with open(pt, "w", encoding="utf-8", newline="") as wf:
      writer = csv.writer(wf)
      for wid, word in sorted(self.id2w.items()):
        writer.writerow([wid, word, self.word_freq[word]])
      print("save vocabulary:" + pt)

  def load(self, pt):
    print("load vocabulary:" + pt)
    with open(pt, "r", encoding="utf-8") as rf:
      reader = csv.reader(rf)
      for wid, word, freq in reader:
        wid, freq = int(wid), int(freq)
        self.w2id[word] = wid
        self.id2w[wid] = word
        self.word_freq[word] = freq

      self.default_word = self.id2w[0]

  def transform_words(self, words):
    return [self.get_wid(word) for word in words]

  def transform_docs(self, docs):
    return list(map(self.transform_words, docs))

  def transform_wids(self, wids):
    return [self.get_word(wid) for wid in wids]

  def transform_dwids(self, dwids):
    return list(map(self.transform_wids, dwids))


def build_voca(docs, freq_threshold=2):
  """词频低于freq_threshold的替换为default_word"""
  word_cnt = Counter()
  for words in docs:
    word_cnt.update(words)

  voca = Voca().fromCounter(word_cnt, freq_threshold)
  return voca
