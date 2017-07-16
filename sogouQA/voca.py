# -*- coding: utf-8 -*-
import re
from collections import Counter
import jieba
import csv

PAD = "<PAD>"
UNK = "<UNK>"
_WORD_SPLIT = re.compile("([.,!?\"':;)(。，？！：；“‘’”])")


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def char_tokenizer(sentence):
  return list(sentence.strip())


def cn_tokenizer(sentence):
  words = jieba.cut(sentence.strip())
  return list(words)


def fit(doc_pt, voca_pt, freq_threshold=2, tokenizer=basic_tokenizer):
  """
  创建文档集合对应的词汇表，低于freq_threshold的会被过滤
  :param doc_pt: 文档集合
  :param voca_pt: 输出的词汇表，格式：'wid,word,freq'
  :param freq_threshold: 低频过滤
  :param tokenizer: 切词器
  """
  word_cnt = Counter()
  print("read:" + doc_pt)
  with open(doc_pt, encoding="utf-8") as rf:
    for doc in rf:
      word_cnt.update(tokenizer(doc))

  UNK_cnt = sum([w for w, n in word_cnt.items() if n < freq_threshold])
  words = [(UNK, UNK_cnt)] + \
          sorted([(w, n) for w, n in word_cnt.items() if n >= freq_threshold],
                 key=lambda d: d[1], reverse=True)
  print("vocabulary size={}".format(len(words)))
  print("write:" + voca_pt)
  with open(voca_pt, "w", encoding="utf-8", newline="") as wf:
    writer = csv.writer(wf)
    for i, (w, cnt) in enumerate(words):
      writer.writerow([i, w, cnt])


def transform(doc_pt, voca_pt, dwid_pt, tokenizer=basic_tokenizer):
  """读取词汇列表，将文档集合中的word转换成id
  :param doc_pt: 文档集合路径
  :param voca_pt: 词汇表路径，格式:"word1\nword2\n..."
  :param dwid_pt: 转换后的文档集合路径
  :param tokenizer: 切词器，默认按空格切分
  """
  w2id = load_w2id(voca_pt)
  default_id = w2id.get(UNK, 0)
  with open(doc_pt, encoding="utf-8") as rf:
    print("transform:" + doc_pt)
    with open(dwid_pt, 'w', encoding="utf-8") as wf:
      for line in rf:
        words = tokenizer(line.strip())
        ws = [str(w2id.get(w, default_id)) for w in words]
        wf.write(" ".join(ws) + "\n")
      print("write:" + dwid_pt)


def load_w2id(voca_pt):
  """读取词汇表，返回{word:id, ...}"""
  with open(voca_pt, encoding="utf-8") as rf:
    return dict([(w, int(wid)) for wid, w, _ in  csv.reader(rf)])


def load_id2w(voca_pt):
  """读取词汇表，返回{id:w, ...}"""
  with open(voca_pt, encoding="utf-8") as rf:
    return dict([(int(wid), w) for wid, w, _ in  csv.reader(rf)])


def fit_and_transform(doc_pt, voca_pt, dwid_pt, freq_threshold=2, tokenizer=basic_tokenizer):
  """
  读取docs中的词表，并将docs中的word转换为id
  :param doc_pt: 原始文档集合，每行是一个doc
  :param voca_pt: 词汇表：wid  word
  :param dwid_pt: 每行格式'wid1 wid2 ...'
  :param freq_threshold: 频率低于此的词将会被替换成<UNK>
  :param tokenizer: 切词器，默认按空格切分
  """
  fit(doc_pt, voca_pt, freq_threshold, tokenizer)
  transform(doc_pt, voca_pt, dwid_pt, tokenizer)
