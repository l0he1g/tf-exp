# -*- coding: utf-8 -*-
from sogouQA import voca
import unittest

test_dir = "/Users/yxh/mp/tf-exp/resources/test/"
doc_pt = test_dir + "docs.txt"
voca_pt = test_dir + "voca.py.txt"
dwid_pt = test_dir + "dwid.txt"

class VocaTest:

  def test_cn_tokenizer(self):
    s = "我的家，是中国。#:你呢？"
    print(voca.cn_tokenizer(s))

  def test_fit(self):
    voca.fit(doc_pt, voca_pt, freq_threshold=1, tokenizer=voca.char_tokenizer)

  def test_transform(self):
    voca.transform(doc_pt, voca_pt, dwid_pt, voca.char_tokenizer)

  def test_fit_and_transform(self):
    voca.fit_and_transform(doc_pt, voca_pt, dwid_pt, freq_threshold=1, tokenizer=voca.basic_tokenizer)

if __name__ == "__main__":
  unittest.main()
