# -*- coding: utf-8 -*-
import re
import jieba

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