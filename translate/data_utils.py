# -*- coding: utf-8 -*-
from collections import defaultdict
import csv
from common.io import readDocs
from common import tokenrizer as tk
from common.voca import build_voca, Voca


class DataHelper:
  """将原始文档集合转换为数字"""

  def __init__(self, doc_pt, tokenrizer="cn", freq_threshold=2):
    docs = readDocs(doc_pt, tokenrizer)
    self.voca = build_voca(docs, freq_threshold)

