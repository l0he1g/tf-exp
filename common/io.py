# -*- coding: utf-8 -*-
from common import tokenrizer as tk

def readDocs(doc_pt, tokenrizer = "cn"):
  with open(doc_pt, encoding="utf-8") as rf:
    docs = []
    for doc in rf:
      if tokenrizer == "cn":
        docs.append(tk.cn_tokenizer(doc))
      elif tokenrizer == "char":
        docs.append(tk.char_tokenizer(doc))
      else:
        docs.append(tk.basic_tokenizer(doc))

  return docs