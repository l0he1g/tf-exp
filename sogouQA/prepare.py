# -*- coding: utf-8 -*-
import json
from sogouQA.voca import fit_and_transform, cn_tokenizer


def run():
  data_dir = "/Users/yxh/mp/tf-exp/resources/test/"
  data_pt = data_dir + "train.1.json"
  with open(data_pt, encoding="utf-8") as rf:
    docs = [json.loads(line)["query"] for line in rf]

  doc_pt = data_dir + "docs.txt"
  with open(doc_pt, "w", encoding="utf-8") as wf:
    wf.write("\n".join(docs))

  voca_pt = data_dir + "voca.py.txt"
  dwid_pt = data_dir + "dwid.txt"
  fit_and_transform(doc_pt, voca_pt, dwid_pt, 1, cn_tokenizer)


if __name__ == '__main__':
  run()
