#!/usr/bin/env python
import sys
import statistics 

import spacy

from .raw_data import loadDataCupMetadata, loadDataCupArticle
from .search import search_multiple_docs
from .roberta import compareText

def pipeline(data_dir, preprocessor=search_multiple_docs, predictor=compareText):
  metadatas = loadDataCupMetadata(data_dir)
  print(f"Number of testdata {len(metadatas)}")
  result = []
  nlp = spacy.load("en_core_web_lg")
  try:
    for meta_index, metadata in enumerate(metadatas):

      claim = metadata["claim"]
      related_articles = metadata["related_articles"]

      docs = [loadDataCupArticle(related_article, data_dir=data_dir) for related_article in related_articles]
      search_results = [relavent_doc["original"].strip() for relavent_doc in search_multiple_docs(query=claim, docs=docs, nlp=nlp, threshold=0.1)]

      scores = [compareText(search_result, claim) for search_result in search_results]

      print(f"{meta_index:02d} / {len(metadatas)}", end="\r")
      
      # use the most occuring
      result.append((metadata["id"], max(set(scores), key = scores.count)))
  except KeyboardInterrupt:
    pass
  print()
  return result

def main():
  data_dir = "/usr/local/dataset"
  if len(sys.argv) > 1:
    data_dir = sys.argv[1]

  result = pipeline(data_dir)

  result_file = "/usr/local/predictions.txt"
  if len(sys.argv) > 2:
    result_file = sys.argv[2]
  with open(result_file, "w+") as fp:
    for data_id, data_prediction in result:
      fp.write(f"{data_id}, {data_prediction}\n")

  print("Finished", file=sys.stderr)
