#!/usr/bin/env python
import sys
import os
import json

from sklearn.metrics import f1_score


def fakedata(metadatas, result_file):
  import random
  print("Creating fakedata", file=sys.stderr)
  with open(result_file, "w+") as fp:
    for md_idx, metadata in enumerate(metadatas):
      ans = random.randint(0, 2)
      data_id = metadata["id"]
      fp.write(f"{data_id}, {ans}\n")
      print(f"{md_idx:02d} / {len(metadatas)}", file=sys.stderr, end="\r")
  print("", file=sys.stderr)

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Missing result file and data folder", file=sys.stderr)

  metadata_file = os.path.join(sys.argv[1], "metadata.json")
  result_file = sys.argv[2]
  
  print(f"score file: {result_file}", file=sys.stderr)
  print(f"metadata file: {metadata_file}", file=sys.stderr)

  with open(metadata_file, "r") as fp:
    metadatas = json.load(fp)

  if not os.path.exists(result_file):
    fakedata(metadatas, result_file)

  with open(result_file, "r") as fp:
    results = list(map(lambda x: list(map(int, x.strip().split(", "))), fp.readlines()))

  id_metadata = {}
  for metadata in metadatas:
    id_metadata[metadata["id"]] = metadata

  y_true = []
  y_pred = []
  for result in results:
    y_pred.append(result[1])
    y_true.append(id_metadata[result[0]]["label"])

  print(f1_score(y_true, y_pred, labels=[0,1,2], average="macro"))
    

  # for r in results:
  #   print(r[0], r[1])


