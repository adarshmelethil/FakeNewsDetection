import os
import json
from functools import lru_cache

import xml.etree.ElementTree as ET

data_folder = "test_data"
data_files = ["RTE1_dev1_3ways.xml"]

entailmentToNum = {
  "YES": 2,
  "NO": 0,
  "UNKNOWN": 1
}

test_data = None
def getTestData():
  global test_data
  if test_data is not None:
    return test_data

  from .RTE1_dev1_3ways import RTE1_dev1_3ways
  tree = ET.fromstring(RTE1_dev1_3ways)
  root = tree.getroot()

  pair = [c for c in root]
  test_data = []
  for p in pair:
    texts = [c for c in p]
    test_data.append({
      "id": p.attrib["id"],
      "entailment": entailmentToNum[p.attrib["entailment"]],
      "task": p.attrib["task"],
      texts[0].tag: texts[0].text,
      texts[1].tag: texts[1].text
    })

  return test_data

def loadDataCupMetadata(data_dir="/usr/local/dataset"):
  metadata_file = os.path.join(data_dir, "metadata.json")

  with open(metadata_file, "r") as fp:
    metadata = json.load(fp)

  return metadata

@lru_cache(maxsize=None)
def loadDataCupArticle(article_id, data_dir="/usr/local/dataset"):
  file_path = os.path.join(data_dir, "articles", f"{article_id}.txt")
  with open(file_path, "r") as fp:
    return fp.read()


def main():
  from sys import argv
  path = "/usr/local/dataset"
  if len(argv) > 1:
    path = argv[1]
  
  import pprint
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(loadDataCupMetadata(data_dir=path))

if __name__ == "__main__":
  main()
