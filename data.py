import os
import xml.etree.ElementTree as ET

data_folder = "test_data"
data_files = ["RTE1_dev1_3ways.xml"]

entailmentToNum = {
  "YES": 2,
  "NO": 0,
  "UNKNOWN": 1
}

def getTestData():
  tree = ET.parse(os.path.join(data_folder, data_files[0]))
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

if __name__ == "__main__":
  d = getTestData()
  print(len(d))
  print(d[0]["t"])
