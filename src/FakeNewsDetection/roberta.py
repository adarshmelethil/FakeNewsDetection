
import torch
from .raw_data import getTestData

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
# 0: contradiction
# 1: neutral
# 2: entailment

def compareText(text1: str, text2: str):
  tokens = roberta.encode(text1, text2)
  return roberta.predict('mnli', tokens).argmax().item()

import time

def timeCompareText(func, *args, **kwargs):
  start = time.time()
  resp = func(*args, **kwargs)
  end = time.time()
  return end - start, resp


def main():
  test_data = getTestData()

  counter = [0, 0]
  mistakes = {}
  for i, td in enumerate(test_data):
    t, actual = timeCompareText(compareText, td["t"], td["h"])
    expected = td["entailment"]
    counter[actual==expected] += 1

    print(f"{i:04}: {t:010} -> {actual==expected}")

    if actual!=expected:
      mistaked = dict(td)
      mistaked["predicted"] = actual
      mistakes[td["id"]] = mistaked


  print()
  print(counter)
  print(f"correct: {counter[1]/sum(counter)}")

  print("\n--- Mistakes ---\n")
  numToEntailment = {0: "contradiction", 1: "neutral", 2: "entailment"}
  for m_id, mistake in mistakes.items():
    print(f"--- {m_id} ---")
    print(f"t: {mistake['t']}")
    print(f"h: {mistake['h']}")
    print(f"actual: {numToEntailment[mistake['entailment']]}, predicted: {numToEntailment[mistake['predicted']]}")
    print("-"*80)

if __name__ == "__main__":
  main()
