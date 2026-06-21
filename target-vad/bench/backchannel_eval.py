"""Measure the lexical backchannel-vs-question classifier against labeled
interjections. Prints accuracy + a confusion breakdown so any future borrowed
model can be compared against this baseline."""

import json
import sys

from modes.talkback.intent import classify_interjection


def main(path: str = "bench/backchannel_labels.json") -> None:
    cases = json.load(open(path))
    correct = 0
    # confusion key = gold_initial + pred_initial (B/I)
    confusion = {"BB": 0, "BI": 0, "IB": 0, "II": 0}
    misses = []
    for c in cases:
        pred = classify_interjection(c["text"]).value
        gold = c["label"]
        if pred == gold:
            correct += 1
        else:
            misses.append((c["text"], gold, pred))
        confusion[gold[0] + pred[0]] += 1
    n = len(cases)
    print(f"lexical accuracy: {correct}/{n} = {correct / n:.1%}")
    print(f"confusion (gold,pred) BB/BI/IB/II: {confusion}")
    if misses:
        print("misclassified:")
        for text, gold, pred in misses:
            print(f"  {gold}->{pred}: {text!r}")


if __name__ == "__main__":
    main(*sys.argv[1:])
