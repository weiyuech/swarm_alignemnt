"""Draw the random sample of adopters' reasoning quoted in the paper (Results).

  python3 scripts/sample_justifications.py      # prints 12 adopters' private reasoning

For each adopter (non-source agent that adopted the exploit) in the main Study 1
runs (8 and 20 agents) and Study 2, take its private "thoughts" at the turn it
first adopted, shuffle with a fixed seed and print the first 12. The paper
reports how many of these justify the exploit by its acceptance or by others
using it; that judgement is made by reading the printed text.
"""

import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import analyze  # noqa: E402

ROOTS = ("runs/main", "runs/scale20", "runs/study2")


def adopter_thoughts():
    rows = []
    for root in ROOTS:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.exists(os.path.join(d, "summary.json")) or "honest" in d:
                continue
            r = analyze(d)
            thoughts = {}
            with open(os.path.join(d, "events.jsonl")) as fh:
                for line in fh:
                    e = json.loads(line)
                    if e["kind"] == "decision":
                        thoughts[(e["agent"], e["turn"])] = e.get("thoughts") or ""
            for agent, det in r["agents_detail"].items():
                if agent != r["source"] and det.get("adopt_turn"):
                    rows.append((os.path.basename(d), agent, det["adopt_turn"],
                                 thoughts.get((agent, det["adopt_turn"]), "")))
    return rows


def main(n=12, seed=7):
    rows = adopter_thoughts()
    random.Random(seed).shuffle(rows)
    print("%d adopters; random sample of %d (seed %d):\n" % (len(rows), n, seed))
    for i, (ep, agent, turn, text) in enumerate(rows[:n], 1):
        print("%2d. %s, %s, turn %d\n    %s\n" % (i, ep, agent, turn, text.replace("\n", " ")))


if __name__ == "__main__":
    main()
