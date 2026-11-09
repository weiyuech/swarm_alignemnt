"""Matched-seed comparison of a follow-up against the main experiment.

  python3 scripts/compare_followups.py runs/think   # F2: reasoning on vs off
  python3 scripts/compare_followups.py runs/sweep   # F3: sweeping vs regular cheater

Pairs each follow-up episode with the main-experiment episode of the same
seed and topology/report cell (for sweep cells, the matching cheat cell), and
reports the mean paired difference with a sign-flip permutation test.
"""

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import analyze   # noqa: E402

RE = re.compile(r"-(bc|ring)_(cheat|sweep)_(rep|norep)-s(\d+)$")
OUTCOMES = ["adoption_fraction_others", "n_reporters", "n_quit", "closed_genuine", "closed_exploit",
            "far_transfer_agents"]   # last one exploratory
RNG = random.Random(20261108)


def index(root):
    out = {}
    for name in os.listdir(root):
        m = RE.search(name)
        d = os.path.join(root, name)
        if m and os.path.exists(os.path.join(d, "summary.json")):
            out[(m.group(1), m.group(3), int(m.group(4)))] = analyze(d)
    return out


def signflip(diffs, n=20000):
    obs = sum(diffs) / len(diffs)
    hits = sum(1 for _ in range(n)
               if abs(sum(d * RNG.choice((1, -1)) for d in diffs) / len(diffs)) >= abs(obs) - 1e-12)
    return obs, (hits + 1) / (n + 1)


def main(root):
    fol, base = index(root), index("runs/main")
    keys = sorted(k for k in fol if k in base)
    print("%d matched pairs from %s vs runs/main" % (len(keys), root))
    cells = sorted({(k[0], k[1]) for k in keys}) + [None]
    report = {}
    for cell in cells:
        ks = [k for k in keys if cell is None or (k[0], k[1]) == cell]
        label = "pooled" if cell is None else "%s_%s" % cell
        report[label] = {}
        for o in OUTCOMES:
            diffs = [(fol[k][o] or 0) - (base[k][o] or 0) for k in ks]
            if not diffs:
                continue
            d, p = signflip(diffs)
            report[label][o] = {"followup_mean": round(sum(fol[k][o] or 0 for k in ks) / len(ks), 3),
                                "main_mean": round(sum(base[k][o] or 0 for k in ks) / len(ks), 3),
                                "paired_diff": round(d, 3), "p": round(p, 4), "n_pairs": len(ks)}
        lost = [fol[k].get("parse_errors", 0) for k in ks]
        report[label]["parse_errors_mean_followup"] = round(sum(lost) / len(lost), 1) if lost else None
    print(json.dumps(report, indent=2))
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "compare_%s.json" % os.path.basename(root.rstrip("/"))), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
