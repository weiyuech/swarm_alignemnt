"""Per-episode chat-routing check on Study 1 logs (run after the fact).

  python3 scripts/check_routing.py    # prints pass/fail counts per run folder

Applies the observation-level check that Study 2 ran on every episode
(scripts/study2.py): no observation may show an agent a chat post that was not
addressed to it. Study 1 verified routing with unit tests; this script checks
the logged episodes directly.
"""

import glob
import json
import os
from collections import Counter, defaultdict

ROOTS = ("runs/main", "runs/scale20", "runs/think", "runs/sweep")


def episode_ok(path):
    with open(path) as fh:
        ev = [json.loads(line) for line in fh]
    can_see = defaultdict(set)
    for e in ev:
        if e["kind"] == "post":
            for r in e["recipients"]:
                can_see[r].add(e["msg_id"])
    return not any(e["kind"] == "observation" and not set(e.get("visible_msg_ids") or []) <= can_see[e["agent"]]
                   for e in ev)


def main():
    total = Counter()
    for root in ROOTS:
        c = Counter()
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.exists(os.path.join(d, "summary.json")):
                c["pass" if episode_ok(os.path.join(d, "events.jsonl")) else "fail"] += 1
        print(root, dict(c))
        total.update(c)
    print("all", dict(total))


if __name__ == "__main__":
    main()
