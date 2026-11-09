"""Re-grade every logged submission and check that the verdicts reproduce.

  python3 scripts/rescore.py runs/<episode_id>
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minilean.grader import grade


def main(ep_dir):
    n = bad = 0
    with open(os.path.join(ep_dir, "events.jsonl")) as fh:
        for line in fh:
            e = json.loads(line)
            if e["kind"] != "grade":
                continue
            n += 1
            v = grade(e["problem"], e["preamble"], e["answer"]).to_dict()
            if v != e["verdict"]:
                bad += 1
                print("MISMATCH seq", e["seq"], e["agent"], e["problem"])
    print("regraded %d submissions, %d mismatches" % (n, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
