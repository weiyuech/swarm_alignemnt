"""Subprocess entry point: read one submission as JSON on stdin, print a verdict."""

import json
import sys

from .grader import grade_inprocess
from .problems import BY_ID


def main():
    req = json.loads(sys.stdin.read())
    v = grade_inprocess(BY_ID[req["pid"]], req.get("preamble", ""), req.get("answer"))
    sys.stdout.write(json.dumps(v.to_dict()))


if __name__ == "__main__":
    main()
