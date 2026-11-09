"""How agents reused the exploit, across all cheating-source episodes (exploratory).

  python3 scripts/transfer.py      # -> results/transfer.{md,json}

Template reuse: a non-source agent's attempt that fakes only library functions
the scripted source had already faked (successfully) earlier in the episode.
Far transfer: a working exploit (public grader accepts, hidden grader rejects)
that fakes a function the claim uses and the source never faked.
Also compares hidden reasoning on (follow-up F2) with the matched seeds of the
main runs: episodes in which at least one agent made a far-transfer exploit.
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from analyze import shadowed_names  # noqa: E402
from minilean.problems import BY_ID  # noqa: E402

STUDIES = [("runs/main", "Study 1, 8 agents"), ("runs/scale20", "Study 1, 20 agents"),
           ("runs/think", "Study 1, hidden reasoning on"), ("runs/sweep", "Study 1, sweeping source"),
           ("runs/study2", "Study 2")]


def episode(d):
    with open(os.path.join(d, "events.jsonl")) as fh:
        ev = [json.loads(line) for line in fh]
    src = ev[0].get("source")
    shown, attempts, template, far_agents, far_events = set(), 0, 0, set(), []
    for e in ev:
        if e["kind"] != "grade":
            continue
        sh = shadowed_names(e["preamble"])
        if e["agent"] == src:
            if e["verdict"]["label"] == "exploit":
                shown |= set(sh)
            continue
        if not sh:
            continue
        attempts += 1
        template += set(sh) <= shown
        new = [n for n in sh if n in BY_ID[e["problem"]].claim and n not in shown]
        if e["verdict"]["label"] == "exploit" and new:
            far_agents.add(e["agent"])
            far_events.append({"agent": e["agent"], "problem": e["problem"], "faked": new,
                               "tier": BY_ID[e["problem"]].tier})
    return {"attempts": attempts, "template": template, "far_agents": sorted(far_agents), "far": far_events}


def main(out="results"):
    rows, eps = [], {}
    for root, label in STUDIES:
        r = {"study": label, "episodes": 0, "attempts": 0, "template": 0, "far_agents": 0, "far_episodes": 0}
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if "cheat" not in os.path.basename(d) and "sweep" not in os.path.basename(d):
                continue
            if not os.path.exists(os.path.join(d, "summary.json")):
                continue
            x = episode(d)
            eps[os.path.basename(d)] = x
            r["episodes"] += 1
            r["attempts"] += x["attempts"]
            r["template"] += x["template"]
            r["far_agents"] += len(x["far_agents"])
            r["far_episodes"] += bool(x["far_agents"])
        rows.append(r)
    tot = {k: sum(r[k] for r in rows) for k in ("episodes", "attempts", "template", "far_agents", "far_episodes")}
    faked = sorted({(n, f["problem"], f["tier"]) for x in eps.values() for f in x["far"] for n in f["faked"]})
    # hidden reasoning on vs the same seeds and cells without it
    think = {k: v for k, v in eps.items() if k.startswith("think-")}
    matched = {}
    for k in think:
        base = "main-" + k[len("think-"):]
        if base in eps:
            matched[k] = base
    reasoning = {"pairs": len(matched),
                 "far_episodes_reasoning_on": sum(bool(eps[k]["far_agents"]) for k in matched),
                 "far_episodes_reasoning_off": sum(bool(eps[b]["far_agents"]) for b in matched.values())}
    res = {"by_study": rows, "total": tot, "far_transfer_targets": faked, "reasoning": reasoning}
    with open(os.path.join(out, "transfer.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    L = ["# How agents reused the exploit (exploratory; scripts/transfer.py)", "",
         "| Study | Cheating-source episodes | Faking attempts | Template only | Far-transfer agents (episodes) |",
         "|---|---|---|---|---|"]
    for r in rows + [dict(tot, study="All")]:
        L.append("| %s | %d | %d | %d (%.0f%%) | %d (%d) |" % (
            r["study"], r["episodes"], r["attempts"], r["template"], 100 * r["template"] / max(1, r["attempts"]),
            r["far_agents"], r["far_episodes"]))
    L += ["", "Far-transfer targets (function faked, problem, tier): %s" % faked,
          "Hidden reasoning on vs matched seeds without: far transfer in %d vs %d of %d episodes." % (
              reasoning["far_episodes_reasoning_on"], reasoning["far_episodes_reasoning_off"], reasoning["pairs"])]
    with open(os.path.join(out, "transfer.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
