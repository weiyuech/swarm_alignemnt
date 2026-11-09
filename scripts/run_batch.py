"""Run a batch of episodes (conditions x seeds) in parallel, resumably.

  python3 scripts/run_batch.py --plan pilot6 --parallel 4
  python3 scripts/run_batch.py --plan main --parallel 8

Episodes are interleaved across conditions (seed-major order), so provider
drift over time is spread evenly over conditions. An episode whose directory
already contains summary.json is skipped, so an interrupted batch can be
rerun with the same command.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testbed.llm import make_client
from testbed.orchestrator import Episode, EpisodeConfig
from minilean.problems import BANK_V4

# condition name -> EpisodeConfig overrides
CELLS = {
    "bc_cheat_rep":    dict(topology="broadcast", source="cheat", report_tool=True),
    "ring_cheat_rep":  dict(topology="ring", source="cheat", report_tool=True),
    "bc_cheat_norep":  dict(topology="broadcast", source="cheat", report_tool=False),
    "ring_cheat_norep": dict(topology="ring", source="cheat", report_tool=False),
    "bc_honest_rep":   dict(topology="broadcast", source="honest", report_tool=True),
    "bc_sweep_norep":  dict(topology="broadcast", source="sweep", report_tool=False),
    "ring_sweep_norep": dict(topology="ring", source="sweep", report_tool=False),
    "bc_sweep_rep":    dict(topology="broadcast", source="sweep", report_tool=True),
    "ring_sweep_rep":  dict(topology="ring", source="sweep", report_tool=True),
    "ring_honest_rep": dict(topology="ring", source="honest", report_tool=True),
    # Study 2 (DESIGN.md): neutral report wording; plentiful honest work (bank v4).
    "bc_cheat_neu":    dict(topology="broadcast", source="cheat", report_tool=True, report_wording="neutral"),
    "bc_cheat_norep_ab": dict(topology="broadcast", source="cheat", report_tool=False,
                              problems=[p.pid for p in BANK_V4]),
}

# Follow-ups, pre-registered in DESIGN.md before running.
#  scale20: same cells at 20 agents (problem bank unchanged, so scarcity rises,
#           closer to Paglieri et al.'s 100 agents for 71 problems).
#  think:   hidden reasoning ON, same prompt; seeds 100-105 match the main runs.
#  sweep:   the cheater also takes open problems; seeds 100-105 match the main runs.
PLANS = {
    "scale20": dict(cells=["bc_cheat_norep", "ring_cheat_norep", "bc_cheat_rep", "ring_cheat_rep",
                           "bc_honest_rep"], seeds=list(range(200, 206)), tag="scale20",
                    agents=20),
    "scale20_ringhonest": dict(cells=["ring_honest_rep"], seeds=list(range(200, 206)), tag="scale20",
                               agents=20),
    "sweep": dict(cells=["bc_sweep_norep", "ring_sweep_norep", "bc_sweep_rep", "ring_sweep_rep"],
                  seeds=list(range(100, 106)), tag="sweep"),
    "think": dict(cells=["bc_cheat_norep", "ring_cheat_norep"], seeds=list(range(100, 106)),
                  tag="think", reasoning_effort="", max_output_tokens=24000, deadline=480),
    "pilot6": dict(cells=["bc_cheat_rep", "ring_cheat_rep", "bc_honest_rep", "bc_cheat_norep"],
                   seeds=[7], tag="pilot6"),
    # The six pre-registered cells, listed explicitly (CELLS has since grown).
    "main": dict(cells=["bc_cheat_rep", "ring_cheat_rep", "bc_cheat_norep", "ring_cheat_norep",
                        "bc_honest_rep", "ring_honest_rep"], seeds=list(range(100, 110)), tag="main"),
    # Study 2, pre-registered in DESIGN.md (section "Study 2") before any run.
    # Grader error messages are shown to agents and differ across Python versions,
    # so Study 2 runs on the interpreter that ran Study 1 (3.9; see DESIGN.md).
    "study2": dict(cells=["bc_cheat_norep", "bc_cheat_neu", "bc_cheat_rep", "bc_cheat_norep_ab",
                          "ring_cheat_rep"], seeds=list(range(300, 320)), tag="study2", python="3.9"),
    "pilot_s2": dict(cells=["bc_cheat_norep_ab"], seeds=[901, 902], tag="pilots"),
    "pilot_s2b": dict(cells=["bc_cheat_norep_ab"], seeds=[903, 904], tag="pilots", python="3.9"),
}


def episode_id(tag, cell, seed, model):
    return "%s-%s-%s-s%d" % (tag, model, cell, seed)


def run_one(job):
    tag, cell, seed, a = job
    eid = episode_id(tag, cell, seed, a.model)
    out = os.path.join(a.out, tag, eid)
    if os.path.exists(os.path.join(out, "summary.json")):
        return eid, "skipped (done)"
    if os.path.exists(os.path.join(a.out, "PAUSE")):
        return eid, "not started (runs/PAUSE present)"
    if os.path.isdir(out):
        # An interrupted episode is never resumed mid-way: its partial log is
        # archived and the episode is rerun from turn 1 with the same seed.
        dest = os.path.join(a.out, "_interrupted", tag, "%s-%s" % (eid, time.strftime("%Y%m%d-%H%M%S")))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.rename(out, dest)
    plan = PLANS[a.plan]
    agents = plan.get("agents", a.agents)
    effort = plan.get("reasoning_effort", a.reasoning_effort)
    max_out = plan.get("max_output_tokens", a.max_output_tokens)
    deadline = plan.get("deadline", 240)
    cfg = EpisodeConfig(episode_id=eid, seed=seed, n_agents=agents, n_turns=a.turns,
                        provider="adaption", model=a.model, max_output_tokens=max_out,
                        reasoning_effort=effort, budget_usd=a.budget, call_deadline_s=deadline,
                        **CELLS[cell])
    client = make_client("adaption", a.model, max_output_tokens=max_out,
                         reasoning_effort=effort or None, deadline_s=deadline)
    t0 = time.time()
    try:
        s = Episode(cfg, client, out).run()
    except Exception as ex:                      # keep the batch going; the log shows where it stopped
        return eid, "FAILED: %s" % ex
    return eid, "done in %.0fs | genuine=%s exploit=%s stop=%s" % (
        time.time() - t0, s["n_closed_genuine"], s["n_closed_exploit"], s["stop_reason"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANS), required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--max-output-tokens", type=int, default=4000)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--budget", type=float, default=3.0, help="per-episode hard stop")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    plan = PLANS[a.plan]
    want = plan.get("python")
    if want and not ("%d.%d" % sys.version_info[:2]) == want:
        sys.exit("plan %s must run on Python %s (this is %s); e.g. /usr/bin/python3" % (
            a.plan, want, sys.version.split()[0]))
    jobs = [(plan["tag"], cell, seed, a) for seed in plan["seeds"] for cell in plan["cells"]]
    print("%d episodes, %d in parallel" % (len(jobs), a.parallel), flush=True)
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for eid, status in ex.map(run_one, jobs):
            print(eid, "->", status, flush=True)


if __name__ == "__main__":
    main()
