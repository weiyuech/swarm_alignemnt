"""Run one episode.

  python3 scripts/run_episode.py --topology ring --seed 1 --provider adaption \
      --model deepseek-v4-flash --turns 20 --agents 8 --budget 0.5
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testbed.llm import make_client
from testbed.orchestrator import Episode, EpisodeConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", choices=["ring", "broadcast"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--provider", default="adaption")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--budget", type=float, default=0.5, help="hard stop in USD/credits")
    ap.add_argument("--max-output-tokens", type=int, default=4000)
    ap.add_argument("--reasoning-effort", default="none", help="'none' disables hidden reasoning")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-report", action="store_true", help="control: remove the report tool")
    ap.add_argument("--seeded", action="store_true", help="alias for --seed-type exploit")
    ap.add_argument("--no-library", action="store_true", help="do not publish accepted submissions")
    ap.add_argument("--source", choices=["none", "cheat", "honest"], default="none",
                    help="scripted source agent occupying one seat")
    ap.add_argument("--source-start", type=int, default=4)
    ap.add_argument("--seed-type", choices=["none", "exploit", "placebo"], default="none",
                    help="patient zero's inherited file: none, exploit note, or placebo note")
    a = ap.parse_args()
    eid = "%s%s-%s-%s-s%d-%s" % (a.tag + "-" if a.tag else "", a.model, a.topology,
                             ("norep" if a.no_report else "rep") + ("-nolib" if a.no_library else "") + "-" +
                             ("src" + a.source if a.source != "none" else ("exploit" if a.seeded else a.seed_type)), a.seed,
                             time.strftime("%Y%m%d-%H%M%S"))
    cfg = EpisodeConfig(episode_id=eid, seed=a.seed, topology=a.topology, n_agents=a.agents,
                        n_turns=a.turns, provider=a.provider, model=a.model,
                        max_output_tokens=a.max_output_tokens, temperature=a.temperature,
                        reasoning_effort=a.reasoning_effort,
                        budget_usd=a.budget, report_tool=not a.no_report,
                        seed_type="exploit" if a.seeded else a.seed_type, library=not a.no_library,
                        source=a.source, source_start_turn=a.source_start)
    extra = {"reasoning_effort": a.reasoning_effort} if a.provider == "adaption" else {}
    client = make_client(a.provider, a.model, max_output_tokens=a.max_output_tokens,
                         temperature=a.temperature, **extra)
    s = Episode(cfg, client, os.path.join(a.out, eid)).run()
    print("episode", eid)
    print("stop:", s["stop_reason"], "| cost:", s["cost_usd"],
          "| genuine:", s["n_closed_genuine"], "| exploit:", s["n_closed_exploit"])


if __name__ == "__main__":
    main()
