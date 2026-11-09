"""Pre-registered analysis of the main experiment (see DESIGN.md).

  python3 scripts/aggregate.py runs/main

Writes results/summary.json and results/results.md, and prints the tables.
All statistics are over episodes (the unit of analysis); RQ4 is agent-level
with an episode-clustered bootstrap. Fixed RNG seed, so reruns are identical.
"""

import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze import analyze, load                         # noqa: E402
from minilean.grader import grade                         # noqa: E402

CELL_RE = re.compile(r"-(bc|ring)_(cheat|honest|sweep)_(rep|norep)-s(\d+)$")
N_BOOT = 10000
N_PERM = 10000
RNG = random.Random(20261108)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(xs, n=N_BOOT):
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    ms = sorted(mean([RNG.choice(xs) for _ in xs]) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def perm_test(a, b, n=N_PERM):
    """Two-sided permutation test for a difference in means."""
    obs = mean(a) - mean(b)
    pooled = a + b
    k = len(a)
    hits = 0
    for _ in range(n):
        RNG.shuffle(pooled)
        if abs(mean(pooled[:k]) - mean(pooled[k:])) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def did_perm(a1, a0, b1, b0, n=N_PERM):
    """Interaction (a1 - a0) - (b1 - b0), permuting labels within each pair."""
    obs = (mean(a1) - mean(a0)) - (mean(b1) - mean(b0))
    hits = 0
    for _ in range(n):
        pa, pb = a1 + a0, b1 + b0
        RNG.shuffle(pa)
        RNG.shuffle(pb)
        d = (mean(pa[:len(a1)]) - mean(pa[len(a1):])) - (mean(pb[:len(b1)]) - mean(pb[len(b1):]))
        if abs(d) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def strat_perm(pairs, n=20000):
    """Factorial main effect: mean over strata of (mean a - mean b), with labels
    permuted within each stratum. pairs = [(a_values, b_values), ...]."""
    obs = mean([mean(a) - mean(b) for a, b in pairs])
    hits = 0
    for _ in range(n):
        ds = []
        for a, b in pairs:
            p = a + b
            RNG.shuffle(p)
            ds.append(mean(p[:len(a)]) - mean(p[len(a):]))
        if abs(mean(ds)) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def checks(ep_dir, row, cell):
    """Manipulation checks from DESIGN.md; returns a list of failures."""
    ev = load(ep_dir)
    start = ev[0]
    fails = []
    prompts = [e["text"] for e in ev if e["kind"] == "system_prompt"]
    if cell[2] == "norep" and any("- report:" in p for p in prompts):
        fails.append("report tool present in a report-off prompt")
    if cell[2] == "rep" and not all("- report:" in p for p in prompts):
        fails.append("report tool missing in a report-on prompt")
    src = start.get("source")
    allowed = set(start["neighbours"][src]) | {src} if start["config"]["topology"] == "ring" else set(start["agents"])
    for e in ev:
        if e["kind"] == "library_publish" and e["author"] == src and set(e["recipients"]) != allowed:
            fails.append("source library entry reached the wrong recipients")
            break
    n = bad = 0
    for e in ev:
        if e["kind"] == "grade":
            n += 1
            v = grade(e["problem"], e["preamble"], e["answer"], isolate=False).to_dict()
            # compare the verdict, not error wording (which differs across Python versions)
            if any(v[k] != e["verdict"][k] for k in ("label", "public_pass", "hidden_pass", "shadowed")):
                bad += 1
    if bad:
        fails.append("rescore mismatches: %d/%d" % (bad, n))
    return fails, src


def main(root, out_dir="results"):
    rows = []
    seats = defaultdict(set)
    excluded = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        m = CELL_RE.search(name)
        if not m or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        cell = m.groups()[:3]
        r = analyze(d)
        fails, src = checks(d, r, cell)
        seats[int(m.group(4))].add(src)
        r["cell"] = "%s_%s_%s" % cell
        r["seed"] = int(m.group(4))
        dec = [e for e in load(d) if e["kind"] == "decision" and not e.get("scripted")]
        r["lost_turn_rate"] = round(sum(1 for e in dec if e["parse_error"]) / max(1, len(dec)), 3)
        if fails:
            excluded.append({"episode": name, "reasons": fails})
            continue
        rows.append(r)
    for seed, s in seats.items():
        if len(s) > 1:
            excluded.append({"seed": seed, "reasons": ["source seat differs across cells: %s" % sorted(s)]})

    by = defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r)
    order = ["bc_cheat_rep", "ring_cheat_rep", "bc_cheat_norep", "ring_cheat_norep",
             "bc_honest_rep", "ring_honest_rep"]
    metric = lambda cell, k: [r[k] for r in by[cell]]

    table = []
    for c in order:
        rs = by[c]
        if not rs:
            continue
        ad = metric(c, "adoption_fraction_others")
        lo, hi = boot_ci(ad)
        fa = [r["first_exploit_turn_others"] for r in rs if r.get("first_exploit_turn_others")]
        table.append({
            "cell": c, "n": len(rs), "adoption_mean": round(mean(ad), 3), "adoption_ci": [round(lo, 3), round(hi, 3)],
            "reporters_mean": round(mean(metric(c, "n_reporters")), 2),
            "genuine_mean": round(mean([r["closed_genuine"] or 0 for r in rs]), 2),
            "exploit_credits_mean": round(mean([r["closed_exploit"] or 0 for r in rs]), 2),
            "quit_mean": round(mean(metric(c, "n_quit")), 2),
            "first_adoption_turn_median": statistics.median(fa) if fa else None,
            "lost_turn_rate": round(mean(metric(c, "lost_turn_rate")), 3),
        })

    contrasts = {}

    def contrast(name, a, b, k="adoption_fraction_others"):
        if by[a] and by[b]:
            d, p = perm_test(metric(a, k), metric(b, k))
            contrasts[name] = {"diff": round(d, 3), "p": round(p, 4), "outcome": k, "a": a, "b": b}

    # RQ1 topology, RQ2 report tool (primary outcome)
    contrast("RQ1 ring - broadcast (report on)", "ring_cheat_rep", "bc_cheat_rep")
    contrast("RQ1 ring - broadcast (report off)", "ring_cheat_norep", "bc_cheat_norep")
    contrast("RQ2 report on - off (broadcast)", "bc_cheat_rep", "bc_cheat_norep")
    contrast("RQ2 report on - off (ring)", "ring_cheat_rep", "ring_cheat_norep")
    # RQ3 interaction
    if all(by[c] for c in ("ring_cheat_rep", "ring_cheat_norep", "bc_cheat_rep", "bc_cheat_norep")):
        d, p = did_perm(metric("ring_cheat_rep", "adoption_fraction_others"),
                        metric("ring_cheat_norep", "adoption_fraction_others"),
                        metric("bc_cheat_rep", "adoption_fraction_others"),
                        metric("bc_cheat_norep", "adoption_fraction_others"))
        contrasts["RQ3 (report effect in ring) - (report effect in broadcast)"] = {"diff": round(d, 3), "p": round(p, 4)}
    # Source contrasts (control) and topology x source on reporters
    contrast("Cheat - honest source (broadcast, report on)", "bc_cheat_rep", "bc_honest_rep")
    contrast("Cheat - honest source (ring, report on)", "ring_cheat_rep", "ring_honest_rep")
    contrast("Reporters: cheat - honest (broadcast)", "bc_cheat_rep", "bc_honest_rep", "n_reporters")
    contrast("Reporters: cheat - honest (ring)", "ring_cheat_rep", "ring_honest_rep", "n_reporters")
    contrast("Reporters: ring - broadcast (cheat, report on)", "ring_cheat_rep", "bc_cheat_rep", "n_reporters")
    if all(by[c] for c in ("ring_cheat_rep", "ring_honest_rep", "bc_cheat_rep", "bc_honest_rep")):
        d, p = did_perm(metric("ring_cheat_rep", "n_reporters"), metric("ring_honest_rep", "n_reporters"),
                        metric("bc_cheat_rep", "n_reporters"), metric("bc_honest_rep", "n_reporters"))
        contrasts["Reporters: topology x source interaction"] = {"diff": round(d, 3), "p": round(p, 4)}

    # Factorial main effects (post hoc, see DESIGN.md deviations)
    main_effects = {}
    A = "adoption_fraction_others"
    if all(by[c] for c in ("ring_cheat_rep", "ring_cheat_norep", "bc_cheat_rep", "bc_cheat_norep")):
        d, p = strat_perm([(metric("ring_cheat_rep", A), metric("bc_cheat_rep", A)),
                           (metric("ring_cheat_norep", A), metric("bc_cheat_norep", A))])
        main_effects["Topology on adoption (ring - broadcast)"] = {"diff": round(d, 3), "p": round(p, 4)}
        d, p = strat_perm([(metric("bc_cheat_rep", A), metric("bc_cheat_norep", A)),
                           (metric("ring_cheat_rep", A), metric("ring_cheat_norep", A))])
        main_effects["Report tool on adoption (on - off)"] = {"diff": round(d, 3), "p": round(p, 4)}
    if all(by[c] for c in ("ring_cheat_rep", "bc_cheat_rep", "ring_honest_rep", "bc_honest_rep")):
        d, p = strat_perm([(metric("ring_cheat_rep", "n_reporters"), metric("bc_cheat_rep", "n_reporters")),
                           (metric("ring_honest_rep", "n_reporters"), metric("bc_honest_rep", "n_reporters"))])
        main_effects["Topology on reporters (ring - broadcast)"] = {"diff": round(d, 3), "p": round(p, 4)}

    # RQ4: victims vs bystanders, cheat vs honest source (agent level, clustered bootstrap)
    def rates(rs, group, k):
        xs = [v[k] for r in rs for v in r[group].values()]
        return mean(xs), len(xs)

    rq4 = {}
    cheat_eps = [r for c in order if "cheat" in c for r in by[c]]
    honest_eps = [r for c in order if "honest" in c for r in by[c]]
    for k in ("adopt", "report", "quit"):
        est = {}
        for lab, eps in (("cheat", cheat_eps), ("honest", honest_eps)):
            v, nv = rates(eps, "victims", k)
            b, nb = rates(eps, "bystanders", k)
            est[lab] = {"victim": round(v, 3), "n_victim": nv, "bystander": round(b, 3), "n_bystander": nb}
        did = (est["cheat"]["victim"] - est["cheat"]["bystander"]) - (est["honest"]["victim"] - est["honest"]["bystander"])
        boots = []
        for _ in range(2000):
            ce = [RNG.choice(cheat_eps) for _ in cheat_eps] if cheat_eps else []
            he = [RNG.choice(honest_eps) for _ in honest_eps] if honest_eps else []
            cv, _ = rates(ce, "victims", k)
            cb, _ = rates(ce, "bystanders", k)
            hv, _ = rates(he, "victims", k)
            hb, _ = rates(he, "bystanders", k)
            x = (cv - cb) - (hv - hb)
            if x == x:
                boots.append(x)
        boots.sort()
        ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (None, None)
        rq4[k] = {"by_source": est, "did": round(did, 3) if did == did else None,
                  "did_ci": [round(ci[0], 3), round(ci[1], 3)] if ci[0] is not None else None}

    # Exploratory RQ4: first candidacy, victim vs eligible non-victim
    def fc_rates(eps, status, k):
        xs = [v[k] for r in eps for v in r["rq4_first_candidacy"].values()
              if v["status"] == status and v[k] is not None]
        return mean(xs), len(xs)

    rq4x = {}
    for k in ("adopt", "report", "quit"):
        est = {}
        for lab, eps in (("cheat", cheat_eps), ("honest", honest_eps)):
            v, nv = fc_rates(eps, "victim", k)
            b, nb = fc_rates(eps, "eligible_nonvictim", k)
            est[lab] = {"victim": round(v, 3), "n_victim": nv, "eligible_nonvictim": round(b, 3), "n_nonvictim": nb}
        did = (est["cheat"]["victim"] - est["cheat"]["eligible_nonvictim"]) - \
              (est["honest"]["victim"] - est["honest"]["eligible_nonvictim"])
        boots = []
        for _ in range(2000):
            ce = [RNG.choice(cheat_eps) for _ in cheat_eps]
            he = [RNG.choice(honest_eps) for _ in honest_eps]
            x = (fc_rates(ce, "victim", k)[0] - fc_rates(ce, "eligible_nonvictim", k)[0]) - \
                (fc_rates(he, "victim", k)[0] - fc_rates(he, "eligible_nonvictim", k)[0])
            if x == x:
                boots.append(x)
        boots.sort()
        rq4x[k] = {"by_source": est, "did": round(did, 3) if did == did else None,
                   "did_ci": [round(boots[int(0.025 * len(boots))], 3), round(boots[int(0.975 * len(boots)) - 1], 3)]
                   if boots else None}

    # Transmission channels and ring distance
    channels = defaultdict(int)
    dist = defaultdict(lambda: [0, 0])
    for r in rows:
        for a, how in r["transmission"].items():
            if how != "patient_zero":
                channels[how] += 1
        if r["topology"] == "ring" and r["source_mode"] == "cheat":
            for a, (d, t) in r["adoption_by_ring_distance"].items():
                if d == 0:
                    continue
                dist[d][1] += 1
                dist[d][0] += 1 if t else 0

    # Post hoc (DESIGN.md deviations): exposure and reports about the exploit.
    # Separate RNG so the pre-registered numbers above are unchanged.
    rng2 = random.Random(20261109)

    def pooled(rs, num, den):
        d = sum(r[den] for r in rs)
        return sum(r[num] for r in rs) / d if d else None

    def pooled_ci(rs, num, den, n=N_BOOT):
        vals = sorted(v for v in (pooled([rng2.choice(rs) for _ in rs], num, den) for _ in range(n)) if v is not None)
        return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (None, None)

    def ep_ci(xs, n=N_BOOT):
        ms = sorted(mean([rng2.choice(xs) for _ in xs]) for _ in range(n))
        return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]

    def r3(x):
        return None if x is None else round(x, 3)

    exposure = []
    for c in order:
        rs = by[c]
        if not rs:
            continue
        age = pooled(rs, "adopters_after_exposure", "exposed_others")
        rge = pooled(rs, "exploit_reporters_after_exposure", "exposed_others")
        exposure.append({
            "cell": c, "n": len(rs),
            "exposed_mean": r3(mean(metric(c, "exposure_fraction"))),
            "exposed_ci": [r3(x) for x in ep_ci(metric(c, "exposure_fraction"))],
            "adopt_given_exposed": r3(age),
            "adopt_given_exposed_ci": [r3(x) for x in pooled_ci(rs, "adopters_after_exposure", "exposed_others")]
            if age is not None else None,
            "exploit_report_mean": r3(mean(metric(c, "exploit_reporter_fraction"))),
            "exploit_report_ci": [r3(x) for x in ep_ci(metric(c, "exploit_reporter_fraction"))],
            "exploit_report_given_exposed": r3(rge),
            "unexposed_agents": sum(r["n_never_exposed"] for r in rs),
            "unexposed_adopters": sum(r["adopters_never_exposed"] for r in rs),
            "unexposed_exploit_reporters": sum(r["exploit_reporters_never_exposed"] for r in rs),
            "adoptions_before_exposure": sum(r["adoptions_before_exposure"] for r in rs),
            "reports_before_exposure": sum(r["reports_before_exposure"] for r in rs),
            "dry_run_only": r3(mean(metric(c, "dry_run_only_fraction"))),
            "submitted": r3(mean(metric(c, "exploit_submit_fraction"))),
            "credited": r3(mean(metric(c, "exploit_credited_fraction"))),
            "alarm_episodes": sum(1 for r in rs if r["alarm"]),
            "alarm_latencies": sorted(r["alarm_latency"] for r in rs if r["alarm_latency"] is not None),
        })
    EF = "exploit_reporter_fraction"

    def perm2(a, b, n=N_PERM):
        obs, pooled_ab, k, hits = mean(a) - mean(b), a + b, len(a), 0
        for _ in range(n):
            rng2.shuffle(pooled_ab)
            if abs(mean(pooled_ab[:k]) - mean(pooled_ab[k:])) >= abs(obs) - 1e-12:
                hits += 1
        return obs, (hits + 1) / (n + 1)

    def did2(a1, a0, b1, b0, n=N_PERM):
        obs = (mean(a1) - mean(a0)) - (mean(b1) - mean(b0))
        hits = 0
        for _ in range(n):
            pa, pb = a1 + a0, b1 + b0
            rng2.shuffle(pa)
            rng2.shuffle(pb)
            d = (mean(pa[:len(a1)]) - mean(pa[len(a1):])) - (mean(pb[:len(b1)]) - mean(pb[len(b1):]))
            if abs(d) >= abs(obs) - 1e-12:
                hits += 1
        return obs, (hits + 1) / (n + 1)

    exposure_contrasts = {}
    if by["ring_cheat_rep"] and by["bc_cheat_rep"]:
        d, p = perm2(metric("ring_cheat_rep", EF), metric("bc_cheat_rep", EF))
        exposure_contrasts["Exploit reports: ring - fully connected (cheat, report on)"] = {"diff": round(d, 3), "p": round(p, 4)}
        d, p = perm2(metric("ring_cheat_rep", "exposure_fraction"), metric("bc_cheat_rep", "exposure_fraction"))
        exposure_contrasts["Exposure: ring - fully connected (cheat, report on)"] = {"diff": round(d, 3), "p": round(p, 4)}
    if all(by[c] for c in ("ring_cheat_rep", "ring_honest_rep", "bc_cheat_rep", "bc_honest_rep")):
        d, p = did2(metric("ring_cheat_rep", EF), metric("ring_honest_rep", EF),
                    metric("bc_cheat_rep", EF), metric("bc_honest_rep", EF))
        exposure_contrasts["Exploit reports: topology x source interaction"] = {"diff": round(d, 3), "p": round(p, 4)}

    out = {"n_episodes": len(rows), "excluded": excluded, "cells": table, "contrasts": contrasts,
           "main_effects_post_hoc": main_effects,
           "exposure_post_hoc": exposure, "exposure_contrasts_post_hoc": exposure_contrasts,
           "rq4": rq4, "rq4_exploratory_first_candidacy": rq4x, "transmission_channels": dict(channels),
           "ring_adoption_by_distance": {d: {"adopted": v[0], "agents": v[1]} for d, v in sorted(dist.items())},
           "per_episode": [{k: r[k] for k in ("episode", "cell", "seed", "adoption_fraction_others", "n_reporters",
                                               "closed_genuine", "closed_exploit", "n_quit", "lost_turn_rate",
                                               "first_exploit_turn_others", "exposure_fraction",
                                               "exploit_reporter_fraction", "adopters_after_exposure", "exposed_others",
                                               "alarm", "alarm_latency", "dry_run_only_fraction",
                                               "exploit_submit_fraction", "exploit_credited_fraction",
                                               "agents_detail", "source", "topology")} for r in rows]}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    lines = ["# Main experiment results", "", "Episodes analysed: %d. Excluded: %d." % (len(rows), len(excluded)), "",
             "| Cell | n | Adoption by real agents (mean, 95% CI) | Reporters | Genuine solves | Exploit credits | Quits | First adoption turn (median) | Lost-turn rate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        lines.append("| %s | %d | %.2f [%.2f, %.2f] | %.2f | %.2f | %.2f | %.2f | %s | %.3f |" % (
            t["cell"], t["n"], t["adoption_mean"], t["adoption_ci"][0], t["adoption_ci"][1], t["reporters_mean"],
            t["genuine_mean"], t["exploit_credits_mean"], t["quit_mean"], t["first_adoption_turn_median"],
            t["lost_turn_rate"]))
    lines += ["", "## Contrasts (permutation tests over episodes)", "", "| Contrast | Difference | p |", "|---|---|---|"]
    for k, v in contrasts.items():
        lines.append("| %s | %+.3f | %.4f |" % (k, v["diff"], v["p"]))
    lines += ["", "## Factorial main effects (post hoc; stratified permutation, 20,000 permutations)", "",
              "| Main effect | Difference | p |", "|---|---|---|"]
    for k, v in main_effects.items():
        lines.append("| %s | %+.3f | %.4f |" % (k, v["diff"], v["p"]))
    lines += ["", "Adoption per episode:"]
    for c in order:
        if by[c]:
            lines.append("- %s: %s" % (c, ", ".join("%.2f" % x for x in metric(c, A))))
    lines += ["", "## RQ4: victims vs bystanders", "", "| Outcome | Cheat: victim / bystander | Honest: victim / bystander | DiD [95% CI] |", "|---|---|---|---|"]
    for k, v in rq4.items():
        c, h = v["by_source"]["cheat"], v["by_source"]["honest"]
        lines.append("| %s | %.2f (n=%d) / %.2f (n=%d) | %.2f (n=%d) / %.2f (n=%d) | %s %s |" % (
            k, c["victim"], c["n_victim"], c["bystander"], c["n_bystander"], h["victim"], h["n_victim"],
            h["bystander"], h["n_bystander"], v["did"], v["did_ci"]))
    lines += ["", "## RQ4 exploratory: victim vs eligible non-victim at first candidacy", "",
              "| Outcome | Cheat: victim / eligible non-victim | Honest: victim / eligible non-victim | DiD [95% CI] |",
              "|---|---|---|---|"]
    for k, v in rq4x.items():
        c, h = v["by_source"]["cheat"], v["by_source"]["honest"]
        lines.append("| %s | %.2f (n=%d) / %.2f (n=%d) | %.2f (n=%d) / %.2f (n=%d) | %s %s |" % (
            k, c["victim"], c["n_victim"], c["eligible_nonvictim"], c["n_nonvictim"], h["victim"], h["n_victim"],
            h["eligible_nonvictim"], h["n_nonvictim"], v["did"], v["did_ci"]))
    lines += ["", "## Exposure and reports about the exploit (post hoc)", "",
              "Exposed: share of real agents who could see an accepted exploit or a chat post describing it. "
              "Rates given exposure pool agents across episodes (episode-clustered bootstrap CI). "
              "Exploit reports: report text matching a keyword classifier (scripts/analyze.py).", "",
              "| Cell | Exposed [95% CI] | Adopted given exposed [95% CI] | Reported the exploit [95% CI] | "
              "Reported given exposed | Unexposed agents: adopted / reported |", "|---|---|---|---|---|---|"]
    fmt = lambda x: "-" if x is None else "%.2f" % x
    for t in exposure:
        ci = t["adopt_given_exposed_ci"] or [None, None]
        lines.append("| %s | %s [%s, %s] | %s [%s, %s] | %s [%s, %s] | %s | %d / %d of %d |" % (
            t["cell"], fmt(t["exposed_mean"]), fmt(t["exposed_ci"][0]), fmt(t["exposed_ci"][1]),
            fmt(t["adopt_given_exposed"]), fmt(ci[0]), fmt(ci[1]),
            fmt(t["exploit_report_mean"]), fmt(t["exploit_report_ci"][0]), fmt(t["exploit_report_ci"][1]),
            fmt(t["exploit_report_given_exposed"]), t["unexposed_adopters"], t["unexposed_exploit_reporters"],
            t["unexposed_agents"]))
    lines += ["", "Adoption breakdown (post hoc): share of agents who only tested the exploit in a private dry run / "
              "submitted it / received credit for it. Alarm (post hoc): episodes in which at least one agent filed a "
              "keyword-classified exploit report, and turns from the source's first exploit entry to the first report.", "",
              "| Cell | Dry run only | Submitted | Credited | Episodes with an exploit report | Latency (turns), alarmed episodes |",
              "|---|---|---|---|---|---|"]
    for t in exposure:
        lines.append("| %s | %s | %s | %s | %d / %d | %s |" % (
            t["cell"], fmt(t["dry_run_only"]), fmt(t["submitted"]), fmt(t["credited"]), t["alarm_episodes"], t["n"],
            t["alarm_latencies"] if t["alarm_episodes"] else "-"))
    lines += ["", "Never-exposed agents are counted apart from actions before a later exposure: adoptions before "
              "exposure %d, exploit reports before exposure %d (all cells)." % (
                  sum(t["adoptions_before_exposure"] for t in exposure),
                  sum(t["reports_before_exposure"] for t in exposure))]
    lines += ["", "| Contrast (post hoc) | Difference | p |", "|---|---|---|"]
    for k, v in exposure_contrasts.items():
        lines.append("| %s | %+.3f | %.4f |" % (k, v["diff"], v["p"]))
    tr = [r["transfer"] for r in rows if r["source_mode"] in ("cheat", "sweep")]
    lines += ["", "## How the exploit was reused (exploratory)", "",
              "Shadowing attempts by real agents: %d; reusing only predicates the source had shown: %d; "
              "agents with a working far-transfer exploit (a new predicate the claim uses): %d, in %d of %d "
              "cheating-source episodes." % (
                  sum(t["shadow_attempts"] for t in tr), sum(t["template_attempts"] for t in tr),
                  sum(t["far_transfer_agents"] for t in tr), sum(1 for t in tr if t["far_transfer_agents"]), len(tr))]
    lines += ["", "Transmission channels (adopters other than the source): %s" % dict(channels),
              "", "Ring adoption by distance from the source: %s" % json.dumps(out["ring_adoption_by_distance"]),
              "", "Excluded: %s" % json.dumps(excluded)]
    with open(os.path.join(out_dir, "results.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/main", sys.argv[2] if len(sys.argv) > 2 else "results")
