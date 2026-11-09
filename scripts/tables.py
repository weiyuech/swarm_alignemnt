"""Main-text tables, built only from the generated result files.

  python3 scripts/tables.py      # after aggregate.py (main, scale20) and compare_followups.py

Writes results/tables.md. Percentages are shares of non-source agents.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)

CELLS = [("bc_cheat_rep", "Cheating", "Fully connected", "On"),
         ("ring_cheat_rep", "Cheating", "Ring", "On"),
         ("bc_cheat_norep", "Cheating", "Fully connected", "Off"),
         ("ring_cheat_norep", "Cheating", "Ring", "Off"),
         ("bc_honest_rep", "Honest (control)", "Fully connected", "On"),
         ("ring_honest_rep", "Honest (control)", "Ring", "On")]


def load(*p):
    with open(R(*p)) as fh:
        return json.load(fh)


def pct(x):
    return "-" if x is None else "%d" % round(100 * x)


def pval(p):
    return "p<0.001" if p < 0.001 else "p=%.2f" % p if p >= 0.01 else "p=%.3f" % p


def pp(d, p):
    return "%+d (%s)" % (round(100 * d), pval(p))


def table1(s8, s20):
    e8 = {r["cell"]: r for r in s8["exposure_post_hoc"]}
    e20 = {r["cell"]: r for r in s20["exposure_post_hoc"]}
    a8 = {r["cell"]: r for r in s8["cells"]}
    a20 = {r["cell"]: r for r in s20["cells"]}
    L = ["**Table 1. Exposure, adoption and reports about the exploit.** Cells give 8-agent / 20-agent values "
         "(10 / 6 episodes per condition; percent of non-source agents). *Exposed*: could see an accepted exploit or "
         "a chat post describing it. *Reported*: filed a report about the exploit (keyword classifier, post hoc).", "",
         "| Source | Graph | Report tool | Exposed | Adopted | Adopted, if exposed | Reported |",
         "|---|---|---|---|---|---|---|"]
    for c, src, g, rep in CELLS:
        L.append("| %s | %s | %s | %s / %s | %s / %s | %s / %s | %s |" % (
            src, g, rep, pct(e8[c]["exposed_mean"]), pct(e20[c]["exposed_mean"]),
            pct(a8[c]["adoption_mean"]), pct(a20[c]["adoption_mean"]),
            pct(e8[c]["adopt_given_exposed"]), pct(e20[c]["adopt_given_exposed"]),
            "n/a" if rep == "Off" else "%s / %s" % (pct(e8[c]["exploit_report_mean"]), pct(e20[c]["exploit_report_mean"]))))
    c8, c20 = s8["contrasts"], s20["contrasts"]
    m8, m20 = s8["main_effects_post_hoc"], s20["main_effects_post_hoc"]
    x8, x20 = s8["exposure_contrasts_post_hoc"], s20["exposure_contrasts_post_hoc"]
    rows = [
        ("Ring vs fully connected, adoption (report off)", "pre-registered",
         c8["RQ1 ring - broadcast (report off)"], c20["RQ1 ring - broadcast (report off)"]),
        ("Ring vs fully connected, adoption (report on)", "pre-registered",
         c8["RQ1 ring - broadcast (report on)"], c20["RQ1 ring - broadcast (report on)"]),
        ("Report tool on vs off, adoption (fully connected)", "pre-registered",
         c8["RQ2 report on - off (broadcast)"], c20["RQ2 report on - off (broadcast)"]),
        ("Ring vs fully connected, adoption (main effect)", "post hoc",
         m8["Topology on adoption (ring - broadcast)"], m20["Topology on adoption (ring - broadcast)"]),
        ("Report tool on vs off, adoption (main effect)", "post hoc",
         m8["Report tool on adoption (on - off)"], m20["Report tool on adoption (on - off)"]),
        ("Ring vs fully connected, exposure (report on)", "post hoc",
         x8["Exposure: ring - fully connected (cheat, report on)"], x20["Exposure: ring - fully connected (cheat, report on)"]),
        ("Ring vs fully connected, exploit reports beyond the honest control (graph x source)", "post hoc",
         x8["Exploit reports: topology x source interaction"], x20["Exploit reports: topology x source interaction"]),
    ]
    L += ["", "*Effects (percentage points; permutation tests over episodes, two-sided)*", "",
          "| Effect | Status | 8 agents | 20 agents |", "|---|---|---|---|"]
    for name, status, v8, v20 in rows:
        L.append("| %s | %s | %s | %s |" % (name, status, pp(v8["diff"], v8["p"]), pp(v20["diff"], v20["p"])))
    un = sum(r["unexposed_agents"] for r in s8["exposure_post_hoc"] + s20["exposure_post_hoc"] if "cheat" in r["cell"])
    ua = sum(r["unexposed_adopters"] for r in s8["exposure_post_hoc"] + s20["exposure_post_hoc"] if "cheat" in r["cell"])
    ur = sum(r["unexposed_exploit_reporters"] for r in s8["exposure_post_hoc"] + s20["exposure_post_hoc"] if "cheat" in r["cell"])
    L += ["", "Unexposed agents in cheating-source episodes: %d; of these, %d adopted and %d reported the exploit." % (un, ua, ur)]
    return L


def table2(s8, s20, think, sweep):
    t, w = think["pooled"], sweep["pooled"]
    r8, r20 = s8["rq4"], s20["rq4"]
    x8, x20 = s8["rq4_exploratory_first_candidacy"], s20["rq4_exploratory_first_candidacy"]

    def paired(o, scale=1, unit=""):
        return "%+.*f%s (%s)" % (0 if scale == 100 else 1, o["paired_diff"] * scale, unit, pval(o["p"]))

    def did(v):
        return "%+d [%+d, %+d]" % (round(100 * v["did"]), round(100 * v["did_ci"][0]), round(100 * v["did_ci"][1]))
    L = ["**Table 2. Follow-ups.** F2 and F3 are matched-seed comparisons with the main runs (same seed and cell; "
         "sign-flip permutation tests). Victim vs bystander: difference-in-differences against the honest-source "
         "control, with 95% episode-clustered bootstrap intervals, in percentage points.", "",
         "| Follow-up | Outcome | Difference | Reading |", "|---|---|---|---|",
         "| F2: hidden reasoning on (%d pairs, report off) | Adoption | %s | no reduction |" % (
             t["adoption_fraction_others"]["n_pairs"], paired(t["adoption_fraction_others"], 100, " pp")),
         "| | Genuine solves per episode | %s | more honest capability |" % paired(t["closed_genuine"]),
         "| | Agents adapting the exploit to an unshown predicate, per episode | %s | exploratory |" % paired(
             t["far_transfer_agents"]),
         "| F3: cheater also sweeps open problems (%d pairs) | Exploit closures per episode | %s | |" % (
             w["closed_exploit"]["n_pairs"], paired(w["closed_exploit"])),
         "| | Adoption | %s | no change |" % paired(w["adoption_fraction_others"], 100, " pp"),
         "| | Reporters per episode | %s | no change |" % paired(w["n_reporters"]),
         "| Victim vs bystander (pre-registered RQ4) | Adoption | 8 agents %s; 20 agents %s | no extra effect |" % (
             did(r8["adopt"]), did(r20["adopt"])),
         "| | Report | 8 agents %s; 20 agents %s | |" % (did(r8["report"]), did(r20["report"])),
         "| Victim vs eligible non-victim (exploratory) | Adoption | 8 agents %s; 20 agents %s | |" % (
             did(x8["adopt"]), did(x20["adopt"]))]
    return L


def main():
    s8, s20 = load("summary.json"), load("scale20", "summary.json")
    out = table1(s8, s20) + ["", ""] + table2(s8, s20, load("compare_think.json"), load("compare_sweep.json"))
    with open(R("tables.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
