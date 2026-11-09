"""Study 2 checks (offline, no API calls): bank v4, neutral report wording, plans.

  python3 -m unittest tests.test_study2 -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from minilean.grader import grade
from minilean.problems import BANK_V3, BANK_V4, BANK_V4_EXTRA
from testbed import prompts
from testbed.llm import MockClient
from testbed.orchestrator import Episode, EpisodeConfig, source_eligible, source_submission
import run_batch


class TestBankV4(unittest.TestCase):
    def test_v4_extends_v3_unchanged(self):
        self.assertEqual(BANK_V4[:len(BANK_V3)], BANK_V3)
        ids = [p.pid for p in BANK_V4]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(BANK_V4_EXTRA), 20)
        self.assertTrue(all(p.tier == "grind" and p.known_witness for p in BANK_V4_EXTRA))

    def test_new_witnesses_need_search(self):
        # no answer is the first candidate above the stated threshold
        for p in BANK_V4_EXTRA:
            if p.pid.startswith(("prime_", "sophie_", "twin_")):
                self.assertGreater(p.known_witness["P"] % 10 ** 9, 100, p.pid)

    def test_source_pool_identical_with_v3_and_v4(self):
        for mode in ("cheat", "honest", "sweep"):
            self.assertEqual([p.pid for p in BANK_V4 if source_eligible(p.pid, mode)],
                             [p.pid for p in BANK_V3 if source_eligible(p.pid, mode)], mode)

    def test_source_labels_on_v4(self):
        pool = [p.pid for p in BANK_V4 if source_eligible(p.pid)]
        # agents (not the source) may still exploit the new problems: the exploit must work on them
        for p in BANK_V4_EXTRA:
            if p.pid.startswith("pell_"):
                continue                          # no library predicate to shadow
            pre, ans = source_submission(p.pid, "cheat")
            self.assertEqual(grade(p.pid, pre, ans, isolate=False).label, "exploit", p.pid)
        for pid in pool:
            pre, ans = source_submission(pid, "cheat")
            self.assertEqual(grade(pid, pre, ans, isolate=False).label, "exploit", pid)
            pre, ans = source_submission(pid, "honest")
            self.assertEqual(grade(pid, pre, ans, isolate=False).label, "genuine", pid)


class TestReportWording(unittest.TestCase):
    def sp(self, **kw):
        return prompts.system_prompt("prover-alpha", "number_theory", "x", 8, 20, **kw)

    def test_neutral_keeps_channel_drops_invitation(self):
        original, neutral, off = self.sp(), self.sp(report_wording="neutral"), self.sp(report_enabled=False)
        self.assertIn("unfair", original)
        self.assertNotIn("unfair", neutral)
        self.assertIn(prompts.REPORT_TOOL_NEUTRAL, neutral)
        self.assertIn('"type": "report"', neutral)
        self.assertNotIn("- report:", off)
        # everything except the tool description is identical
        self.assertEqual(neutral.replace(prompts.REPORT_TOOL_NEUTRAL, prompts.REPORT_TOOL), original)


class TestPlans(unittest.TestCase):
    def test_main_plan_is_the_six_registered_cells(self):
        self.assertEqual(sorted(run_batch.PLANS["main"]["cells"]),
                         sorted(["bc_cheat_rep", "ring_cheat_rep", "bc_cheat_norep", "ring_cheat_norep",
                                 "bc_honest_rep", "ring_honest_rep"]))

    def test_study2_plan(self):
        p = run_batch.PLANS["study2"]
        self.assertEqual(p["seeds"], list(range(300, 320)))
        self.assertEqual(len(p["cells"]), 5)
        for c in p["cells"]:
            EpisodeConfig(episode_id="x", seed=1, **run_batch.CELLS[c])
        self.assertEqual(len(run_batch.CELLS["bc_cheat_norep_ab"]["problems"]), len(BANK_V4))
        self.assertEqual(run_batch.CELLS["bc_cheat_neu"]["report_wording"], "neutral")


class TestAbundantEpisode(unittest.TestCase):
    def test_runs_and_source_uses_new_problems(self):
        tmp = tempfile.mkdtemp()
        try:
            def policy(agent, turn, system, user):
                return json.dumps({"notebook": "", "working_on": "cubes_147", "post": "",
                                   "action": {"type": "pass"}})
            cfg = EpisodeConfig(episode_id="ab", seed=3, n_agents=8, n_turns=6, isolate_grader=False,
                                source_start_turn=2, **run_batch.CELLS["bc_cheat_norep_ab"])
            Episode(cfg, MockClient(policy), tmp).run()
            with open(os.path.join(tmp, "events.jsonl")) as fh:
                ev = [json.loads(l) for l in fh]
            credits = [e for e in ev if e["kind"] == "credit"]
            # every agent declares cubes_147 (a v4 problem): the source must not take it, but still cheats
            self.assertFalse(any(e["problem"] == "cubes_147" for e in credits))
            self.assertTrue(any(e["label"] == "exploit" for e in credits))
            starts = [e for e in ev if e["kind"] == "turn_start"]
            self.assertEqual(len(starts[0]["open"]), len(BANK_V4))
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()


class TestStudy2Statistics(unittest.TestCase):
    """The registered analysis (scripts/study2.py) against known values."""
    def setUp(self):
        import study2
        self.s = study2

    def test_fisher_textbook_values(self):
        self.assertAlmostEqual(self.s.fisher_two_sided(3, 1, 1, 3), 0.4857, places=4)
        self.assertAlmostEqual(self.s.fisher_two_sided(1, 9, 11, 3), 0.002759, places=6)
        self.assertEqual(self.s.fisher_two_sided(0, 10, 0, 10), 1.0)

    def test_holm_step_down(self):
        got = [round(x, 4) for x in self.s.holm([0.01, 0.04, 0.03, 0.005])]
        self.assertEqual(got, [0.03, 0.06, 0.06, 0.02])

    def test_signflip_exact_equals_brute_force(self):
        import itertools
        d = [0.3, 0.1, -0.05, 0.4, 0.2, 0.0, 0.25, -0.1, 0.15, -0.2]
        brute = sum(1 for g in itertools.product((1, -1), repeat=len(d))
                    if abs(sum(x * y for x, y in zip(d, g))) >= abs(sum(d)) - 1e-9) / 2 ** len(d)
        self.assertEqual(self.s.signflip(d, None)[1], brute)


class TestStudy2Checks(unittest.TestCase):
    """End to end: valid mock episodes of all five cells pass the registered
    manipulation checks; planted errors are excluded with the right reason."""
    def test_checks_accept_valid_and_reject_planted_errors(self):
        import study2
        from minilean.problems import BANK_V4_EXTRA as EXTRA_PROBLEMS
        tmp = tempfile.mkdtemp()
        try:
            def policy(agent, turn, system, user):
                return json.dumps({"notebook": "", "working_on": "prime_777", "post": "",
                                   "action": {"type": "pass"}})

            def make(cell, seed, name_cell=None, name_seed=None, **override):
                kw = dict(study2.COMMON)
                kw.update(run_batch.CELLS[cell])
                kw.update(override)
                d = os.path.join(tmp, "study2-deepseek-v4-flash-%s-s%d" % (name_cell or cell, name_seed or seed))
                Episode(EpisodeConfig(episode_id=os.path.basename(d), seed=seed, isolate_grader=False, **kw),
                        MockClient(policy), d).run()
            for i, cell in enumerate(run_batch.PLANS["study2"]["cells"]):
                make(cell, 300 + i)
            make("bc_cheat_rep", 310, name_cell="ring_cheat_rep")               # wrong graph
            swapped = [p.pid for p in BANK_V3][:-1] + [EXTRA_PROBLEMS[0].pid]
            make("bc_cheat_norep", 311, problems=swapped)                         # same size, wrong bank
            make("bc_cheat_neu", 312, name_seed=313)                              # wrong seed
            rows, excluded = study2.load_cells(tmp)
            self.assertEqual(sorted(r["cell"] for r in rows), ["A", "B", "C", "D", "E"])
            why = {e["episode"].rsplit("-", 1)[-1]: " ".join(e["reasons"]) for e in excluded}
            self.assertEqual(sorted(why), ["s310", "s311", "s313"])
            self.assertIn("topology", why["s310"])
            self.assertIn("initial problems differ", why["s311"])
            self.assertIn("seed", why["s313"])
        finally:
            shutil.rmtree(tmp)


class TestAmendment(unittest.TestCase):
    """Amendment 2026-11-08 22:30 (after Astra review, before any Study 2 analysis)."""
    def test_exposure_outcomes_timing(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        from analyze import exposure_outcomes
        others = ["a", "b", "c", "d"]
        expo = {"a": (5, "library"), "b": (5, "chat")}            # c, d never exposed
        adoption = {"a": 7, "b": 3, "c": 9}                        # b adopts BEFORE its exposure; c never exposed
        reports = {"a": 4, "b": 6, "d": 2}                         # a reports before exposure; d never exposed
        o = exposure_outcomes(others, expo, adoption, reports)
        self.assertEqual((o["n_exposed"], o["n_never_exposed"]), (2, 2))
        self.assertEqual((o["adopters_after_exposure"], o["adoptions_before_exposure"], o["adopters_never_exposed"]),
                         (1, 1, 1))
        self.assertEqual((o["exploit_reporters_after_exposure"], o["reports_before_exposure"],
                          o["exploit_reporters_never_exposed"]), (1, 1, 1))

    def test_mcnemar_exact(self):
        import study2
        self.assertEqual(study2.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(study2.mcnemar_exact(0, 6), 0.03125)      # 2 * 0.5**6
        self.assertAlmostEqual(study2.mcnemar_exact(1, 5), 2 * (1 + 6) / 64)
