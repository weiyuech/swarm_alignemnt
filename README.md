# Swarm testbed

A contained multi-agent testbed for studying how a grader exploit spreads through a
research swarm, and who reports it. LLM agents compete to solve number-theory claims; a
scripted agent wins either by exploiting the grader or, in the control, honestly. The
communication graph (one shared channel or a ring), a private report tool and the amount
of honest work can be varied.

## Setup

Python 3.9, standard library only (matplotlib 3.9.2 is needed for figures). Use 3.9 to
reproduce outputs exactly: grader error messages shown to agents are worded differently
on newer versions.

```bash
python3 -m unittest discover -s tests -v   # offline checks, no API calls
```

New runs need an API key for a hosted DeepSeek-V4-Flash endpoint; `testbed/llm.py` shows
where the endpoint and key are configured. Re-analysing the published logs needs no key.

## Logs

The episode logs, pre-registrations and design log are a release download. From the
repository root:

```bash
curl -L -o episode-logs.tar.gz https://github.com/weiyuech/swarm_alignemnt/releases/download/v1.0/episode-logs.tar.gz
tar xzf episode-logs.tar.gz   # restores runs/, DESIGN.md and the cached report labels
```

## Reproduce the analyses

```bash
python3 scripts/aggregate.py runs/main                     # Study 1, 8 agents -> results/
python3 scripts/aggregate.py runs/scale20 results/scale20  # Study 1, 20 agents
python3 scripts/compare_followups.py runs/think            # follow-up: hidden reasoning
python3 scripts/compare_followups.py runs/sweep            # follow-up: sweeping source
python3 scripts/study2.py runs/study2                      # Study 2 (pre-registered)
python3 scripts/validate_reports.py runs/study2            # report-rule check (cached labels)
python3 scripts/tables.py                                  # tables
python3 scripts/main_figures.py && python3 scripts/figures.py   # figures
python3 scripts/sample_justifications.py                   # sample of adopters' reasoning
```

## Run new episodes

```bash
python3 scripts/run_batch.py --plan main --parallel 10     # Study 1 (resumable)
python3 scripts/run_batch.py --plan study2 --parallel 10   # Study 2
python3 scripts/run_episode.py --topology ring --seed 1    # a single episode
```

## Layout

| Path | Contents |
|---|---|
| `minilean/` | Sandboxed interpreter, problem bank, public and hidden graders |
| `testbed/` | Episode loop, scripted source, communication graphs, prompts, model client |
| `scripts/` | Batch runner, analyses, tables and figures |
| `tests/` | Grader, isolation, source and Study 2 checks |

Agent code never runs as Python: a whitelist interpreter evaluates it with step and size
limits in a separate process, with no access to the shell, file system or network. Agents
see the shared library and their notebook only as text in each turn's observation.
