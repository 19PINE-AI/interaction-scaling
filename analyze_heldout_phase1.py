"""Analyze the held-out Phase 1 code harness results and emit a comparison
against the in-distribution Phase 1 numbers.

Reads:
  - results/heldout_phase1/code_heldout_harness.json  (this run)
  - results/hard_benchmarks/code_onpolicy_run{1,2,3}.json  (in-dist)

Writes a markdown summary to stdout (caller redirects).
"""

import json
from pathlib import Path
from statistics import mean, stdev


def load(p):
    with open(p) as f:
        return json.load(f)


def summarize(entries):
    n = len(entries)
    ss_pass = sum(1 for e in entries if e.get("ss_meets"))
    rv_pass = sum(1 for e in entries if e.get("rv_meets"))
    ss_tokens = mean(e.get("ss_tokens", 0) for e in entries)
    rv_tokens = mean(e.get("rv_tokens", 0) for e in entries)
    return {
        "n": n,
        "ss_pass": ss_pass,
        "rv_pass": rv_pass,
        "ss_rate": ss_pass / n if n else 0,
        "rv_rate": rv_pass / n if n else 0,
        "ss_tokens": ss_tokens,
        "rv_tokens": rv_tokens,
    }


def main():
    indist_runs = []
    for r in [1, 2, 3]:
        p = Path(f"results/hard_benchmarks/code_onpolicy_run{r}.json")
        if p.exists():
            indist_runs.append(load(p))

    indist_summaries = [summarize(r) for r in indist_runs]
    indist_ss = [s["ss_rate"] for s in indist_summaries]
    indist_rv = [s["rv_rate"] for s in indist_summaries]
    indist_ss_tok = [s["ss_tokens"] for s in indist_summaries]
    indist_rv_tok = [s["rv_tokens"] for s in indist_summaries]

    heldout = load("results/heldout_phase1/code_heldout_harness.json")
    # Single run expected; group by run_id if multiple
    by_run = {}
    for e in heldout:
        by_run.setdefault(e.get("run_id", 1), []).append(e)
    ho_summaries = [summarize(by_run[r]) for r in sorted(by_run)]

    print("# Held-out Phase 1 (Sonnet 4 harness) — code modality\n")
    print("## In-distribution baseline (15 tasks, 3 runs, claude-sonnet-thinking)\n")
    print("| Run | N | SS pass | RV pass | SS rate | RV rate | SS tokens | RV tokens |")
    print("|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(indist_summaries, 1):
        print(f"| {i} | {s['n']} | {s['ss_pass']} | {s['rv_pass']} | "
              f"{s['ss_rate']*100:.1f}% | {s['rv_rate']*100:.1f}% | "
              f"{s['ss_tokens']:.0f} | {s['rv_tokens']:.0f} |")
    print(f"| **mean ± sd** | 15 | — | — | "
          f"**{mean(indist_ss)*100:.1f} ± {stdev(indist_ss)*100:.1f}%** | "
          f"**{mean(indist_rv)*100:.1f} ± {stdev(indist_rv)*100:.1f}%** | "
          f"{mean(indist_ss_tok):.0f} | {mean(indist_rv_tok):.0f} |\n")

    print("## Held-out (32 tasks, this run)\n")
    print("| Run | N | SS pass | RV pass | SS rate | RV rate | SS tokens | RV tokens |")
    print("|---|---|---|---|---|---|---|---|")
    for rid, s in zip(sorted(by_run), ho_summaries):
        print(f"| {rid} | {s['n']} | {s['ss_pass']} | {s['rv_pass']} | "
              f"{s['ss_rate']*100:.1f}% | {s['rv_rate']*100:.1f}% | "
              f"{s['ss_tokens']:.0f} | {s['rv_tokens']:.0f} |")
    print()

    # Head-to-head comparison: per-task SS/RV pairs on held-out
    print("## Per-task held-out outcomes\n")
    print("| task_id | bug_class | SS | RV | rv_iters | Δ |")
    print("|---|---|---|---|---|---|")

    # Load task metadata for bug_class
    tasks = load("data/hard_benchmarks/code/code_tasks_heldout_v2.json")
    bug_class_by_id = {t["task_id"]: t.get("bug_class", "?") for t in tasks}

    fixes = 0     # SS fail, RV pass
    regressions = 0  # SS pass, RV fail
    both_pass = 0
    both_fail = 0
    for e in sorted(heldout, key=lambda x: x["task_id"]):
        ss = e.get("ss_meets")
        rv = e.get("rv_meets")
        delta = ""
        if ss and rv:
            both_pass += 1
            delta = "="
        elif not ss and rv:
            fixes += 1
            delta = "+1 (fix)"
        elif ss and not rv:
            regressions += 1
            delta = "-1 (regress)"
        else:
            both_fail += 1
            delta = "= (still fail)"
        bc = bug_class_by_id.get(e["task_id"], "?")
        print(f"| {e['task_id']} | {bc} | "
              f"{'pass' if ss else 'fail'} | "
              f"{'pass' if rv else 'fail'} | "
              f"{e.get('rv_iters')} | {delta} |")
    print()
    print(f"**Both pass:** {both_pass}  "
          f"**Harness fixes (SS fail → RV pass):** {fixes}  "
          f"**Regressions (SS pass → RV fail):** {regressions}  "
          f"**Both fail:** {both_fail}\n")

    # Compact comparison table
    ho = ho_summaries[0]
    print("## Headline comparison\n")
    print("| Split | N | Single-shot | Reviewed | Δ (pp) |")
    print("|---|---|---|---|---|")
    print(f"| In-dist (15 tasks, 3-run mean) | 15 | "
          f"{mean(indist_ss)*100:.1f}% | {mean(indist_rv)*100:.1f}% | "
          f"+{(mean(indist_rv)-mean(indist_ss))*100:.1f} |")
    print(f"| Held-out (this run) | {ho['n']} | "
          f"{ho['ss_rate']*100:.1f}% | {ho['rv_rate']*100:.1f}% | "
          f"+{(ho['rv_rate']-ho['ss_rate'])*100:.1f} |\n")


if __name__ == "__main__":
    main()
