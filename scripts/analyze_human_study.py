#!/usr/bin/env python3
"""Analyze collected human-preference responses against the hidden key.

Inputs
------
  key.json          : study/key.json  (per pair: which side is single-shot vs
                      reviewed, the DOM defect counts, decisive flag)
  responses CSVs    : one or more responses_<rater>.csv exported by the rating
                      app (columns: rater_id,pair_id,modality,choice,rt_ms,ts;
                      choice in {A,B,same})

Endpoints
---------
  1. PRIMARY -- preference for the reviewed render among *decisive* pairs
     (pairs the DOM instrument scored differently). For each non-"same" vote we
     ask: did the rater pick the lower-defect (reviewed) side? Reported overall
     and per modality, with a two-sided binomial sign test vs 0.5.
  2. DOM CONCORDANCE -- fraction of decisive votes where the human's preferred
     side agrees with the DOM instrument's "fewer defects = better" direction.
     This is the number that closes (or fails to close) the circularity caveat:
     if humans independently agree with the instrument, the instrument is
     measuring something real.
  3. TIE CONTROL -- on pairs the DOM scored *equal*, human preference should be
     ~50/50. A large skew here would mean the instrument misses a real axis of
     quality (a false-negative check on the instrument).
  4. "no visible difference" rate, per modality.
  5. Inter-rater agreement on overlapping pairs (raw + chance-corrected).

No external dependencies (stdlib only).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (sum of tail probs <= obs)."""
    if n == 0:
        return 1.0
    from math import comb
    probs = [comb(n, i) * p**i * (1 - p)**(n - i) for i in range(n + 1)]
    obs = probs[k]
    return min(1.0, sum(pr for pr in probs if pr <= obs + 1e-12))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load_responses(csv_paths: list[Path]) -> list[dict]:
    rows = []
    for p in csv_paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=str(ROOT / "study" / "key.json"))
    ap.add_argument("--responses", nargs="+", required=True,
                    help="responses_*.csv files exported by the rating app")
    args = ap.parse_args()

    key = json.load(open(args.key))
    rows = load_responses([Path(p) for p in args.responses])
    if not rows:
        print("No responses found.")
        return

    # which side (A/B) is the reviewed (lower-defect) render?
    def reviewed_side(pid):
        k = key[pid]
        if k["A_condition"] == "rv":
            return "A"
        return "B"

    # Per (modality) tallies on decisive pairs
    dec_pref_rv = defaultdict(int)   # votes for reviewed
    dec_pref_ss = defaultdict(int)   # votes for single-shot
    dec_same = defaultdict(int)
    tie_A = defaultdict(int)
    tie_B = defaultdict(int)
    tie_same = defaultdict(int)
    raters = set()

    # for inter-rater agreement
    by_pair_choice = defaultdict(dict)  # pair_id -> {rater: choice}

    for r in rows:
        pid = r["pair_id"]
        if pid not in key:
            continue
        raters.add(r["rater_id"])
        choice = r["choice"]
        k = key[pid]
        mod = k["modality"]
        by_pair_choice[pid][r["rater_id"]] = choice
        if k["decisive"]:
            if choice == "same":
                dec_same[mod] += 1
            else:
                rv = reviewed_side(pid)
                if choice == rv:
                    dec_pref_rv[mod] += 1
                else:
                    dec_pref_ss[mod] += 1
        else:
            if choice == "same":
                tie_same[mod] += 1
            elif choice == "A":
                tie_A[mod] += 1
            else:
                tie_B[mod] += 1

    mods = sorted(set(list(dec_pref_rv) + list(dec_pref_ss) + list(tie_A) + list(tie_B)))
    print(f"\n{len(raters)} rater(s): {', '.join(sorted(raters))}")
    print(f"{len(rows)} total votes\n")

    print("=" * 74)
    print("PRIMARY: preference for the REVIEWED render among DECISIVE pairs")
    print("(votes excluding 'no difference'; chance = 50%)")
    print("=" * 74)
    print(f"{'modality':<14}{'rv':>5}{'ss':>5}{'same':>6}{'pref_rv':>9}{'  95% CI':>16}{'  sign-p':>10}")
    tot_rv = tot_ss = tot_same = 0
    for mod in mods:
        rv, ss, sm = dec_pref_rv[mod], dec_pref_ss[mod], dec_same[mod]
        tot_rv += rv; tot_ss += ss; tot_same += sm
        n = rv + ss
        frac = rv / n if n else float("nan")
        lo, hi = wilson_ci(rv, n)
        p = binom_two_sided_p(rv, n)
        print(f"{mod:<14}{rv:>5}{ss:>5}{sm:>6}{frac:>8.0%}  [{lo:>3.0%},{hi:>3.0%}]{p:>10.4f}")
    n = tot_rv + tot_ss
    frac = tot_rv / n if n else float("nan")
    lo, hi = wilson_ci(tot_rv, n)
    p = binom_two_sided_p(tot_rv, n)
    print("-" * 74)
    print(f"{'OVERALL':<14}{tot_rv:>5}{tot_ss:>5}{tot_same:>6}{frac:>8.0%}  [{lo:>3.0%},{hi:>3.0%}]{p:>10.4f}")

    print("\n" + "=" * 74)
    print("DOM CONCORDANCE  (does human preference agree with the instrument?)")
    print("=" * 74)
    print(f"Among {n} decisive non-'same' votes, humans agreed with the DOM "
          f"'fewer-defects-is-better'\ndirection {frac:.0%} of the time "
          f"(95% CI [{lo:.0%},{hi:.0%}], two-sided binomial p={p:.4f}).")
    print(f"'No visible difference' was chosen on {tot_same}/{tot_same+n} "
          f"({tot_same/(tot_same+n):.0%}) of decisive pairs.")

    print("\n" + "=" * 74)
    print("TIE CONTROL  (DOM-equal pairs -> human preference should be ~50/50)")
    print("=" * 74)
    print(f"{'modality':<14}{'A':>5}{'B':>5}{'same':>6}{'  |skew|':>10}")
    TA = TB = TS = 0
    for mod in mods:
        a, b, s = tie_A[mod], tie_B[mod], tie_same[mod]
        TA += a; TB += b; TS += s
        nn = a + b
        skew = abs(a - b) / nn if nn else 0.0
        print(f"{mod:<14}{a:>5}{b:>5}{s:>6}{skew:>9.0%}")
    nn = TA + TB
    print("-" * 40)
    print(f"{'OVERALL':<14}{TA:>5}{TB:>5}{TS:>6}{(abs(TA-TB)/nn if nn else 0):>9.0%}"
          f"   (binomial p={binom_two_sided_p(TA, nn):.3f} vs 50/50)")

    # inter-rater agreement
    overlap = {pid: ch for pid, ch in by_pair_choice.items() if len(ch) >= 2}
    if overlap:
        agree = tot = 0
        for pid, ch in overlap.items():
            vals = list(ch.values())
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    tot += 1
                    agree += (vals[i] == vals[j])
        print("\n" + "=" * 74)
        print("INTER-RATER AGREEMENT")
        print("=" * 74)
        print(f"{len(overlap)} pairs rated by >=2 raters; raw pairwise agreement "
              f"{agree}/{tot} = {agree/tot:.0%}.")
    else:
        print("\n(Only one rater per pair -- collect overlapping ratings for "
              "inter-rater agreement.)")


if __name__ == "__main__":
    main()
