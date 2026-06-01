"""Bootstrap 95% CIs for the geometric (and code) interaction-scaling results.

Paired bootstrap over task-run pairs (resample with replacement), 10k resamples,
percentile CI on the mean reviewed-minus-single-shot change and the % defect
reduction. For code, CI on the pass-rate delta. Deterministic via fixed seed.
"""
import json, os, random, statistics
random.seed(20260601)
R = "results/hard_benchmarks/"

def load_geom(files):
    ss, rv = [], []
    for f in files:
        p = R + f
        if not os.path.exists(p): continue
        for r in json.load(open(p)):
            if r.get("ss_n_defects") is not None and r.get("rv_n_defects") is not None:
                ss.append(int(r["ss_n_defects"])); rv.append(int(r["rv_n_defects"]))
    return ss, rv

def boot_ci(pairs, fn, B=10000):
    n = len(pairs); vals = []
    for _ in range(B):
        sample = [pairs[random.randrange(n)] for _ in range(n)]
        vals.append(fn(sample))
    vals.sort()
    return vals[int(0.025*B)], vals[int(0.975*B)]

def report(name, ss, rv):
    pairs = list(zip(ss, rv)); n = len(pairs)
    mss, mrv = statistics.mean(ss), statistics.mean(rv)
    delta = mrv - mss
    redu = 100*(1 - sum(rv)/max(sum(ss),1))
    d_lo, d_hi = boot_ci(pairs, lambda s: statistics.mean([b-a for a,b in s]))
    r_lo, r_hi = boot_ci(pairs, lambda s: 100*(1 - sum(b for _,b in s)/max(sum(a for a,_ in s),1)))
    print(f"{name}: n={n} meanSS={mss:.2f} meanRV={mrv:.2f} | mean delta={delta:+.2f} [95% CI {d_lo:+.2f},{d_hi:+.2f}] | reduction={redu:.0f}% [95% CI {r_lo:.0f}%,{r_hi:.0f}%]")

print("=== GEOMETRIC defect reduction, paired bootstrap 95% CI (n=10k) ===")
report("Academic figures (20)", *load_geom([f"paperfig_geom_hard_run{i}.json" for i in (1,2,3)]))
report("Dense slides    (12)", *load_geom([f"slides_hard2_geom_run{i}.json" for i in (1,2,3)]))
report("Web pages       (20)", *load_geom([f"web_hard_geom_run{i}.json" for i in (1,2,3)]+[f"web_hard_geom_b_run{i}.json" for i in (1,2,3)]))
report("Animations      (20)", *load_geom([f"anim_clean_geom_run{i}.json" for i in (1,2,3)]))

# code pass-rate delta CI
ss=[];rv=[]
for i in (1,2,3):
    p=R+f"code_hard_onpolicy_run{i}.json"
    if not os.path.exists(p): continue
    for r in json.load(open(p)):
        ss.append(1 if str(r.get("ss_meets"))=="True" else 0); rv.append(1 if str(r.get("rv_meets"))=="True" else 0)
pairs=list(zip(ss,rv)); n=len(pairs)
lo,hi=boot_ci(pairs, lambda s: 100*(statistics.mean([b for _,b in s])-statistics.mean([a for a,_ in s])))
print(f"\n=== CODE pass-rate ===\nn={n} SS={100*statistics.mean(ss):.1f}% RV={100*statistics.mean(rv):.1f}% | delta=+{100*(statistics.mean(rv)-statistics.mean(ss)):.1f}pp [95% CI {lo:.1f},{hi:.1f}]pp")
