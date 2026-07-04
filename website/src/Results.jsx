import React from 'react'
import { C } from './data.js'
import { Reveal, FigureCard } from './bits.jsx'
import {
  ScalingChart, Dumbbell, HBars, PairColumns, AllocationChart, VarianceChart, DataTable,
} from './charts.jsx'

export default function Results({ site }) {
  const scal = site.scaling
  const geom = site.geom_reduction
  const abl = site.ablation.rows
  const tc = site.typecontrol.arms
  const xm = site.crossmodel
  const blind = site.blindness
  const dist = site.distill

  return (
    <section className="band tint" id="results">
      <div className="wrap">
        <Reveal>
          <div className="sec-head">
            <div className="no">Part 2 · The evidence</div>
            <h2>Eight results, all from real runs</h2>
            <p className="lede">
              Every number below comes from the paper&rsquo;s experiments — frozen frontier models,
              matched token budgets, three independent seeds. Hover any mark for the underlying values;
              most figures have a table view.
            </p>
          </div>
        </Reveal>

        <div className="results-grid">
          {/* -------- 1. scaling curves -------- */}
          <Reveal className="span2" style={{ gridColumn: '1 / -1' }}>
            <FigureCard no="1" title="Internal scaling stops. Interaction keeps going."
              sub="Hard coding tasks, identical token budgets. The two internal strategies (orange) flatten — even best-of-N with a perfect oracle judge. The two interaction strategies (blue) climb past the ceiling; the proposer–reviewer harness reaches 100% in all three seeds."
              note="15 tasks × 3 seeds (reasoning-only: 1 seed); faint dots show individual seeds. The shaded region above 86.7% is reachable in this experiment only through interaction."
              table={<DataTable
                cols={['strategy', 'type', '1K', '5K', '20K']}
                rows={scal.series.map(s => [s.label, s.type, ...s.means.map(v => v + '%')])} />}>
              <ScalingChart data={scal} />
            </FigureCard>
          </Reveal>

          {/* -------- 2. blindness -------- */}
          <Reveal>
            <FigureCard no="2" title="The standard judge misses almost everything"
              sub="15 dense figures, one attempt each. The usual screenshot-reading AI judge calls 14 of 15 “perfect.” A deterministic instrument that measures the real layout finds only 3 actually clean."
              note="The failure is mechanical: content that overflows the canvas is cropped out of the screenshot before the judge sees it. No prompt can fix what the channel deletes.">
              <BlindDots blind={blind} />
            </FigureCard>
          </Reveal>

          {/* -------- 3. reviewer swap -------- */}
          <Reveal>
            <FigureCard no="3" title="Swap one ingredient: the critic"
              sub="Same tasks, same model, same number of review passes — only the reviewer's instrument changes. An AI critic reading screenshots makes layouts worse; the measuring critic fixes them."
              note={`Mean layout defects per artifact, before → after one review pass. Geometric reviewer: slides −73% (p=0.0018), figures −74% (p=7×10⁻⁴). VLM reviewer: both suites got worse.`}
              table={<DataTable cols={['suite', 'reviewer', 'before', 'after']}
                rows={abl.map(r => [r.suite, r.arm, r.ss, r.rv])} />}>
              <Dumbbell max={2.6} rows={abl.map(r => ({
                label: r.arm.replace(' reviewer', ''),
                sub: r.suite, a: r.ss, b: r.rv, tip: r.sig,
              }))} labelWidth={120} />
            </FigureCard>
          </Reveal>

          {/* -------- 4. geometric reduction -------- */}
          <Reveal className="span2" style={{ gridColumn: '1 / -1' }}>
            <FigureCard no="4" title="Measured with a real instrument, the loop works on every visual modality"
              sub="Defect reduction from the grounded loop, per modality — the same configuration everywhere, three seeds, 95% confidence intervals from a paired bootstrap. Every reduction is statistically decisive."
              note="A (task, seed) pair counts as improved/regressed if its defect count moved. Ratios at right: improved / regressed pairs. All sign tests p < 2×10⁻³."
              table={<DataTable cols={['modality', 'defects before', 'after', 'reduction', 'improved/regressed', 'pairs']}
                rows={geom.map(r => [r.label, r.ss_mean, r.rv_mean, r.reduction_pct + '%', `${r.improved} / ${r.regressed}`, r.pairs])} />}>
              <HBars max={100} unit="%" W={720}
                rows={geom.map(r => ({
                  label: r.label,
                  sub: `${r.ss_mean} → ${r.rv_mean} defects`,
                  v: Math.abs(r.reduction_pct),
                  ci: [Math.abs(r.ci[1]) < Math.abs(r.ci[0]) ? Math.abs(r.ci[1]) : Math.abs(r.ci[0]), Math.min(100, Math.max(Math.abs(r.ci[0]), Math.abs(r.ci[1])))],
                  vlabel: `−${Math.abs(r.reduction_pct)}%`,
                  tipv: `−${Math.abs(r.reduction_pct)}% defects`,
                  tip: `${r.improved} improved / ${r.regressed} regressed · p=${r.p}`,
                }))} />
            </FigureCard>
          </Reveal>

          {/* -------- 5. feedback coverage -------- */}
          <Reveal>
            <FigureCard no="5" title="Grounding helps exactly as far as the instrument can see"
              sub="Three feedback channels on the same coding tasks. A linter is a real instrument — but it can't observe runtime bugs, so it buys nothing over opinion. Execution can, and does."
              note="One review pass per arm. The execution arm also converged ~2.5× cheaper (2.6K vs 7.1K tokens). This is the paper's “coverage principle” in one chart."
              table={<DataTable cols={['feedback', 'pass rate', 'tokens to converge']}
                rows={tc.map(r => [r.label, r.pass + '%', (r.tokens / 1000).toFixed(1) + 'K'])} />}>
              <HBars max={100} labelWidth={185} W={560}
                color={r => r.grounded2 ? C.gnd : C.ung}
                rows={tc.map((r, i) => ({
                  label: r.label, v: r.pass, grounded2: i === 2,
                  sub: i === 2 ? 'sees the failing behavior' : i === 1 ? 'sees only code form' : 'model opinion only',
                  tip: `${(r.tokens / 1000).toFixed(1)}K tokens to converge`,
                }))} />
            </FigureCard>
          </Reveal>

          {/* -------- 6. cross model -------- */}
          <Reveal>
            <FigureCard no="6" title="Not a quirk of one model"
              sub="Same harness, three different model families. Every family's reviewed ceiling is 93–100% — with zero variance across seeds."
              note="Whiskers span the three single-shot seeds. On the visual side the result also transfers: the same instrument removes 73% of Sonnet's slide defects and 93% of Gemini 3.1 Pro's."
              table={<DataTable cols={['model', 'single-shot', 'with loop']}
                rows={xm.map(r => [r.model, r.ss + '%', r.rv + '%'])} />}>
              <PairColumns rows={xm} />
            </FigureCard>
          </Reveal>

          {/* -------- 7. allocation -------- */}
          <Reveal>
            <FigureCard no="7" title="Where should the budget go? To the writer."
              sub="A fixed 10K-token budget split between proposing, executing, and reviewing, nine ways. Pass rate is monotone in the proposer's share — an 86.6-point spread from worst to best split."
              note="Reviewing and executing are cheap; drafting is where tokens turn into working artifacts. Practical rule: give the proposer the majority of the budget."
              table={<DataTable cols={['split (propose/execute/review)', 'pass rate']}
                rows={[...site.allocation.rows].sort((a, b) => a.propose - b.propose).map(r =>
                  [`${Math.round(r.propose * 100)} / ${Math.round(r.execute * 100)} / ${Math.round(r.review * 100)}`, r.pass_rate + '%'])} />}>
              <AllocationChart data={site.allocation} />
            </FigureCard>
          </Reveal>

          {/* -------- 8. distillation -------- */}
          <Reveal>
            <FigureCard no="8" title="The habit can be taught — carefully"
              sub="Distilling the harness's behavior into an 8-billion-parameter student keeps about half its capability at 10× lower cost. But post-training that crushes output variety (RFT) flattens exactly the curve that sampling needs."
              note={`Student on held-out hard tasks: ${dist.hard.pass1}% first try, ${dist.hard.pass3}% within three tries. Variance is a budget: the SFT student keeps converting extra samples into solved tasks; the RFT student stops.`}
              table={<DataTable cols={['student', 'pass@1', 'pass@2', 'pass@3']}
                rows={[['SFT', ...dist.variance.sft.map(v => v + '%')], ['SFT + RFT', ...dist.variance.rft.map(v => v + '%')]]} />}>
              <VarianceChart data={dist.variance} />
            </FigureCard>
          </Reveal>
        </div>
      </div>
    </section>
  )
}

function BlindDots({ blind }) {
  const rows = [
    { label: 'AI judge (screenshot)', clean: blind.vlm_perfect, col: C.ung, cap: `${blind.vlm_perfect} of ${blind.n} rated “perfect”` },
    { label: 'Measuring instrument', clean: blind.geom_clean, col: C.gnd, cap: `${blind.geom_clean} of ${blind.n} actually clean` },
  ]
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22, padding: '6px 0 2px' }}>
      {rows.map(r => (
        <div key={r.label}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 9 }}>
            <span style={{ fontSize: 13.5, fontWeight: 600 }}>{r.label}</span>
            <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 12.5, color: C.ink2 }}>{r.cap}</span>
          </div>
          <div className="dots15">
            {Array.from({ length: blind.n }, (_, i) => (
              <span key={i} className="d" title={i < r.clean ? 'judged clean' : 'judged defective'}
                style={{
                  background: i < r.clean ? r.col : 'transparent',
                  border: `2px solid ${i < r.clean ? r.col : C.ink3}`,
                  opacity: i < r.clean ? 1 : 0.45,
                }} />
            ))}
          </div>
        </div>
      ))}
      <div style={{ fontSize: 13, color: C.ink3, borderTop: `1px solid var(--hairline)`, paddingTop: 12 }}>
        Same 15 artifacts, two scoreboards. Eleven broken figures are invisible to the screenshot judge.
      </div>
    </div>
  )
}
