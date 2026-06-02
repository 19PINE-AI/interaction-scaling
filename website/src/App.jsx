import React, { useEffect, useMemo, useRef, useState } from 'react'

const BASE = import.meta.env.BASE_URL

function useSite() {
  const [d, setD] = useState(null)
  useEffect(() => {
    fetch(`${BASE}data/site.json`).then(r => r.json()).then(setD).catch(() => setD({ error: true }))
  }, [])
  return d
}

const NAV = [
  ['overview', 'Overview'], ['geometry', 'Grounded evaluation'], ['ablation', 'Ablation'],
  ['code', 'Code & cross-model'], ['scaling', 'Scaling'], ['galleries', 'Galleries'],
  ['prompts', 'Prompts'], ['limits', 'Limits'],
]

function Stat({ big, lbl, sig, color }) {
  return <div className="card stat">
    <div className="big" style={{ color }}>{big}</div>
    <div className="lbl">{lbl}</div>
    {sig && <div className="sig">{sig}</div>}
  </div>
}

// horizontal bar (0..100) with optional 95% CI whiskers. `value` keeps its sign
// (negative = a reduction); bar width and CI use magnitude.
function Bar({ name, value, ci, color = 'var(--good)', suffix = '%' }) {
  const label = (value < 0 ? '−' : '') + Math.abs(value) + suffix
  return <div className="barrow">
    <div className="nm">{name}</div>
    <div className="bar">
      <span style={{ width: `${Math.min(100, Math.abs(value))}%`, background: color, opacity: .85 }} />
      {ci && <div className="ci" style={{ left: `${ci[0]}%`, width: `${ci[1] - ci[0]}%` }} />}
    </div>
    <div className="vv">{label}</div>
  </div>
}

function Cmp({ ss, rv }) {
  const [x, setX] = useState(50)
  return <div className="cmp" style={{ '--x': `${x}%` }}>
    <img src={`${BASE}${ss}`} alt="single-shot" />
    <img className="rev" src={`${BASE}${rv}`} alt="reviewed" />
    <div className="handle" />
    <span className="tag l">single-shot</span>
    <span className="tag r">reviewed (grounded)</span>
    <input type="range" min="0" max="100" value={x} onChange={e => setX(+e.target.value)} aria-label="reveal" />
  </div>
}

function Gallery({ galleries }) {
  const cats = Object.keys(galleries)
  const [cat, setCat] = useState(cats[0])
  const items = galleries[cat] || []
  return <>
    <div className="tabs">
      {cats.map(c => <button key={c} className={c === cat ? 'on' : ''} onClick={() => setCat(c)}>
        {c} ({galleries[c].length})
      </button>)}
    </div>
    <p className="note">Drag the divider: <b>left = single-shot</b>, <b>right = reviewed</b> under grounded geometric feedback.
      Numbers are exact DOM-geometry defect counts (lower is better).</p>
    <div className="gal">
      {items.map(it => {
        const better = it.rv_defects < it.ss_defects
        return <div className="shot" key={it.task_id}>
          <div className="ttl">
            <b>{it.name || it.task_id}</b>
            <span>
              <span className="pill r">{it.ss_defects}</span>{' → '}
              <span className={`pill ${better ? 'g' : 'r'}`}>{it.rv_defects}</span>
              {it.iters ? <span className="muted" style={{ fontSize: 12, marginLeft: 8 }}>{it.iters} it.</span> : null}
            </span>
          </div>
          <Cmp ss={it.ss_image} rv={it.rv_image} />
        </div>
      })}
    </div>
  </>
}

export default function App() {
  const d = useSite()
  if (!d) return <div className="wrap" style={{ padding: 60 }}>Loading…</div>
  if (d.error) return <div className="wrap" style={{ padding: 60 }}>Could not load data.</div>

  const geomMax = 100
  const promptGroups = ['generation', 'feedback', 'evaluation']
  const groupName = { generation: 'Generation prompts', feedback: 'Feedback / reviewer', evaluation: 'Evaluation / judge & instruments' }

  return <>
    <header className="nav"><div className="inner">
      <span className="brand">Grounding the Loop</span>
      {NAV.map(([id, label]) => <a key={id} href={`#${id}`}>{label}</a>)}
    </div></header>

    <div className="hero"><div className="wrap">
      <div className="eyebrow">Interaction Scaling · evaluation results</div>
      <h1>{d.title}</h1>
      <p className="sub">{d.subtitle}</p>
      <p className="lead" style={{ marginTop: 18 }}>{d.thesis}</p>
    </div></div>

    <div className="wrap">

      {/* OVERVIEW */}
      <section id="overview">
        <div className="grid g4">
          <Stat big="−40 to −74%" lbl="layout defects removed by grounded feedback (4 visual modalities)" sig="all p < 0.002 · CIs exclude 0" />
          <Stat big="66.7→100%" lbl="code pass-rate, harness vs single-shot" sig="+33.3pp · p = 0.008" color="var(--accent)" />
          <Stat big="14 vs 3" lbl="figures the VLM judge calls 'perfect' vs actually clean (of 15)" sig="the judge is blind" color="var(--bad)" />
          <Stat big="73% / 87%" lbl="where reasoning / best-of-N saturate; interaction → 100%" sig="DPI ceiling" color="var(--mut)" />
        </div>
        <p className="note" style={{ marginTop: 16 }}>
          This site reports the full evaluation behind the paper: every benchmark, the single-shot (baseline) vs reviewed
          (grounded) comparison for each, the exact generation/feedback/judge prompts, and per-task before/after renders.
        </p>
      </section>

      {/* GROUNDED EVALUATION */}
      <section id="geometry">
        <h2>Grounded evaluation across four visual modalities</h2>
        <p className="note">{d.geometric.note}</p>
        <div style={{ maxWidth: 640, margin: '8px 0 14px' }}>
          {d.geometric.rows.map(r => <Bar key={r.modality} name={r.modality} value={r.delta_pct} ci={r.ci} />)}
        </div>
        <table><thead><tr>
          <th>Modality</th><th className="num">n</th><th className="num">single-shot</th>
          <th className="num">reviewed</th><th className="num">Δ</th><th className="num">95% CI</th>
          <th className="num">impr/regr</th><th className="num">p</th>
        </tr></thead><tbody>
          {d.geometric.rows.map(r => <tr key={r.modality}>
            <td>{r.modality}</td><td className="num">{r.n}</td><td className="num bad">{r.ss}</td>
            <td className="num good">{r.rv}</td><td className="num good">{r.delta_pct}%</td>
            <td className="num">[{r.ci[0]},{r.ci[1]}]%</td><td className="num">{r.impr}/{r.regr}</td>
            <td className="num">{r.p}</td>
          </tr>)}
        </tbody></table>
        <div className="card" style={{ marginTop: 16, borderColor: 'var(--bad)' }}>
          <h3 style={{ marginTop: 0, color: 'var(--bad)' }}>Why a VLM judge can't see this</h3>
          <p className="muted" style={{ margin: 0 }}>{d.blindness.note}</p>
          <div className="grid g2" style={{ marginTop: 12 }}>
            <Bar name="VLM judge: 'perfect'" value={Math.round(100 * d.blindness.vlm_perfect / d.blindness.n)} color="var(--bad)" suffix="%" />
            <Bar name="DOM geometry: clean" value={Math.round(100 * d.blindness.geom_clean / d.blindness.n)} color="var(--good)" suffix="%" />
          </div>
        </div>
        <h3>Cross-model: grounded geometry replicates off Claude</h3>
        <p className="note">{d.crossmodel_geometry.note}</p>
        <table><thead><tr><th>Model</th><th>Modality</th><th className="num">SS</th><th className="num">Rev.</th><th className="num">Δ</th><th className="num">impr/regr</th><th className="num">p</th></tr></thead>
          <tbody><tr>
            <td className="accent">{d.crossmodel_geometry.model}</td><td>{d.crossmodel_geometry.modality}</td>
            <td className="num bad">{d.crossmodel_geometry.ss}</td><td className="num good">{d.crossmodel_geometry.rv}</td>
            <td className="num good">{d.crossmodel_geometry.delta_pct}%</td>
            <td className="num">{d.crossmodel_geometry.impr}/{d.crossmodel_geometry.regr}</td><td className="num">{d.crossmodel_geometry.p}</td>
          </tr></tbody></table>
      </section>

      {/* ABLATION */}
      <section id="ablation">
        <h2>Grounding the feedback side too (controlled ablation)</h2>
        <p className="note">{d.ablation.note}</p>
        <table><thead><tr>
          <th>Modality</th><th className="num">VLM-feedback SS→Rev.</th><th className="num">geometric-feedback SS→Rev.</th>
        </tr></thead><tbody>
          {d.ablation.rows.map(r => <tr key={r.modality}>
            <td>{r.modality}</td>
            <td className="num"><span className="bad">{r.vlm_ss} → {r.vlm_rv}</span> {r.vlm_rv > r.vlm_ss ? '↑ worse' : ''}</td>
            <td className="num"><span className="good">{r.geom_ss} → {r.geom_rv}</span> ↓ better</td>
          </tr>)}
        </tbody></table>
      </section>

      {/* CODE + CROSS-MODEL */}
      <section id="code">
        <h2>Execution-grounded code</h2>
        <table><thead><tr><th>Suite</th><th className="num">single-shot</th><th className="num">reviewed</th><th className="num">Δ</th><th className="num">p</th></tr></thead>
          <tbody>{d.code.rows.map(r => <tr key={r.suite}>
            <td>{r.suite}</td><td className="num bad">{r.ss}</td><td className="num good">{r.rv}</td><td className="num good">{r.delta}</td><td className="num">{r.p}</td>
          </tr>)}</tbody></table>
        <h3>Replicates across model families</h3>
        <table><thead><tr><th>Model</th><th className="num">single-shot</th><th className="num">reviewed</th><th className="num">Δ</th></tr></thead>
          <tbody>{d.crossmodel_code.map(r => <tr key={r.model}>
            <td>{r.model}</td><td className="num bad">{r.ss}</td><td className="num good">{r.rv}</td><td className="num good">{r.delta}</td>
          </tr>)}</tbody></table>
      </section>

      {/* SCALING */}
      <section id="scaling">
        <h2>Two axes saturate; the third does not</h2>
        <p className="note">{d.scaling.note} The first two strategies are bounded by the data-processing inequality; only grounded interaction breaks the ceiling.</p>
        <div style={{ maxWidth: 640 }}>
          {d.scaling.rows.map(r => <Bar key={r.strategy} name={r.strategy} value={r.pass}
            color={r.grounded ? 'var(--good)' : 'var(--bad)'} suffix="%" />)}
        </div>
        <div className="grid g2" style={{ marginTop: 14 }}>
          <div className="card"><h3 style={{ marginTop: 0 }}>Budget allocation</h3>
            <p className="muted" style={{ margin: 0 }}>{d.allocation.note} <b className="accent">{d.allocation.spread_pp} pp</b> spread.</p></div>
          <div className="card"><h3 style={{ marginTop: 0 }}>Distillation into an 8B student</h3>
            <p className="muted" style={{ marginBottom: 8 }}>{d.distill.note}</p>
            <table><tbody>{d.distill.rows.map(r => <tr key={r.metric}>
              <td>{r.metric}</td><td className="num accent">{r.value}</td></tr>)}</tbody></table></div>
        </div>
      </section>

      {/* GALLERIES */}
      <section id="galleries">
        <h2>Before / after: per-task renders</h2>
        <Gallery galleries={d.galleries} />
      </section>

      {/* PROMPTS */}
      <section id="prompts">
        <h2>Prompts &amp; evaluation instruments</h2>
        <p className="note">The exact generation, feedback, and judge prompts, and the deterministic instruments used to both
          drive revision and score the artifacts.</p>
        {promptGroups.map(g => <div key={g}>
          <h3>{groupName[g]}</h3>
          {d.prompts.filter(p => p.group === g).map(p => <details className="acc" key={p.id}>
            <summary><span>{p.title}</span><span className="grp">{g}</span></summary>
            <pre>{p.text}</pre>
          </details>)}
        </div>)}
      </section>

      {/* LIMITS + baseline-vs-updated */}
      <section id="limits">
        <h2>Baseline → updated, and honest limits</h2>
        <h3>What the measurement &amp; prompt changes moved</h3>
        <p className="note">{d.baseline_vs_updated.note}</p>
        <table><thead><tr><th>Result</th><th>Baseline</th><th>Updated</th></tr></thead>
          <tbody>{d.baseline_vs_updated.rows.map((r, i) => <tr key={i}>
            <td>{r.item}</td><td className="muted">{r.baseline}</td><td className="good">{r.updated}</td></tr>)}</tbody></table>
        <h3>Modalities that saturate (reported honestly)</h3>
        <table><thead><tr><th>Modality</th><th>Result</th></tr></thead>
          <tbody>{d.saturated.map(r => <tr key={r.modality}><td>{r.modality}</td><td className="muted">{r.result}</td></tr>)}</tbody></table>
      </section>

      <div className="foot"><div>
        Companion site for <i>{d.title}</i>. All numbers are computed by the deterministic instruments and harnesses in the
        repository; before/after renders are the actual single-shot vs reviewed artifacts. {geomMax && ''}
      </div></div>
    </div>
  </>
}
