import React, { useState } from 'react'
import { BASE, C } from './data.js'
import { Reveal } from './bits.jsx'

const TAB_ORDER = ['figures', 'slides', 'web', 'animations', 'code', 'catalog']

export default function Explorer({ site }) {
  const [tab, setTab] = useState('figures')
  const ex = site.explorer
  return (
    <section className="band" id="explorer">
      <div className="wrap">
        <Reveal>
          <div className="sec-head">
            <div className="no">Part 3 · Look for yourself</div>
            <h2>Every test case. Every model response.</h2>
            <p className="lede">
              This is the paper&rsquo;s raw material: the actual benchmark tasks, the model&rsquo;s
              actual outputs, and — where the loop ran — the full trajectory: draft, instrument
              feedback, and revision. Pick a case to open its dossier.
            </p>
          </div>
        </Reveal>

        <div className="exp-tabs">
          {TAB_ORDER.map(t => {
            const n = t === 'catalog'
              ? site.catalog.reduce((a, s) => a + s.n, 0)
              : ex[t]?.items.length
            const title = t === 'catalog' ? 'All task suites' : ex[t]?.title
            return (
              <button key={t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>
                {title}<span className="n">{n}</span>
              </button>
            )
          })}
        </div>

        {tab === 'catalog' ? <Catalog suites={site.catalog} />
          : tab === 'code' ? <CodeList items={ex.code.items} />
            : <VisualGrid mod={tab} items={ex[tab].items} />}
      </div>
    </section>
  )
}

function VisualGrid({ mod, items }) {
  return (
    <div className="case-grid">
      {items.map(it => {
        const hasGeom = it.ss_defects !== undefined
        const delta = hasGeom ? it.rv_defects - it.ss_defects : null
        return (
          <a key={it.id} className="case-card" href={`#/case/${mod}/${it.id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
            <div className="thumb"><img loading="lazy" src={`${BASE}${it.rv_img}`} alt={it.name} /></div>
            <div className="meta">
              <div className="nm">{it.name}</div>
              <div className="sub">
                {hasGeom && (
                  <span className={`delta-chip ${delta < 0 ? 'better' : delta > 0 ? 'worse' : 'same'}`}>
                    {it.ss_defects} → {it.rv_defects} defects
                  </span>
                )}
                {it.has_trace && <span className="chip external">full trajectory</span>}
              </div>
            </div>
          </a>
        )
      })}
    </div>
  )
}

function CodeList({ items }) {
  const suites = [
    ['dev', 'Development suite — 15 tasks, the 4-strategy scaling experiment ran here'],
    ['deep-spec', 'Deep-spec suite — 11 from-scratch implementations, with full harness trajectories'],
  ]
  return (
    <div>
      {suites.map(([key, blurb]) => (
        <div key={key} style={{ marginBottom: 34 }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 12.5, letterSpacing: '.08em', textTransform: 'uppercase', color: C.ink3, margin: '0 0 12px' }}>{blurb}</div>
          <div className="code-list">
            {items.filter(i => i.suite === key).map(it => (
              <button key={it.id} className="code-row" onClick={() => { window.location.hash = `#/case/code/${it.id}` }}>
                <span className="id">{it.id}</span>
                <span className="ds">{it.desc}</span>
                <span className="marks">
                  {key === 'dev' ? (
                    <>
                      {it.r_passed !== undefined && <span className={`chip ${it.r_passed ? 'pass' : 'fail'}`}>reasoning {it.r_passed ? '✓' : '✗'}</span>}
                      {it.h_passed !== undefined && <span className={`chip ${it.h_passed ? 'pass' : 'fail'}`}>loop {it.h_passed ? '✓' : '✗'}</span>}
                    </>
                  ) : (
                    <>
                      {it.ss_passed !== undefined && <span className={`chip ${it.ss_passed ? 'pass' : 'fail'}`}>single-shot {it.ss_passed ? '✓' : '✗'}</span>}
                      {it.rv_passed !== undefined && <span className={`chip ${it.rv_passed ? 'pass' : 'fail'}`}>loop {it.rv_passed ? '✓' : '✗'}</span>}
                      {it.has_trace && <span className="chip external">trajectory</span>}
                    </>
                  )}
                </span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function Catalog({ suites }) {
  const [open, setOpen] = useState(null)
  return (
    <div>
      <p style={{ maxWidth: 680, color: C.ink2, fontSize: 15, marginTop: 0 }}>
        The complete task inventory behind the paper — including the two modalities where the
        loop is <em>not</em> claimed to help (video editing and deep research, which frontier
        models largely solve in one shot). Expand any suite to read every task specification.
      </p>
      {suites.map((s, i) => (
        <div className="suite" key={i}>
          <button onClick={() => setOpen(open === i ? null : i)} aria-expanded={open === i}>
            <span className="s-title">{s.suite}</span>
            <span className="s-blurb">{s.blurb}</span>
            <span className="s-n">{s.n} tasks {open === i ? '▾' : '▸'}</span>
          </button>
          {open === i && (
            <div className="s-body">
              {s.tasks.map(t => (
                <details key={t.id}>
                  <summary>
                    <code style={{ fontSize: 12, color: C.ink3, marginRight: 8 }}>{t.id}</code>
                    {t.name}
                    {t.difficulty && <span className="chip plain" style={{ marginLeft: 8 }}>{t.difficulty}</span>}
                  </summary>
                  {t.description && <div className="dt">{t.description}</div>}
                  {Array.isArray(t.requirements) && t.requirements.length > 0 && (
                    <ul>{t.requirements.map((r, j) => <li key={j}>{typeof r === 'string' ? r : JSON.stringify(r)}</li>)}</ul>
                  )}
                </details>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
