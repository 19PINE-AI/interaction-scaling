import React, { useEffect } from 'react'
import { BASE, C, useCase, fmtTok } from './data.js'
import { Compare, RawOutput } from './bits.jsx'

const MOD_TITLES = { figures: 'Academic figure', slides: 'Dense slide', web: 'Web page', animations: 'Animation', code: 'Code task' }

export default function CaseView({ mod, id }) {
  const { data, err } = useCase(mod, id)
  useEffect(() => { window.scrollTo(0, 0) }, [mod, id])
  const close = () => { window.location.hash = '#explorer' }
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="caseview">
      <div className="cv-top">
        <div className="inner">
          <button className="backbtn" onClick={close}>← all cases</button>
          <h2>{MOD_TITLES[mod]} · {data ? data.name : id}</h2>
          {data?.trace && <span className="chip external">full trajectory</span>}
        </div>
      </div>
      <div className="inner-body">
        {err && <div className="loading">Could not load this case.</div>}
        {!data && !err && <div className="loading">loading case dossier…</div>}
        {data && (mod === 'code' ? <CodeCase d={data} /> : <VisualCase d={data} />)}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------ visual case */
function VisualCase({ d }) {
  const runs = d.runs || []
  return (
    <div>
      <TaskPanel task={d.task} />
      <div className="panel">
        <h3>Single-shot vs. after the grounded loop</h3>
        <Compare ssImg={`${BASE}${d.ss_img}`} rvImg={`${BASE}${d.rv_img}`} />
      </div>

      {runs.length > 0 && (
        <div className="panel">
          <h3>What the instrument measured — {runs.length} independent seeds</h3>
          <DefectBreakdown runs={runs} />
        </div>
      )}

      {d.trace && <Trajectory d={d} />}
    </div>
  )
}

function DefectBreakdown({ runs }) {
  const kinds = [
    ['text_overlap', 'text-on-text overlap'],
    ['clipped', 'clipped / cut off'],
    ['overflow', 'container overflow'],
  ]
  const avg = (pref, k) => runs.reduce((a, r) => a + (r[`${pref}_${k}`] ?? 0), 0) / runs.length
  const tot = pref => runs.reduce((a, r) => a + (r[`${pref}_n_defects`] ?? 0), 0) / runs.length
  const max = Math.max(tot('ss'), tot('rv'), 1)
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 26 }}>
      {['ss', 'rv'].map(pref => (
        <div key={pref}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
            <b style={{ fontSize: 14 }}>{pref === 'ss' ? 'single-shot' : 'after the loop'}</b>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 13, color: pref === 'ss' ? '#A34413' : C.gndDk, fontWeight: 600 }}>
              {tot(pref).toFixed(1)} defects avg
            </span>
          </div>
          {kinds.map(([k, lab]) => {
            const v = avg(pref, k)
            return (
              <div className="defbar" key={k}>
                <span className="lab">{lab}</span>
                <span className="track">
                  <span className="bar" style={{
                    width: `${Math.min(100, v / max * 100)}%`,
                    background: pref === 'ss' ? C.ung : C.gnd,
                  }} />
                </span>
                <span className="num">{v.toFixed(1)}</span>
              </div>
            )
          })}
          <div style={{ fontSize: 12, color: C.ink3, fontFamily: 'var(--mono)', marginTop: 8 }}>
            {pref === 'rv' && `${(runs.reduce((a, r) => a + (r.rv_iterations ?? 1), 0) / runs.length).toFixed(1)} loop iterations · `}
            {fmtTok(Math.round(runs.reduce((a, r) => a + (r[`${pref}_total_tokens`] ?? 0), 0) / runs.length))} tokens avg
          </div>
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------- the trajectory */
function Trajectory({ d }) {
  const steps = d.trace || []
  return (
    <div className="panel">
      <h3>The trajectory — real transcript{d.trace_model ? ` · ${d.trace_model}` : ''}</h3>
      <p style={{ fontSize: 14, color: C.ink2, margin: '0 0 20px', maxWidth: 700 }}>
        This is the actual conversation that produced the “after” artifact: the model&rsquo;s first
        draft, the reviewer&rsquo;s feedback on what the instrument observed, and each revision.
        Grey italic passages are the model thinking out loud.
      </p>
      <div className="trace">
        {steps.map((s, i) => <TraceStep key={i} step={s} idx={i} />)}
        <div className="tstep feedback" style={{ marginBottom: 0 }}>
          <div className="t-title"><b>Loop ends</b>
            {d.rv_quality !== undefined && <span className="chip plain">final judge score {typeof d.rv_quality === 'number' ? d.rv_quality.toFixed(2) : String(d.rv_quality)}</span>}
          </div>
          <div className="t-body">The best-scoring iteration is kept as the final artifact.</div>
        </div>
      </div>
    </div>
  )
}

function TraceStep({ step, idx }) {
  const kind = step.step === 'generate' ? 'generate' : 'revise'
  return (
    <>
      {step.feedback && (
        <div className="tstep feedback">
          <div className="t-title">
            <b>Instrument feedback</b>
            <span className="chip external">grounded observation</span>
          </div>
          <div className="t-body"><pre className="fb">{step.feedback}</pre></div>
        </div>
      )}
      <div className={`tstep ${kind}`}>
        <div className="t-title">
          <b>{step.step === 'generate' ? `Draft ${idx === 0 ? '1 — first attempt' : idx + 1}` : `Revision ${idx}`}</b>
          <span className="chip plain">{step.step}</span>
        </div>
        <div className="t-body">
          <RawOutput text={step.output} label="show the model's full response" open={false} />
        </div>
      </div>
    </>
  )
}

/* -------------------------------------------------------------- code case */
const STRAT_LABELS = {
  R: ['Reasoning-only', 'internal'], S: ['Best-of-N (oracle)', 'internal'],
  L: ['Single-agent loop', 'external'], H: ['Proposer–reviewer', 'external'],
}

function CodeCase({ d }) {
  return (
    <div>
      <TaskPanel task={d.task} code />
      {d.suite === 'dev' && d.strategies && <StrategyMatrix d={d} />}
      {d.cross_model && Object.keys(d.cross_model).length > 0 && <CrossModelPanel xm={d.cross_model} />}
      {d.trace && <Trajectory d={d} />}
      {(d.ss_code || d.rv_code) && (
        <div className="panel">
          <h3>Final code — single-shot vs. after the loop</h3>
          <div style={{ display: 'grid', gap: 18 }}>
            {d.ss_code && (
              <div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                  <b style={{ fontSize: 14 }}>single-shot</b>
                  {d.ss_passed !== undefined && <span className={`chip ${d.ss_passed ? 'pass' : 'fail'}`}>{d.ss_passed ? 'tests pass' : 'tests fail'}</span>}
                </div>
                <pre className="raw">{d.ss_code}</pre>
              </div>
            )}
            {d.rv_code && (
              <div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                  <b style={{ fontSize: 14 }}>after the loop</b>
                  {d.rv_passed !== undefined && <span className={`chip ${d.rv_passed ? 'pass' : 'fail'}`}>{d.rv_passed ? 'tests pass' : 'tests fail'}</span>}
                </div>
                <pre className="raw">{d.rv_code}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function StrategyMatrix({ d }) {
  const budgets = ['1000', '5000', '20000']
  return (
    <div className="panel">
      <h3>Did it pass? — four strategies × three budgets (seed 1)</h3>
      <div style={{ overflowX: 'auto' }}>
        <table className="stratgrid">
          <thead>
            <tr><th style={{ textAlign: 'left' }}>strategy</th>{budgets.map(b => <th key={b}>{+b / 1000}K tokens</th>)}<th>tokens used @20K</th></tr>
          </thead>
          <tbody>
            {['R', 'S', 'L', 'H'].map(k => {
              const [label, typ] = STRAT_LABELS[k]
              const row = d.strategies[k] || {}
              return (
                <tr key={k}>
                  <td className="rowlab">{label}<span className={`chip ${typ}`}>{typ}</span></td>
                  {budgets.map(b => {
                    const cell = row[b]
                    return (
                      <td key={b}>
                        {cell ? (
                          <span className={cell.passed ? 'cell-pass' : 'cell-fail'} title={cell.error || (cell.passed ? 'passed' : 'failed')}>
                            {cell.passed ? '✓' : '✗'}
                          </span>
                        ) : '—'}
                      </td>
                    )
                  })}
                  <td style={{ fontFamily: 'var(--mono)', fontSize: 12.5, color: C.ink2 }}>
                    {row['20000'] ? fmtTok(row['20000'].tokens) : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {/* harness turn log + final code at 20K */}
      {d.strategies.H?.['20000']?.turn_log && d.strategies.H['20000'].turn_log.length > 1 && (
        <div style={{ marginTop: 20 }}>
          <b style={{ fontSize: 14 }}>Proposer–reviewer turn log (20K budget)</b>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
            {d.strategies.H['20000'].turn_log.map(t => (
              <span key={t.turn} className={`chip ${t.passed ? 'pass' : 'fail'}`}
                title={t.error_message || ''}>
                turn {t.turn + 1}: {t.passed ? 'tests pass ✓' : (t.error_message || 'fail').slice(0, 44)}
              </span>
            ))}
          </div>
        </div>
      )}
      {d.strategies.H?.['20000']?.code && (
        <div style={{ marginTop: 18 }}>
          <RawOutput text={d.strategies.H['20000'].code} label="show the harness's final code (20K budget)" />
        </div>
      )}
      {d.strategies.R?.['20000'] && !d.strategies.R['20000'].passed && d.strategies.R['20000'].code && (
        <div style={{ marginTop: 10 }}>
          <RawOutput text={(d.strategies.R['20000'].error ? `# failed: ${d.strategies.R['20000'].error}\n\n` : '') + d.strategies.R['20000'].code}
            label="show reasoning-only's failed attempt for contrast" />
        </div>
      )}
    </div>
  )
}

function CrossModelPanel({ xm }) {
  return (
    <div className="panel">
      <h3>Other model families on this task</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 18 }}>
        {Object.entries(xm).map(([model, r]) => (
          <div key={model} style={{ border: '1px solid var(--hairline)', borderRadius: 8, padding: 16 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
              <b style={{ fontSize: 14.5 }}>{model}</b>
              <span className={`chip ${r.ss_passed ? 'pass' : 'fail'}`}>single-shot {r.ss_passed ? '✓' : '✗'}</span>
              <span className={`chip ${r.rv_passed ? 'pass' : 'fail'}`}>loop {r.rv_passed ? '✓' : '✗'}</span>
              <span className="chip plain">{r.rv_iterations} iter</span>
            </div>
            <RawOutput text={r.rv_code} label="final code after the loop" />
          </div>
        ))}
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- shared */
function TaskPanel({ task, code }) {
  if (!task) return null
  return (
    <div className="panel">
      <h3>The task, as given to the model</h3>
      {task.paper && <div style={{ fontSize: 13.5, color: C.ink3, marginBottom: 6 }}>source paper: {task.paper}</div>}
      {task.description && <p style={{ margin: 0, fontSize: 15.5, whiteSpace: 'pre-wrap' }}>{task.description}</p>}
      {Array.isArray(task.requirements) && task.requirements.length > 0 && (
        <ul className="req">
          {task.requirements.map((r, i) => <li key={i}>{typeof r === 'string' ? r : JSON.stringify(r)}</li>)}
        </ul>
      )}
      {task.difficulty && <div style={{ marginTop: 12 }}><span className="chip plain">difficulty: {task.difficulty}</span></div>}
      {code && task.test_code && (
        <div style={{ marginTop: 14 }}>
          <RawOutput text={task.test_code} label="show the hidden test suite (the instrument)" />
        </div>
      )}
    </div>
  )
}
