// Hand-rolled SVG charts following the dataviz mark specs:
// 2px lines, >=8px ring-carrying markers, hairline solid grid, bars <=24px with
// 4px rounded data-ends, legends for >=2 series, selective direct labels,
// tooltips on a generous hit layer, and a table-view twin for every chart.
import React, { useRef, useState } from 'react'
import { C } from './data.js'

function useTip() {
  const [tip, setTip] = useState(null)
  const ref = useRef(null)
  const show = (evt, html) => {
    const box = ref.current?.getBoundingClientRect()
    if (!box) return
    setTip({ x: evt.clientX - box.left, y: evt.clientY - box.top, html })
  }
  const hide = () => setTip(null)
  const Tip = tip ? (
    <div className="tip" style={{ left: tip.x, top: tip.y }}
      dangerouslySetInnerHTML={{ __html: tip.html }} />
  ) : null
  return { ref, show, hide, Tip }
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map(it => (
        <span className="key" key={it.label}>
          <span className={`sw ${it.dot ? 'dot' : ''}`} style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  )
}

/* ---- scaling curves: pass rate vs token budget, log-x, seed dots ---- */
export function ScalingChart({ data }) {
  const { ref, show, hide, Tip } = useTip()
  const W = 720, H = 340, m = { l: 52, r: 190, t: 18, b: 46 }
  const iw = W - m.l - m.r, ih = H - m.t - m.b
  const Y0 = 40, Y1 = 102
  const xs = v => m.l + (Math.log10(v) - 3) / (Math.log10(20000) - 3) * iw
  const ys = v => m.t + (Y1 - v) / (Y1 - Y0) * ih
  const colors = { H: C.gnd, L: C.gndDk, S: C.ung, R: C.ungDk }
  const ceil = data.ceiling

  return (
    <div className="chart" ref={ref}>
      <Legend items={data.series.map(s => ({ label: s.label, color: colors[s.key] }))} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Pass rate versus token budget for four strategies">
        {/* ceiling band: the region only interaction reaches */}
        <rect x={m.l} y={ys(100)} width={iw} height={ys(ceil) - ys(100)}
          fill={C.gnd} opacity="0.06" />
        <line x1={m.l} x2={m.l + iw} y1={ys(ceil)} y2={ys(ceil)}
          stroke={C.ung} strokeWidth="1.5" strokeDasharray="5 4" />
        <text x={m.l + 6} y={ys(ceil) + 16} fontSize="11.5" fontFamily="IBM Plex Mono" fill="#A34413">
          ← internal ceiling {ceil}%
        </text>
        <text x={m.l + 6} y={ys(100) + 14} fontSize="11.5" fontFamily="IBM Plex Mono" fill={C.gndDk}>
          reached only by interaction ↑
        </text>
        {/* grid + axes */}
        {[40, 60, 80, 100].map(v => (
          <g key={v}>
            <line x1={m.l} x2={m.l + iw} y1={ys(v)} y2={ys(v)} stroke={C.grid} strokeWidth="1" />
            <text x={m.l - 9} y={ys(v) + 4} textAnchor="end" fontSize="11.5"
              fontFamily="IBM Plex Mono" fill={C.ink3}>{v}</text>
          </g>
        ))}
        {[1000, 5000, 20000].map(b => (
          <text key={b} x={xs(b)} y={H - m.b + 22} textAnchor="middle" fontSize="11.5"
            fontFamily="IBM Plex Mono" fill={C.ink3}>{b / 1000}K</text>
        ))}
        <text x={m.l + iw / 2} y={H - 6} textAnchor="middle" fontSize="12" fill={C.ink2}>
          token budget per task (log scale)
        </text>
        <text transform={`translate(14 ${m.t + ih / 2}) rotate(-90)`} textAnchor="middle"
          fontSize="12" fill={C.ink2}>pass rate (%)</text>

        {data.series.map(s => {
          const col = colors[s.key]
          const pts = s.budgets.map((b, i) => [xs(b), ys(s.means[i])])
          const d = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0]},${p[1]}`).join(' ')
          return (
            <g key={s.key}>
              {/* seed spread */}
              {s.budgets.map((b, i) => (s.seeds[i] || []).map((v, j) => (
                <circle key={j} cx={xs(b)} cy={ys(v)} r="3" fill={col} opacity="0.3" />
              )))}
              <path d={d} fill="none" stroke={col} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
              {pts.map((p, i) => (
                <g key={i}>
                  <circle cx={p[0]} cy={p[1]} r="6.5" fill={C.card} />
                  <circle cx={p[0]} cy={p[1]} r="4.5" fill={col} />
                  <circle cx={p[0]} cy={p[1]} r="14" fill="transparent"
                    onMouseMove={e => show(e, `<div class="t-h">${s.label}</div>${s.budgets[i] / 1000}K budget → ${s.means[i]}%<br/>seeds: ${(s.seeds[i] || []).join(' / ') || '—'}`)}
                    onMouseLeave={hide} />
                </g>
              ))}
              {/* direct label at line end */}
              <text x={pts[2][0] + 12} y={pts[2][1] + 4 + (s.key === 'L' ? 12 : 0) - (s.key === 'H' ? 8 : 0)}
                fontSize="12" fontWeight="600" fill={C.ink2} fontFamily="IBM Plex Sans">
                {s.means[2]}% <tspan fill={C.ink3} fontWeight="450">{shortLabel(s.key)}</tspan>
              </text>
            </g>
          )
        })}
      </svg>
      {Tip}
    </div>
  )
}
const shortLabel = k => ({
  H: 'proposer–reviewer', L: 'single-agent loop', S: 'best-of-N (oracle)', R: 'reasoning-only',
}[k])

/* ---- paired dumbbell: before -> after, e.g. defects or pass rates ---- */
export function Dumbbell({ rows, unit = '', max, good = 'down', labelWidth = 150, fmt = v => v, W = 560 }) {
  const { ref, show, hide, Tip } = useTip()
  const rowH = 46, m = { l: labelWidth, r: 84, t: 8, b: 26 }
  const H = m.t + rows.length * rowH + m.b
  const lim = max ?? Math.max(...rows.flatMap(r => [r.a, r.b])) * 1.12
  const xs = v => m.l + (v / lim) * (W - m.l - m.r)
  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Before and after comparison">
        {[0, 0.25, 0.5, 0.75, 1].map(f => (
          <line key={f} x1={xs(lim * f)} x2={xs(lim * f)} y1={m.t} y2={H - m.b}
            stroke={C.grid} strokeWidth="1" />
        ))}
        {rows.map((r, i) => {
          const y = m.t + i * rowH + rowH / 2
          const improved = good === 'down' ? r.b < r.a : r.b > r.a
          const col = improved ? C.gnd : C.ungDk
          return (
            <g key={i}>
              <text x={m.l - 12} y={y - 2} textAnchor="end" fontSize="12.5" fontWeight="600" fill={C.ink}>{r.label}</text>
              {r.sub && <text x={m.l - 12} y={y + 13} textAnchor="end" fontSize="10.5" fontFamily="IBM Plex Mono" fill={C.ink3}>{r.sub}</text>}
              <line x1={xs(r.a)} x2={xs(r.b)} y1={y} y2={y} stroke={col} strokeWidth="2.5" />
              {/* arrowhead toward the outcome */}
              <path d={arrow(xs(r.a), xs(r.b), y)} fill={col} />
              <circle cx={xs(r.a)} cy={y} r="6.5" fill={C.card} />
              <circle cx={xs(r.a)} cy={y} r="4.5" fill={C.ink3} />
              <circle cx={xs(r.b)} cy={y} r="7.5" fill={C.card} />
              <circle cx={xs(r.b)} cy={y} r="5.5" fill={col} />
              <text x={Math.max(xs(r.a), xs(r.b)) + 14} y={y + 4} fontSize="12" fontFamily="IBM Plex Mono" fill={C.ink2}>
                {fmt(r.a)} → {fmt(r.b)}{unit}
              </text>
              <rect x={m.l} y={y - rowH / 2} width={W - m.l - m.r} height={rowH} fill="transparent"
                onMouseMove={e => show(e, `<div class="t-h">${r.label}</div>single-shot ${fmt(r.a)}${unit} → reviewed ${fmt(r.b)}${unit}${r.tip ? `<br/>${r.tip}` : ''}`)}
                onMouseLeave={hide} />
            </g>
          )
        })}
        <text x={m.l} y={H - 6} fontSize="11" fontFamily="IBM Plex Mono" fill={C.ink3}>0</text>
        <text x={W - m.r} y={H - 6} textAnchor="end" fontSize="11" fontFamily="IBM Plex Mono" fill={C.ink3}>{fmt(lim)}{unit}</text>
      </svg>
      <Legend items={[
        { label: 'single-shot', color: C.ink3, dot: true },
        { label: 'after the grounded loop', color: C.gnd, dot: true },
        { label: 'got worse', color: C.ungDk, dot: true },
      ]} />
      {Tip}
    </div>
  )
}
function arrow(x1, x2, y) {
  if (Math.abs(x2 - x1) < 14) return ''
  const dir = x2 > x1 ? 1 : -1
  const tip = x2 - dir * 9
  return `M ${tip} ${y - 4.5} L ${x2 - dir * 1} ${y} L ${tip} ${y + 4.5} Z`
}

/* ---- horizontal bars with optional CI whiskers ---- */
export function HBars({ rows, unit = '%', max = 100, color = () => C.gnd, labelWidth = 160, W = 720 }) {
  const { ref, show, hide, Tip } = useTip()
  const rowH = 42, m = { l: labelWidth, r: 86, t: 4, b: 8 }
  const H = m.t + rows.length * rowH + m.b
  const xs = v => m.l + (Math.abs(v) / max) * (W - m.l - m.r)
  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Bar chart">
        {[0.25, 0.5, 0.75, 1].map(f => (
          <line key={f} x1={xs(max * f)} x2={xs(max * f)} y1={m.t} y2={H - m.b} stroke={C.grid} strokeWidth="1" />
        ))}
        {rows.map((r, i) => {
          const y = m.t + i * rowH + rowH / 2
          const w = xs(r.v) - m.l
          const col = color(r)
          return (
            <g key={i}
              onMouseMove={e => show(e, `<div class="t-h">${r.label}</div>${r.tipv ?? (Math.abs(r.v) + unit)}${r.tip ? `<br/>${r.tip}` : ''}`)}
              onMouseLeave={hide}>
              <text x={m.l - 12} y={y - (r.sub ? 3 : -4)} textAnchor="end" fontSize="12.5" fontWeight="600" fill={C.ink}>{r.label}</text>
              {r.sub && <text x={m.l - 12} y={y + 12} textAnchor="end" fontSize="10.5" fontFamily="IBM Plex Mono" fill={C.ink3}>{r.sub}</text>}
              <path d={`M ${m.l} ${y - 9} H ${m.l + Math.max(w, 2) - 4} a4 4 0 0 1 4 4 v10 a4 4 0 0 1 -4 4 H ${m.l} Z`} fill={col} />
              {r.ci && (
                <line x1={xs(r.ci[0])} x2={xs(r.ci[1])} y1={y} y2={y} stroke={C.ink} strokeWidth="1.6" />
              )}
              {r.ci && [0, 1].map(k => (
                <line key={k} x1={xs(r.ci[k])} x2={xs(r.ci[k])} y1={y - 4.5} y2={y + 4.5} stroke={C.ink} strokeWidth="1.6" />
              ))}
              <text x={Math.max(xs(r.v), r.ci ? xs(r.ci[1]) : 0) + 10} y={y + 4} fontSize="12"
                fontFamily="IBM Plex Mono" fill={C.ink2}>{r.vlabel ?? `${r.v}${unit}`}</text>
              <rect x={0} y={y - rowH / 2} width={W} height={rowH} fill="transparent" />
            </g>
          )
        })}
      </svg>
      {Tip}
    </div>
  )
}

/* ---- grouped SS vs RV columns (cross-model) ---- */
export function PairColumns({ rows, seedNote, W = 560 }) {
  const { ref, show, hide, Tip } = useTip()
  const H = 330, m = { l: 50, r: 12, t: 26, b: 56 }
  const iw = W - m.l - m.r, ih = H - m.t - m.b
  const ys = v => m.t + (1 - v / 100) * ih
  const groupW = iw / rows.length
  const barW = 24
  return (
    <div className="chart" ref={ref}>
      <Legend items={[
        { label: 'single-shot', color: C.ung },
        { label: 'with the grounded loop', color: C.gnd },
      ]} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Single-shot versus reviewed pass rate per model family">
        {[0, 25, 50, 75, 100].map(v => (
          <g key={v}>
            <line x1={m.l} x2={m.l + iw} y1={ys(v)} y2={ys(v)} stroke={C.grid} strokeWidth="1" />
            <text x={m.l - 9} y={ys(v) + 4} textAnchor="end" fontSize="11.5" fontFamily="IBM Plex Mono" fill={C.ink3}>{v}</text>
          </g>
        ))}
        {rows.map((r, i) => {
          const cx = m.l + groupW * i + groupW / 2
          const bars = [
            { x: cx - barW - 1, v: r.ss, col: C.ung, lab: 'single-shot', seeds: r.ss_seeds },
            { x: cx + 1, v: r.rv, col: C.gnd, lab: 'grounded loop', seeds: r.rv_seeds },
          ]
          return (
            <g key={r.model}>
              {bars.map((b, j) => (
                <g key={j}
                  onMouseMove={e => show(e, `<div class="t-h">${r.model} — ${b.lab}</div>${b.v}% pass rate<br/>seeds: ${(b.seeds || []).join(' / ') || '—'}`)}
                  onMouseLeave={hide}>
                  <path d={`M ${b.x} ${ys(0)} V ${ys(b.v) + 4} a4 4 0 0 1 4 -4 h ${barW - 8} a4 4 0 0 1 4 4 V ${ys(0)} Z`} fill={b.col} />
                  {/* seed whiskers */}
                  {b.seeds && b.seeds.length > 1 && (
                    <line x1={b.x + barW / 2} x2={b.x + barW / 2}
                      y1={ys(Math.min(...b.seeds))} y2={ys(Math.max(...b.seeds))}
                      stroke={C.ink} strokeWidth="1.6" />
                  )}
                  <text x={b.x + barW / 2} y={ys(b.seeds ? Math.max(b.v, ...b.seeds) : b.v) - 7}
                    textAnchor="middle" fontSize="11.5"
                    fontFamily="IBM Plex Mono" fill={C.ink2}>{b.v}</text>
                  <rect x={b.x - 4} y={m.t} width={barW + 8} height={ih} fill="transparent" />
                </g>
              ))}
              <text x={cx} y={H - m.b + 20} textAnchor="middle" fontSize="12.5" fontWeight="600" fill={C.ink}>{r.model}</text>
              {r.note && <text x={cx} y={H - m.b + 36} textAnchor="middle" fontSize="10.5" fontFamily="IBM Plex Mono" fill={C.ink3}>{r.note}</text>}
            </g>
          )
        })}
        <line x1={m.l} x2={m.l + iw} y1={ys(0)} y2={ys(0)} stroke={C.ink3} strokeWidth="1" />
        <text transform={`translate(14 ${m.t + ih / 2}) rotate(-90)`} textAnchor="middle" fontSize="12" fill={C.ink2}>pass rate (%)</text>
      </svg>
      {seedNote && <div style={{ fontSize: 12.5, color: C.ink3, fontFamily: 'IBM Plex Mono' }}>{seedNote}</div>}
      {Tip}
    </div>
  )
}

/* ---- allocation: pass rate vs proposer share ---- */
export function AllocationChart({ data, W = 560 }) {
  const { ref, show, hide, Tip } = useTip()
  const H = 330, m = { l: 52, r: 30, t: 16, b: 52 }
  const iw = W - m.l - m.r, ih = H - m.t - m.b
  const xs = v => m.l + v / 0.9 * iw
  const ys = v => m.t + (1 - v / 100) * ih
  const rows = [...data.rows].sort((a, b) => a.propose - b.propose)
  // collapse duplicate propose-shares into min/max dots + a line through the max
  const byShare = {}
  rows.forEach(r => { (byShare[r.propose] ||= []).push(r) })
  const shares = Object.keys(byShare).map(Number).sort((a, b) => a - b)
  const line = shares.map(s => [xs(s), ys(Math.max(...byShare[s].map(r => r.pass_rate)))])
  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Pass rate versus share of budget given to the proposer">
        {[0, 25, 50, 75, 100].map(v => (
          <g key={v}>
            <line x1={m.l} x2={m.l + iw} y1={ys(v)} y2={ys(v)} stroke={C.grid} strokeWidth="1" />
            <text x={m.l - 9} y={ys(v) + 4} textAnchor="end" fontSize="11.5" fontFamily="IBM Plex Mono" fill={C.ink3}>{v}</text>
          </g>
        ))}
        {[0.1, 0.25, 0.4, 0.5, 0.8].map(v => (
          <text key={v} x={xs(v)} y={H - m.b + 20} textAnchor="middle" fontSize="11.5" fontFamily="IBM Plex Mono" fill={C.ink3}>{Math.round(v * 100)}%</text>
        ))}
        <text x={m.l + iw / 2} y={H - 8} textAnchor="middle" fontSize="12" fill={C.ink2}>
          share of the 10K-token budget given to the proposer
        </text>
        <path d={line.map((p, i) => `${i ? 'L' : 'M'}${p[0]},${p[1]}`).join(' ')}
          fill="none" stroke={C.gnd} strokeWidth="2.5" strokeLinejoin="round" />
        {rows.map((r, i) => (
          <g key={i}
            onMouseMove={e => show(e, `<div class="t-h">${r.label}</div>propose ${Math.round(r.propose * 100)}% / execute ${Math.round(r.execute * 100)}% / review ${Math.round(r.review * 100)}%<br/>pass rate ${r.pass_rate}%`)}
            onMouseLeave={hide}>
            <circle cx={xs(r.propose)} cy={ys(r.pass_rate)} r="7" fill={C.card} />
            <circle cx={xs(r.propose)} cy={ys(r.pass_rate)} r="5" fill={C.gnd} />
            <circle cx={xs(r.propose)} cy={ys(r.pass_rate)} r="14" fill="transparent" />
          </g>
        ))}
        <text x={xs(0.8)} y={ys(93.3) - 14} textAnchor="end" fontSize="12" fontWeight="600" fill={C.ink2}>give the proposer the budget → 93%</text>
        <text x={xs(0.1) + 16} y={ys(6.7) + 4} fontSize="12" fontWeight="600" fill={C.ink2}>starve it → 7%</text>
      </svg>
      {Tip}
    </div>
  )
}

/* ---- distillation variance curves ---- */
export function VarianceChart({ data, W = 560 }) {
  const { ref, show, hide, Tip } = useTip()
  const H = 300, m = { l: 52, r: 120, t: 16, b: 46 }
  const iw = W - m.l - m.r, ih = H - m.t - m.b
  const xs = i => m.l + (i / 2) * iw
  const ys = v => m.t + (1 - v / 60) * ih
  const series = [
    { label: 'SFT student (keeps variance)', col: C.stu, vals: data.sft },
    { label: '+ RFT (variance crushed)', col: C.ungDk, vals: data.rft },
  ]
  return (
    <div className="chart" ref={ref}>
      <Legend items={series.map(s => ({ label: s.label, color: s.col }))} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Held-out solve rate versus samples for SFT and RFT students">
        {[0, 20, 40, 60].map(v => (
          <g key={v}>
            <line x1={m.l} x2={m.l + iw} y1={ys(v)} y2={ys(v)} stroke={C.grid} strokeWidth="1" />
            <text x={m.l - 9} y={ys(v) + 4} textAnchor="end" fontSize="11.5" fontFamily="IBM Plex Mono" fill={C.ink3}>{v}</text>
          </g>
        ))}
        {data.k.map((k, i) => (
          <text key={k} x={xs(i)} y={H - m.b + 22} textAnchor="middle" fontSize="11.5" fontFamily="IBM Plex Mono" fill={C.ink3}>k={k}</text>
        ))}
        <text x={m.l + iw / 2} y={H - 6} textAnchor="middle" fontSize="12" fill={C.ink2}>samples drawn (pass@k)</text>
        <text transform={`translate(14 ${m.t + ih / 2}) rotate(-90)`} textAnchor="middle" fontSize="12" fill={C.ink2}>tasks kept by judge (%)</text>
        {series.map(s => (
          <g key={s.label}>
            <path d={s.vals.map((v, i) => `${i ? 'L' : 'M'}${xs(i)},${ys(v)}`).join(' ')}
              fill="none" stroke={s.col} strokeWidth="2.5" strokeLinecap="round" />
            {s.vals.map((v, i) => (
              <g key={i}
                onMouseMove={e => show(e, `<div class="t-h">${s.label}</div>pass@${data.k[i]}: ${v}%`)}
                onMouseLeave={hide}>
                <circle cx={xs(i)} cy={ys(v)} r="6.5" fill={C.card} />
                <circle cx={xs(i)} cy={ys(v)} r="4.5" fill={s.col} />
                <circle cx={xs(i)} cy={ys(v)} r="14" fill="transparent" />
              </g>
            ))}
            <text x={xs(2) + 12} y={ys(s.vals[2]) + 4} fontSize="12" fontWeight="600" fill={C.ink2}>
              {s.vals[2]}%
            </text>
          </g>
        ))}
      </svg>
      {Tip}
    </div>
  )
}

/* ---- shared table twin ---- */
export function DataTable({ cols, rows }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="dataview">
        <thead><tr>{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j}>{v}</td>)}</tr>)}
        </tbody>
      </table>
    </div>
  )
}
