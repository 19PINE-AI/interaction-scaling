import React, { useEffect, useRef, useState } from 'react'
import { C } from './data.js'

/* fade-up on first scroll into view */
export function Reveal({ children, as: Tag = 'div', ...rest }) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ob = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { el.classList.add('in'); ob.disconnect() }
    }, { threshold: 0.12 })
    ob.observe(el)
    return () => ob.disconnect()
  }, [])
  return <Tag ref={ref} className="reveal" {...rest}>{children}</Tag>
}

export function StatTile({ value, small, label, color = C.ink }) {
  return (
    <div className="stat-tile">
      <div className="v" style={{ color }}>{value}{small && <small> {small}</small>}</div>
      <div className="l">{label}</div>
    </div>
  )
}

/* Figure card with number, caption and optional chart/table twin toggle */
let figCounter = 0
export function FigureCard({ no, title, sub, note, table, children, className = '' }) {
  const [mode, setMode] = useState('chart')
  return (
    <div className={`figure ${className}`}>
      <div className="fig-head">
        <span className="fig-no">FIG. {no}</span>
        <h3>{title}</h3>
        {table && (
          <div className="fig-tools">
            <button className={mode === 'chart' ? 'on' : ''} onClick={() => setMode('chart')}>chart</button>
            <button className={mode === 'table' ? 'on' : ''} onClick={() => setMode('table')}>table</button>
          </div>
        )}
      </div>
      {sub && <p className="fig-sub">{sub}</p>}
      {mode === 'chart' ? children : table}
      {note && <div className="fig-note">{note}</div>}
    </div>
  )
}

/* The propose -> observe -> revise loop, drawn as an animated instrument diagram */
export function LoopDiagram() {
  return (
    <div className="loopwrap">
      <svg viewBox="0 0 760 300" role="img" aria-label="The interaction loop: the model proposes an artifact, an instrument observes it, the observation comes back as feedback">
        <defs>
          <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={C.gnd} />
          </marker>
          <marker id="ah2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={C.ink3} />
          </marker>
        </defs>

        {/* model box */}
        <g>
          <rect x="30" y="100" width="170" height="100" rx="12" fill="#fff" stroke={C.ink} strokeWidth="1.5" />
          <text x="115" y="140" textAnchor="middle" fontFamily="Fraunces" fontSize="19" fontWeight="600" fill={C.ink}>The model</text>
          <text x="115" y="163" textAnchor="middle" fontSize="12" fill={C.ink2}>frozen weights —</text>
          <text x="115" y="180" textAnchor="middle" fontSize="12" fill={C.ink2}>no fine-tuning</text>
        </g>

        {/* artifact box */}
        <g>
          <rect x="310" y="35" width="150" height="86" rx="12" fill="#fff" stroke={C.ink3} strokeWidth="1.2" />
          <text x="385" y="70" textAnchor="middle" fontFamily="Fraunces" fontSize="16" fontWeight="600" fill={C.ink}>Artifact</text>
          <text x="385" y="92" textAnchor="middle" fontSize="11.5" fill={C.ink2}>code · slide · page</text>
          <text x="385" y="108" textAnchor="middle" fontSize="11.5" fill={C.ink2}>figure · animation</text>
        </g>

        {/* instrument box */}
        <g>
          <rect x="560" y="100" width="170" height="100" rx="12" fill={C.gnd} opacity="0.08" />
          <rect x="560" y="100" width="170" height="100" rx="12" fill="none" stroke={C.gnd} strokeWidth="2" />
          <text x="645" y="135" textAnchor="middle" fontFamily="Fraunces" fontSize="18" fontWeight="600" fill={C.gndDk}>Instrument</text>
          <text x="645" y="158" textAnchor="middle" fontSize="11.5" fill={C.ink2}>runs the tests · renders</text>
          <text x="645" y="174" textAnchor="middle" fontSize="11.5" fill={C.ink2}>the page · measures</text>
          <text x="645" y="190" textAnchor="middle" fontSize="11.5" fill={C.ink2}>every pixel box</text>
        </g>

        {/* arrows */}
        <path d="M 200 130 C 240 110, 265 90, 305 80" fill="none" stroke={C.ink3} strokeWidth="2" className="flow" markerEnd="url(#ah2)" />
        <text x="240" y="78" fontSize="12.5" fontFamily="IBM Plex Mono" fill={C.ink2}>1 · propose</text>

        <path d="M 465 80 C 510 90, 535 110, 567 128" fill="none" stroke={C.ink3} strokeWidth="2" className="flow" markerEnd="url(#ah2)" />
        <text x="486" y="78" fontSize="12.5" fontFamily="IBM Plex Mono" fill={C.ink2}>2 · observe</text>

        <path d="M 560 185 C 460 245, 310 245, 208 185" fill="none" stroke={C.gnd} strokeWidth="2.5" className="flow" markerEnd="url(#ah)" />
        <text x="380" y="262" textAnchor="middle" fontSize="12.5" fontFamily="IBM Plex Mono" fill={C.gndDk} fontWeight="600">3 · real feedback — “line 12 fails”, “title overlaps the axis”</text>
      </svg>
      <div className="loop-cap">
        Each trip around the loop imports a fact the model could not have generated itself.
      </div>
    </div>
  )
}

/* mini pictograms for the three axis cards */
export function PicReasoning() {
  // one long meandering line: thinking longer
  return (
    <svg viewBox="0 0 220 92" className="picsvg" style={{ width: '100%', height: '100%' }} aria-hidden="true">
      <path d="M12 46 h28 c14 0 14 -22 28 -22 s14 44 28 44 s14 -44 28 -44 s14 30 28 30 h36"
        fill="none" stroke={C.ung} strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="12" cy="46" r="5" fill={C.ung} />
      <circle cx="196" cy="54" r="5" fill={C.ung} />
      <text x="205" y="58" fontSize="11" fill={C.ink3} fontFamily="IBM Plex Mono">…</text>
    </svg>
  )
}
export function PicSampling() {
  // several parallel attempts, one circled
  const ys = [14, 33, 52, 71]
  return (
    <svg viewBox="0 0 220 92" style={{ width: '100%', height: '100%' }} aria-hidden="true">
      {ys.map((y, i) => (
        <g key={y}>
          <line x1="16" y1={y} x2={i === 2 ? 178 : 150 + i * 8} y2={y} stroke={C.ung} strokeWidth="2.5" strokeLinecap="round" opacity={i === 2 ? 1 : 0.42} />
          <circle cx="16" cy={y} r="4.5" fill={C.ung} opacity={i === 2 ? 1 : 0.42} />
        </g>
      ))}
      <circle cx="178" cy="52" r="11" fill="none" stroke={C.ink} strokeWidth="1.8" strokeDasharray="3 3" />
      <text x="196" y="56" fontSize="11" fill={C.ink3} fontFamily="IBM Plex Mono">pick 1</text>
    </svg>
  )
}
export function PicInteraction() {
  return (
    <svg viewBox="0 0 220 92" style={{ width: '100%', height: '100%' }} aria-hidden="true">
      <path d="M20 60 C 40 20, 80 20, 100 45 S 160 80 200 30" fill="none" stroke={C.gnd} strokeWidth="2.5" strokeLinecap="round" />
      {[[58, 33], [113, 52], [163, 57]].map(([x, y], i) => (
        <g key={i}>
          <circle cx={x} cy={y} r="7" fill="#fff" />
          <circle cx={x} cy={y} r="5" fill={C.gnd} />
          <path d={`M ${x - 3} ${y} l 2.4 2.6 l 4 -5`} stroke="#fff" strokeWidth="1.6" fill="none" />
        </g>
      ))}
      <circle cx="200" cy="30" r="5.5" fill={C.gnd} />
      <text x="30" y="84" fontSize="11" fill={C.ink3} fontFamily="IBM Plex Mono">check · fix · check · fix</text>
    </svg>
  )
}

/* before/after slider with drag + side-by-side toggle */
export function Compare({ ssImg, rvImg, ssLabel = 'single-shot', rvLabel = 'after the loop' }) {
  const [mode, setMode] = useState('sbs')
  const [x, setX] = useState(50)
  return (
    <div>
      <div className="cmp-toggle fig-tools" style={{ marginLeft: 0 }}>
        <button className={mode === 'sbs' ? 'on' : ''} onClick={() => setMode('sbs')}>side by side</button>
        <button className={mode === 'slide' ? 'on' : ''} onClick={() => setMode('slide')}>slider</button>
      </div>
      {mode === 'slide' ? (
        <div className="compare">
          <img src={rvImg} alt={rvLabel} />
          <div className="topimg" style={{ clipPath: `inset(0 ${100 - x}% 0 0)` }}>
            <img src={ssImg} alt={ssLabel} />
          </div>
          <div className="rail" style={{ left: `${x}%` }} />
          <div className="knob" style={{ left: `${x}%` }}>⇄</div>
          <span className="cornertag" style={{ left: 10, background: C.ungDk }}>{ssLabel}</span>
          <span className="cornertag" style={{ right: 10, background: C.gndDk }}>{rvLabel}</span>
          <input type="range" min="0" max="100" step="0.5" value={x}
            onChange={e => setX(+e.target.value)} aria-label="Reveal single-shot versus reviewed" />
        </div>
      ) : (
        <div className="sbs">
          <figure>
            <img src={ssImg} alt={ssLabel} />
            <figcaption><span className="swatchdot" style={{ background: C.ungDk }} />{ssLabel}</figcaption>
          </figure>
          <figure>
            <img src={rvImg} alt={rvLabel} />
            <figcaption><span className="swatchdot" style={{ background: C.gndDk }} />{rvLabel}</figcaption>
          </figure>
        </div>
      )}
    </div>
  )
}

/* raw model output with <think> sections de-emphasized */
export function RawOutput({ text, open: openDefault = false, label = 'show the raw model output' }) {
  const [open, setOpen] = useState(openDefault)
  if (!text) return null
  return (
    <div>
      <button className="disclose" onClick={() => setOpen(o => !o)}>
        {open ? '▾ hide' : `▸ ${label}`} <span style={{ opacity: .6 }}>({(text.length / 1000).toFixed(1)}K chars)</span>
      </button>
      {open && <pre className="raw">{renderThink(text)}</pre>}
    </div>
  )
}
function renderThink(text) {
  const parts = []
  let rest = text, i = 0
  while (rest.length) {
    const a = rest.indexOf('<think>')
    if (a === -1) { parts.push(<span key={i++}>{rest}</span>); break }
    if (a > 0) parts.push(<span key={i++}>{rest.slice(0, a)}</span>)
    let b = rest.indexOf('</think>')
    if (b === -1) b = rest.length; else b += '</think>'.length
    parts.push(<span key={i++} className="think">{rest.slice(a, b)}</span>)
    rest = rest.slice(b)
  }
  return parts
}
