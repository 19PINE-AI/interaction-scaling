import React, { useEffect, useState } from 'react'
import { useSite, useRoute, C } from './data.js'
import { Reveal, StatTile } from './bits.jsx'
import Idea from './Idea.jsx'
import Results from './Results.jsx'
import Explorer from './Explorer.jsx'
import CaseView from './CaseView.jsx'

const LoopMark = () => (
  <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
    <circle cx="11" cy="11" r="8" fill="none" stroke={C.gnd} strokeWidth="2.4" strokeDasharray="34 16" strokeLinecap="round" />
    <path d="M 17.5 5.5 l 3 1 l -2.4 2.2 z" fill={C.gnd} />
  </svg>
)

export default function App() {
  const { site, err } = useSite()
  const route = useRoute()
  const [active, setActive] = useState('idea')

  // scroll-spy for the top nav
  useEffect(() => {
    if (route[0] === 'case') return
    const ids = ['idea', 'results', 'explorer']
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) setActive(e.target.id) })
    }, { rootMargin: '-30% 0px -60% 0px' })
    ids.forEach(id => { const el = document.getElementById(id); if (el) obs.observe(el) })
    return () => obs.disconnect()
  }, [site, route[0]])

  if (err) return <div className="loading">Failed to load site data. Run <code>python -m scripts.build_site_v2_data</code>.</div>
  if (!site) return <div className="loading">loading…</div>

  if (route[0] === 'case' && route[1] && route[2]) {
    return <CaseView mod={route[1]} id={route[2]} />
  }

  return (
    <>
      <header className="topnav">
        <div className="inner">
          <a className="brand" href="#top"><LoopMark /><span className="bt">Grounding the Loop</span></a>
          <nav>
            <a href="#idea" className={active === 'idea' ? 'on' : ''}>The idea</a>
            <a href="#results" className={active === 'results' ? 'on' : ''}>Results</a>
            <a href="#explorer" className={active === 'explorer' ? 'on' : ''}>Explorer</a>
            <a href={site.meta.repo} target="_blank" rel="noreferrer" className="opt">Code ↗</a>
          </nav>
        </div>
      </header>

      <Hero site={site} />
      <Idea site={site} />
      <Results site={site} />
      <Explorer site={site} />
      <Footer site={site} />
    </>
  )
}

function Hero({ site }) {
  return (
    <div className="hero" id="top">
      <div className="inner">
        <div className="kicker">A research paper, explained · Pine AI × University of Washington</div>
        <h1>What if the model could <em>check its work</em> — and why you&rsquo;d never know it helped</h1>
        <p className="stand">
          AI models are given extra compute in two ways: <strong>thinking longer</strong> and{' '}
          <strong>trying many times</strong>. Both eventually stall, because nothing new enters
          the model&rsquo;s head. This paper studies a third way — <strong>letting the model test
          its own work against real instruments</strong> — and shows it breaks that ceiling,
          but only when both the feedback <em>and</em> the scoring truly observe the work.
        </p>
        <div className="byline">
          <b>Bojie Li</b> (Pine AI) · <b>Noah Shi</b> (University of Washington) · 2026
        </div>
        <div className="cta">
          <a className="btn solid" href="#idea">Read the 3-minute version ↓</a>
          <a className="btn line" href="#explorer">Browse the real test cases</a>
          <a className="btn line" href={site.meta.repo} target="_blank" rel="noreferrer">GitHub ↗</a>
        </div>
        <Reveal>
          <div className="statrow">
            <StatTile value="100" small="%" color={C.gndDk}
              label="pass rate on hard coding tasks with the grounded loop — across all seeds and three model families" />
            <StatTile value="14 / 15" color={C.ungDk}
              label="broken figures the standard AI judge rated “perfect” — the defects never reach its eyes" />
            <StatTile value="−73" small="%" color={C.gndDk}
              label="slide layout defects removed when the critic is a measuring instrument instead of an AI opinion" />
          </div>
        </Reveal>
      </div>
    </div>
  )
}

function Footer({ site }) {
  return (
    <footer>
      <div className="wrap">
        <div className="cols">
          <div>
            <h4>Grounding the Loop on Both Sides</h4>
            <p style={{ fontSize: 14.5, lineHeight: 1.65, margin: '0 0 16px' }}>
              Interaction as a third test-time compute axis, and why its gains are invisible
              without grounded evaluation. Every chart and every case on this page is generated
              from the paper&rsquo;s raw experiment logs.
            </p>
            <p style={{ fontSize: 14.5 }}>
              <a href={site.meta.repo} target="_blank" rel="noreferrer">Code, task suites, prompts &amp; instruments ↗</a>
            </p>
          </div>
          <div>
            <h4>Cite</h4>
            <pre>{`@article{li2026grounding,
  title  = {Grounding the Loop on Both Sides:
            Interaction as a Third Test-Time
            Compute Axis},
  author = {Li, Bojie and Shi, Noah},
  year   = {2026}
}`}</pre>
          </div>
        </div>
        <div className="small">
          Companion site to the paper · built from the experiment logs in the repository ·
          single-shot vs. reviewed artifacts are unedited model outputs.
        </div>
      </div>
    </footer>
  )
}
