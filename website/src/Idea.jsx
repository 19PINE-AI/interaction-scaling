import React from 'react'
import { C, BASE } from './data.js'
import { Reveal, LoopDiagram, PicReasoning, PicSampling, PicInteraction, Compare } from './bits.jsx'

export default function Idea({ site }) {
  return (
    <section className="band" id="idea">
      <div className="wrap">
        <Reveal>
          <div className="sec-head">
            <div className="no">Part 1 · The idea</div>
            <h2>How do you make an AI better at a task, without training it further?</h2>
            <p className="lede">
              You give it more time at the moment it answers. Until recently there were
              two ways to spend that time — and they share a hidden weakness.
            </p>
          </div>
        </Reveal>

        <Reveal>
          <div className="axes3">
            <div className="axis-card">
              <div className="top">
                <h3>1 · Think longer</h3>
                <span className="chip internal">internal</span>
              </div>
              <div className="pic"><PicReasoning /></div>
              <p>
                Let the model reason step by step before answering — like giving a student
                extra time on the same exam question.
              </p>
              <div className="verdict">on our hard coding tasks: <b className="flat">plateaus at 73%</b> — more thinking stopped helping</div>
            </div>
            <div className="axis-card">
              <div className="top">
                <h3>2 · Try many times</h3>
                <span className="chip internal">internal</span>
              </div>
              <div className="pic"><PicSampling /></div>
              <p>
                Generate many independent answers and keep the best one. We even gave this
                strategy a <em>perfect</em> judge that always picks a correct answer if one exists.
              </p>
              <div className="verdict">even with a perfect judge: <b className="flat">plateaus at 87%</b> — you can’t pick an answer that was never written</div>
            </div>
            <div className="axis-card hot">
              <div className="top">
                <h3>3 · Check the work</h3>
                <span className="chip external">external</span>
              </div>
              <div className="pic"><PicInteraction /></div>
              <p>
                Let the model hand its draft to a real instrument — run the code, render the
                slide, measure it — then revise using what the instrument actually saw.
              </p>
              <div className="verdict">same token budget: <b className="up">keeps climbing to 100%</b>, across every model family we tried</div>
            </div>
          </div>
        </Reveal>

        <Reveal>
          <div className="prose" style={{ marginTop: 46 }}>
            <p>
              The first two strategies are <strong>internal</strong>: every extra word the model
              produces still comes from the same frozen brain reading the same question. It is a
              closed-book exam. No matter how long the model stares at its own answer, nothing
              it doesn&rsquo;t already know can enter the room — so both strategies eventually hit a
              wall the paper calls the <strong>internal ceiling</strong>.
            </p>
            <p>
              The third strategy is different in kind, not degree. Running the tests, or rendering
              the slide and measuring where every box actually landed, <strong>imports a fact from
              outside the model</strong>. That is why the paper treats interaction as a genuine
              third axis of test-time compute — not a variant of the other two.
            </p>
          </div>
        </Reveal>

        <Reveal><LoopDiagram /></Reveal>

        {/* ---------------- the catch: grounding on both sides ---------------- */}
        <Reveal>
          <div className="sec-head" style={{ marginTop: 70 }}>
            <div className="no">The catch</div>
            <h2>The loop only works if someone actually <em>looks</em></h2>
            <p className="lede">
              “Check the work” sounds easy. The subtlety — and the paper&rsquo;s central claim — is
              that the checking must be <strong>grounded</strong> on <em>both</em> sides of the loop.
            </p>
          </div>
        </Reveal>

        <Reveal>
          <div className="sides">
            <div className="side-card" style={{ borderTop: `3px solid ${C.gnd}` }}>
              <div className="tag" style={{ color: C.gndDk }}>Side 1 — the feedback</div>
              <h4>The critic must observe the defect, not guess at it</h4>
              <p>
                Feedback can come from a real instrument (run the tests; measure the rendered
                layout) or from another AI looking at a screenshot and offering an opinion.
                The two feel similar. They are not: in our experiments the screenshot-critic made
                slides <em>worse</em>, while the measuring instrument fixed 73% of their layout
                defects — same model, same budget, one swapped ingredient.
              </p>
            </div>
            <div className="side-card" style={{ borderTop: `3px solid ${C.gnd}` }}>
              <div className="tag" style={{ color: C.gndDk }}>Side 2 — the scoring</div>
              <h4>The scoreboard must observe the defect too</h4>
              <p>
                The standard way to grade AI-made visuals is to show a screenshot to a
                vision-language model. We found that judge rated 14 of 15 objectively broken
                figures “perfect” — overflowing content is cropped out of the screenshot before
                the judge ever sees it. Use that scoreboard and real improvements become
                statistically invisible.
              </p>
            </div>
          </div>
        </Reveal>

        <Reveal>
          <div className="prose">
            <p className="aside">
              <strong>One sentence to take away:</strong> letting a model revise against a real
              instrument is a third, stronger way to spend compute — but you will only build it,
              and only <em>see</em> that it works, if both the feedback and the measurement come
              from instruments that truly observe the artifact.
            </p>
          </div>
        </Reveal>

        {/* ---------------- see it once, concretely ---------------- */}
        <Reveal>
          <div className="sec-head" style={{ marginTop: 64 }}>
            <div className="no">See it once</div>
            <h2>One real example, start to finish</h2>
            <p className="lede">
              Below is an actual task from the benchmark: “draw the Transformer
              encoder–decoder architecture as a dense academic figure.” Left: the model&rsquo;s
              single attempt. Right: after three trips around the grounded loop. Drag the handle.
            </p>
          </div>
        </Reveal>

        <Reveal>
          <div style={{ maxWidth: 860, margin: '0 auto' }}>
            <Compare
              ssImg={`${BASE}images/figures/figure_001_ss.jpg`}
              rvImg={`${BASE}images/figures/figure_001_rv.jpg`} />
            <div style={{ display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap', justifyContent: 'center' }}>
              <span className="chip plain">real benchmark output — Claude Sonnet 4</span>
              <a className="chip external" style={{ textDecoration: 'none' }} href="#explorer">
                browse all 98 cases with full model transcripts ↓
              </a>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
