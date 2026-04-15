# Think, Do, and Review: Interaction Scaling for LLM Agents

## Research Plan

---

## 1. Core Thesis

Current test-time scaling research focuses on **reasoning scaling** — generating longer chains of thought, sampling more candidates, or searching over reasoning trees. These approaches scale compute within the LLM's own token space, operating on the same fixed information available at generation time.

We propose **Interaction Scaling** — a fundamentally different dimension of test-time compute where agents improve their outputs by iteratively interacting with the external world: executing code and observing test results, rendering artifacts and obtaining visual feedback, calling tools and incorporating their outputs. Each interaction cycle introduces **genuinely new information** that did not exist during generation, breaking the information-theoretic ceiling that limits reasoning-only scaling.

The **proposer-reviewer** architecture is the natural primitive for interaction scaling. The proposer generates artifacts; the reviewer grounds its evaluation in external feedback (execution results, visual rendering, tool outputs) and provides structured improvement signals. We study how to allocate a fixed compute budget across three phases — proposal (think), execution (do), and review — and demonstrate that budget-aware interaction scaling consistently outperforms both reasoning scaling and naive (budget-unaware) interaction scaling across code generation, web page generation, and document editing tasks.

---

## 2. Contributions

### Contribution 1: The "New Information" Framework

We propose an information-theoretic taxonomy of feedback types that explains when multi-agent review helps and when it does not:

- **Type 0 — Self-Review (zero new information):** The same model re-reads its own output. The reviewer has access to exactly the same weights and context as the proposer. By the Data Processing Inequality, this cannot increase information — and empirically, it often decreases accuracy (Huang et al., ICLR 2024). This is analogous to re-reading your own essay without showing it to anyone else.

- **Type 1 — LLM Cross-Review (near-zero new information):** A different LLM (or the same LLM with a different prompt) critiques the output. The reviewer operates on the same textual information as the proposer. The IAD paper (2025) confirms that such "self-LLM feedback becomes repetitive and uninformative." The Stanford single-agent paper (2026) shows that under equal compute, this provides no architectural advantage over a single agent. This is analogous to re-reading your own essay wearing a different hat.

- **Type 2 — Environment-Grounded Review (genuine new information):** The output is executed, rendered, or verified by an external system, and the results are fed to the reviewer. Each cycle injects information that was physically impossible for the proposer to possess at generation time — a test failure, a visual rendering, a tool output. RLEF (ICML 2025), WebGen-Agent (2025), and CRITIC (ICLR 2024) all demonstrate strong gains from this type of feedback. This is analogous to running your experiment and looking at the actual data.

This taxonomy unifies and explains results from multiple prior papers:
- Why self-correction fails (Huang et al.) → Type 0
- Why debate gains collapse under equal compute (Stanford 2026) → Type 1
- Why CRITIC works with tools but not without → Type 2 vs Type 0
- Why WebGen-Agent nearly doubles accuracy → Type 2

The theoretical contribution: we formalize this using an information-theoretic argument. Let $I(X; Y)$ denote the mutual information between input $X$ and output $Y$. For Type 0/1 feedback, the review is a deterministic function of existing information, so $I(X; Y_{revised}) \leq I(X; Y_{original})$ by DPI. For Type 2 feedback, the environment introduces a new random variable $E$ (execution result, rendering), and $I(X; Y_{revised} | E) > I(X; Y_{original})$ becomes possible — the DPI no longer bounds us because new information has entered the system.

### Contribution 2: Interaction Scaling as a New Dimension of Test-Time Compute

We define **Interaction Scaling** as increasing test-time compute through additional cycles of environment interaction (execute → observe → revise), as opposed to:
- **Reasoning Scaling:** more thinking tokens (chain-of-thought, tree search, self-consistency)
- **Sampling Scaling:** more independent attempts (best-of-N)

"Thinking vs. Doing" (NeurIPS 2025) introduced the distinction between thinking and doing (environment interaction). We extend this with a third dimension: **reviewing** — a structured evaluation step grounded in environment feedback that converts raw execution results into actionable improvement signals. The three dimensions are:

| Dimension | What scales | New information? | Example |
|---|---|---|---|
| Think | Reasoning tokens per step | No | Longer chain-of-thought |
| Do | Environment interaction steps | Yes (raw) | More tool calls, more browsing |
| Review | Evaluation cycles with structured feedback | Yes (interpreted) | Execute code → analyze failures → provide revision plan |

The key insight: "Doing" alone (more interactions without structured review) hits a ceiling — the General AgentBench (2026) identifies a "context ceiling" where accumulated interaction history degrades performance. "Reviewing" extracts and structures the information from interactions, preventing context overload while preserving the information gain.

### Contribution 3: Budget-Aware Proposer-Reviewer Allocation

Given a total budget of $N$ steps, how should an agent allocate between:
- $k_1$ proposal attempts (generation/revision cycles)
- $k_2$ execution steps (running tests, rendering, tool calls)
- $k_3$ review cycles (analyzing execution results, producing structured feedback)

where $k_1 + k_2 + k_3 \leq N$.

We hypothesize and test:
1. **The optimal allocation is task-dependent.** Code generation benefits from more $k_2$ (test execution); visual generation benefits from more $k_3$ (visual review with VLM).
2. **Naive equal allocation is suboptimal.** The IAD paper uses fixed $N=K$ ratios without optimization. We show that adaptive allocation (more review early when uncertainty is high, more generation later when the solution is nearly correct) outperforms fixed ratios.
3. **The marginal value of review decreases faster than the marginal value of execution.** After 2-3 review cycles, the reviewer's feedback becomes redundant; but each new execution still introduces fresh information. This predicts a specific shape for the scaling curve.
4. **Budget-awareness enables qualitatively different strategies.** With a 30-step budget, the optimal strategy is generate-execute-submit. With 300 steps, it becomes plan-generate-execute-review-revise-retest-review. The transition between strategies is sharp, not gradual (echoing the "phase transition" finding of 2601.17311).

### Contribution 4: Cross-Modal Generalization

Prior work studies individual modalities in isolation:
- Code generation: RLEF, Agentic Verifier, SWE-bench agents
- Web page generation: WebGen-Agent, Sketch2Code
- Web browsing: TTI, CATTS
- Document editing: (understudied)

We demonstrate that the proposer-reviewer pattern with environment-grounded feedback works as a **unified architectural pattern** across:
1. **Code generation** — reviewer uses test execution results (pass/fail, error messages, coverage)
2. **Web page generation** — reviewer uses rendered screenshots + VLM analysis
3. **Slide/presentation generation** — reviewer uses rendered slides + visual comparison to spec
4. **Document editing** — reviewer uses rendered PDF + structural analysis

The same architecture (proposer generates → environment executes/renders → reviewer analyzes with multimodal feedback → proposer revises) applies across all four. We measure whether the "new information" framework's predictions hold uniformly: Type 2 feedback helps across all modalities, Type 0/1 does not.

---

## 3. Related Work

### 3.1 Self-Correction and Self-Review in LLMs

Huang et al. (ICLR 2024) demonstrated that LLMs cannot self-correct reasoning without external feedback — accuracy often decreases after self-review. The TACL 2025 survey "When Can LLMs Actually Correct Their Own Mistakes?" confirmed this across a broad range of settings: self-correction works only when (1) reliable external feedback is available, or (2) the model has been specifically fine-tuned for correction. The CRITIC framework (ICLR 2024) provided a controlled comparison: tool-grounded verification yields substantial gains, but removing the tool and relying on self-evaluation eliminates most improvement. Our work explains these findings through the "new information" framework and builds on them architecturally.

### 3.2 Test-Time Scaling

The foundational result from Snell et al. (ICLR 2025) showed that scaling test-time compute can be more effective than scaling model parameters for reasoning tasks. Subsequent work has explored multiple scaling strategies:

- **Reasoning scaling:** Chain-of-thought (Wei et al., 2022), tree-of-thought (Yao et al., 2023), self-consistency (Wang et al., 2023)
- **Sampling scaling:** Best-of-N, majority voting, reward-model reranking
- **Budget-aware scaling:** BATS (Google, 2025) introduced budget tracking for tool-augmented agents; BAVT (2026) proposed step-level value evaluation with budget-conditioned node selection; CATTS (2026) uses confidence-aware compute allocation for web agents

All of these operate primarily in the "Think" or "Sample" dimensions. Our work adds "Do" and "Review" as new dimensions and studies their scaling properties.

### 3.3 Thinking vs. Doing

Shen et al. (NeurIPS 2025) introduced the distinction between scaling per-step reasoning ("thinking") and scaling environment interaction ("doing"), showing that "given a fixed token budget, acting longer yields larger gains than thinking longer" for web agents. Their TTI approach uses curriculum-based RL to train agents with adaptive rollout lengths. Our work extends this trichotomy with "reviewing" as a third dimension, proposes an inference-time architectural pattern (proposer-reviewer) rather than a training approach, and generalizes beyond web browsing to code, visual, and document tasks.

### 3.4 Multi-Agent Architectures

The multi-agent debate literature (Du et al., 2023; Liang et al., 2023) shows that multiple LLMs debating can improve factual accuracy, but Tran & Kiela (2026) demonstrated that these gains collapse under equal compute budgets — a single agent is as good as debate when tokens are normalized. "Why Do Multi-Agent LLM Systems Fail?" (ICLR 2025) catalogued 14 failure modes across 7 MAS frameworks, identifying inter-agent misalignment and task verification failure as fundamental challenges. Our work shows that multi-agent (proposer-reviewer) provides genuine architectural advantage specifically when the reviewer introduces Type 2 (environment-grounded) feedback — the one case where the DPI argument does not apply.

### 3.5 Feedback in Agentic Scaling

The IAD paper (2025) is the most directly related work. It studies feedback's role in inference-time alignment through iterative agent decoding, demonstrating ~10% gains across Sketch2Code, Text2SQL, and WebShop. However, IAD is a single-agent approach, uses fixed allocation ratios (N=K), does not formalize feedback types, and does not study budget-optimal allocation. Our work provides the missing theoretical framework (why some feedback works), the missing optimization (how to allocate budget), and the missing architectural pattern (proposer-reviewer).

The Scaling Agentic Verifier (2026) uses a separate verifier agent for competitive coding — the closest existing work to our proposer-reviewer architecture. However, it is limited to competitive coding verification (finding counterexamples) and does not study budget allocation or cross-modal generalization.

### 3.6 Planner-Executor Architectures

Plan-and-Act (2025) separates planning from execution, achieving SOTA on WebArena-Lite. Their key finding — weak planners are the critical bottleneck — informs our architecture design. Our proposer-reviewer extends the planner-executor pattern with an explicit review phase grounded in environment feedback, closing the loop that Plan-and-Act leaves open (plan → execute, but no structured review of execution results).

### 3.7 Visual Feedback for Code Generation

WebGen-Agent (2025) uses multi-level visual feedback (screenshots + VLM descriptions) for iterative web page refinement, improving accuracy from 26.4% to 51.9%. ScreenCoder (2025) decomposes front-end generation into grounding, planning, and generation with screenshot comparison. Vision-Guided Iterative Refinement (2026) uses a multimodal LLM as a visual critic. These works validate the effectiveness of visual feedback for specific tasks; we unify them under the interaction scaling framework and study their scaling properties systematically.

---

## 4. Framing

### Paper Narrative (Introduction Flow)

**Opening (the problem):** LLM agents are increasingly used for artifact-producing tasks — writing code, generating web pages, creating presentations, editing documents. Test-time scaling has emerged as a powerful lever for improving these agents, with recent work showing that more inference-time compute can compensate for model limitations. But current scaling approaches focus overwhelmingly on "thinking more" — longer reasoning traces, more samples, deeper search trees.

**The gap:** We observe that for artifact-producing tasks, the most valuable test-time compute is not additional thinking but additional *interaction with the real world*. When a coding agent executes its code and sees test failures, it gains information that no amount of reasoning could have produced. When a web generation agent renders its HTML and sees a screenshot, it obtains visual feedback that was physically impossible to derive from the code alone. This "interaction compute" has fundamentally different scaling properties than reasoning compute.

**The framework:** We formalize this observation through an information-theoretic lens. We define three types of feedback by their information content (Type 0/1/2) and show that only environment-grounded feedback (Type 2) can break through the information ceiling that bounds reasoning-only scaling. We introduce **Interaction Scaling** — a new dimension of test-time compute that scales performance through cycles of environment interaction and structured review.

**The architecture:** The proposer-reviewer pattern is the natural primitive for interaction scaling. The proposer generates artifacts ("Think"); the environment executes/renders them ("Do"); the reviewer analyzes the results with multimodal feedback and provides structured revision guidance ("Review"). We study how to allocate a fixed budget across these three phases and demonstrate that budget-aware allocation significantly outperforms both fixed-ratio allocation and reasoning-only scaling.

**The results:** Across four modalities (code, web pages, slides, documents), interaction scaling with budget-aware proposer-reviewer consistently outperforms: (1) single-agent with self-review (Type 0), (2) multi-agent debate (Type 1), (3) single-agent with naive external feedback (Type 2 without budget awareness), and (4) best-of-N sampling. The gains are largest at moderate budgets (50-200 steps) and the optimal allocation strategy varies predictably by task modality.

---

## 5. Experimental Plan

### 5.1 Tasks and Benchmarks

We select four tasks that span different feedback modalities:

| Task | Artifact | Environment Feedback | Review Signal | Benchmark |
|---|---|---|---|---|
| Code generation | Python code | Test execution (pytest) | Pass/fail, error messages, coverage | HumanEval+, MBPP+, SWE-bench Lite |
| Web page generation | HTML/CSS/JS | Browser rendering | Screenshot + VLM comparison to spec | Sketch2Code, Design2Code |
| Slide generation | PPTX (via python-pptx) | Slide rendering | Rendered slide images + VLM evaluation | Custom benchmark (50 slide specs with reference designs) |
| Document editing | LaTeX/Markdown → PDF | PDF rendering | Rendered PDF + structural/visual analysis | Custom benchmark (50 editing tasks with acceptance criteria) |

For the custom benchmarks (slides, documents), we create evaluation datasets with:
- Input specifications (natural language description + optional reference design)
- Automated evaluation metrics (structural similarity, visual fidelity, content accuracy)
- Human evaluation for a subset (50 examples, 3 annotators)

### 5.2 Baselines

**B1 — Single-Agent, No Review (Think only):**
Standard generation with chain-of-thought. The agent generates the artifact in one shot, using all $N$ steps for reasoning and generation. No execution, no review.

**B2 — Single-Agent, Self-Review (Think + Type 0):**
The agent generates, then re-reads its output and self-critiques. Implements the "review your code" prompt pattern. Budget split: $N/2$ for generation, $N/2$ for self-review and revision.

**B3 — Multi-Agent Debate (Think + Type 1):**
Two agents debate the output quality, then a third agent synthesizes. No environment execution. Budget split: $N/3$ per agent.

**B4 — Best-of-N Sampling (Sample):**
Generate $N$ independent candidates, select the best using a reward model or self-consistency. Pure sampling scaling.

**B5 — Single-Agent with External Feedback, Fixed Allocation (Do + Review, naive):**
The agent generates, executes, reads feedback, and revises in a fixed loop. Equal allocation: generate for $N/3$ steps, execute for $N/3$ steps, revise for $N/3$ steps. No budget awareness.

**B6 — IAD Reproduction (IAD baseline):**
Reproduce the Iterative Agent Decoding approach with fixed $N=K$ ratios.

**Ours — Budget-Aware Proposer-Reviewer (Think + Do + Review, optimized):**
Proposer generates → environment executes/renders → reviewer analyzes with multimodal feedback → proposer revises. Budget allocation is dynamic, optimized through the mechanisms described in 5.4.

### 5.3 Core Experiments

**Experiment 1: Feedback Type Ablation**

Hold the architecture constant (proposer-reviewer with $N$ = 100 steps) and vary the feedback type:
- Type 0: Reviewer sees only the generated code/HTML (no execution)
- Type 1: Reviewer is a different LLM that critiques without execution
- Type 2a: Reviewer sees execution results (test pass/fail, error messages)
- Type 2b: Reviewer sees rendered visual output (screenshot)
- Type 2c: Reviewer sees both execution results and visual output

Prediction: Type 2 >> Type 1 ≈ Type 0. Type 2c ≥ Type 2a and Type 2c ≥ Type 2b (complementary signals).

This directly tests Contribution 1 (the "new information" framework).

**Experiment 2: Scaling Curves by Dimension**

Fix total budget $N \in \{10, 30, 50, 100, 200, 500\}$ and measure performance for:
- Reasoning scaling only (all budget to chain-of-thought/self-consistency)
- Sampling scaling only (all budget to best-of-N)
- Interaction scaling only (all budget to propose-execute-review cycles)
- Combined (optimized allocation across all three)

Plot performance vs. $N$ for each strategy and each task.

Prediction: Interaction scaling dominates reasoning and sampling scaling for artifact-producing tasks. The gap widens at moderate budgets (50-200) where interaction scaling enables qualitatively different strategies (planning + testing + review), while reasoning scaling saturates.

This directly tests Contribution 2 (interaction scaling as a new dimension).

**Experiment 3: Budget Allocation Optimization**

For fixed $N$ = 100, sweep the allocation ratio:
- Vary $k_1$ (proposal steps) from 10% to 80% of budget
- Vary $k_2$ (execution steps) from 10% to 80% of budget
- Vary $k_3$ (review steps) from 10% to 80% of budget
- Subject to $k_1 + k_2 + k_3 = N$

Plot performance as a heatmap over the allocation simplex. Identify the optimal allocation for each task.

Then test adaptive allocation: a meta-controller that adjusts $k_1/k_2/k_3$ ratios dynamically based on:
- Iteration number (more review early, more generation late)
- Current confidence/quality estimate
- Remaining budget

Compare adaptive vs. fixed-optimal vs. fixed-equal allocation.

This directly tests Contribution 3 (budget-aware allocation).

**Experiment 4: Cross-Modal Generalization**

Run the full proposer-reviewer system with Type 2 feedback on all four tasks. Compare:
- Task-specific hand-tuned reviewer prompts
- Generic reviewer prompt ("analyze the execution/rendering results and provide structured feedback")
- Does the same architectural pattern work across code, web, slides, documents?

This directly tests Contribution 4 (cross-modal generalization).

**Experiment 5: Verification Gap Analysis**

Inspired by General AgentBench's "verification gap" finding:
- For each task, generate $K$ candidates with the proposer
- Measure pass@$K$ (best candidate exists among $K$)
- Measure choose@$K$ with self-selection (proposer picks best)
- Measure choose@$K$ with environment-grounded reviewer selection
- Quantify how much of the verification gap the reviewer closes

Prediction: The reviewer with Type 2 feedback closes significantly more of the verification gap than self-selection, especially for visual tasks where the gap between "code looks right" and "rendering looks right" is largest.

### 5.4 Budget-Aware Allocation Mechanism

We propose and compare three allocation strategies:

**Strategy A — Fixed Ratio:**
Pre-determined split (e.g., 40% propose, 30% execute, 30% review). Baseline approach.

**Strategy B — Phase-Adaptive:**
Rule-based adaptation:
- Round 1: Heavy on proposal (60/20/20) — establish initial solution
- Round 2+: Heavy on review (20/40/40) — identify and fix issues
- Final round: Heavy on execution (20/60/20) — thorough testing

**Strategy C — Confidence-Conditioned:**
Use the reviewer's confidence score (derived from execution pass rate or visual similarity metric) to dynamically allocate:
- Low confidence → more review cycles (need to understand what's wrong)
- Medium confidence → more execution (need to verify fixes)
- High confidence → stop early (save budget, solution is good)

The meta-controller is a simple rule-based system (not a learned model), to keep the contribution focused on the architectural pattern rather than the optimization algorithm. We leave learned allocation as future work.

### 5.5 Models

- **Proposer:** Claude Sonnet 4.6 (primary), GPT-4.1 (secondary, for generalization)
- **Reviewer:** Claude Sonnet 4.6 with multimodal input (screenshots + text)
- **Ablation:** Test with weaker reviewer models (Haiku) to study whether the reviewer needs to be as capable as the proposer

### 5.6 Evaluation Metrics

| Task | Primary Metric | Secondary Metrics |
|---|---|---|
| Code generation | pass@1 (test execution) | Coverage, code quality (pylint score) |
| Web page generation | Visual similarity (SSIM + CLIP) | Structural accuracy (DOM comparison), layout IoU |
| Slide generation | Visual fidelity (VLM score) | Content accuracy, layout consistency |
| Document editing | Task completion rate | Formatting accuracy, content preservation |

All experiments report:
- Performance vs. total budget $N$ (scaling curves)
- Performance vs. allocation ratio (allocation sensitivity)
- Wall-clock time and token cost (practical efficiency)

### 5.7 Ablation Studies

1. **Reviewer architecture:** Single reviewer agent vs. reviewer-as-tool (reviewer is a function call within the proposer's loop) — does architectural separation matter?
2. **Feedback granularity:** Full execution output vs. pass/fail only vs. error message only — how much feedback detail is needed?
3. **Number of review cycles:** 1 vs. 2 vs. 3 vs. adaptive — where do diminishing returns kick in?
4. **Proposer-reviewer model asymmetry:** Strong proposer + weak reviewer vs. weak proposer + strong reviewer — which matters more? (Connect to Plan-and-Act's "weak planner" finding)
5. **Cross-review:** Can a code reviewer help with web generation and vice versa? How modality-specific must the reviewer be?

---

## 6. Expected Results and Story

### The scaling curve story

We expect to produce a figure showing four scaling curves (performance vs. total budget $N$) for each task:

```
Performance
    |          ___________  ← Interaction Scaling (ours)
    |        /
    |      /    ________   ← Interaction (naive, no budget awareness)
    |    /    /
    |   /   /   ______    ← Sampling Scaling (best-of-N)
    |  /  /   /
    | / /   /  ______     ← Reasoning Scaling (more CoT)
    |//   /  /
    |   /  /
    |  / /
    | //
    |/
    +------------------------→ Budget N
```

Key narrative: reasoning scaling saturates early (the model can't think its way to correctness without seeing execution results); sampling scaling is more efficient but still limited (generating more candidates doesn't help if you can't evaluate them); naive interaction scaling helps significantly (executing and observing is valuable); budget-aware interaction scaling helps most (smart allocation of propose/do/review maximizes information gain per step).

### The allocation heatmap story

For code generation, we expect the optimal allocation to favor execution (run many tests):
- Optimal: ~30% propose, ~45% execute, ~25% review

For visual generation (web pages, slides), we expect the optimal allocation to favor review (VLM analysis is expensive but critical):
- Optimal: ~35% propose, ~25% execute (rendering is fast), ~40% review

This modality-dependent optimal allocation is itself a finding: there is no one-size-fits-all ratio, and budget-aware adaptation is necessary.

### The information-theoretic story

We expect the feedback type ablation to show a clear hierarchy:
- Type 2 (environment-grounded) >> Type 1 (LLM cross-review) ≈ Type 0 (self-review)
- With the gap largest at small budgets (where each feedback cycle's information content matters most)

This validates the "new information" framework as a predictive theory, not just a post-hoc explanation.

---

## 7. Paper Outline

1. **Introduction** (1.5 pages) — Problem, gap, thesis, contributions
2. **Background and Framework** (2 pages)
   - 2.1 Test-time scaling: reasoning, sampling, and interaction
   - 2.2 The "new information" framework (Type 0/1/2 feedback)
   - 2.3 Information-theoretic analysis
3. **Method: Budget-Aware Proposer-Reviewer** (2 pages)
   - 3.1 Architecture (proposer, environment, reviewer)
   - 3.2 Budget allocation strategies (fixed, phase-adaptive, confidence-conditioned)
   - 3.3 Cross-modal instantiation (code, web, slides, documents)
4. **Experiments** (3 pages)
   - 4.1 Feedback type ablation (validates framework)
   - 4.2 Scaling curves by dimension (validates interaction scaling)
   - 4.3 Budget allocation optimization (validates budget awareness)
   - 4.4 Cross-modal generalization (validates universality)
   - 4.5 Verification gap analysis
5. **Analysis and Discussion** (1 page)
   - 5.1 When does interaction scaling fail?
   - 5.2 The role of reviewer capability
   - 5.3 Connection to human work patterns
6. **Related Work** (1 page)
7. **Conclusion** (0.5 page)
8. **Appendix** — Ablation studies, prompt templates, custom benchmark details

Target: 9 pages + references + appendix (standard top-venue format).

---

## 8. Key Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Type 2 feedback doesn't consistently beat Type 1 across all tasks | Low (strong prior evidence from RLEF, WebGen-Agent, CRITIC) | If some tasks show weaker gains, this is itself a finding — characterize when interaction scaling matters most |
| Budget allocation optimization shows flat sensitivity (any reasonable ratio works) | Medium | If true, reframe: "interaction scaling is robust to allocation" is also a useful finding; focus the contribution on the framework and cross-modal generalization instead |
| Custom benchmarks (slides, documents) are criticized for lack of standardization | Medium | Use established benchmarks where available (HumanEval+, Sketch2Code); keep custom benchmarks small and well-documented; release them publicly |
| Reviewers argue this is "just engineering" not research | Low-Medium | The information-theoretic framework and scaling law analysis provide the conceptual contribution; experiments test specific predictions derived from the theory |
| A concurrent paper publishes similar results | Medium (active field) | Move fast; differentiate on the unified framework (most concurrent work is modality-specific) |

---

## 9. Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Literature deep-dive and framework formalization | 2 weeks | Section 2 draft, formal definitions |
| Infrastructure: proposer-reviewer system for 4 modalities | 3 weeks | Working system for code, web, slides, documents |
| Experiment 1 (feedback type ablation) | 2 weeks | Key framework validation |
| Experiment 2 (scaling curves) | 2 weeks | Core scaling results |
| Experiment 3 (budget allocation) | 2 weeks | Allocation optimization results |
| Experiments 4-5 + ablations | 2 weeks | Cross-modal + verification gap |
| Writing and iteration | 3 weeks | Complete paper draft |
| **Total** | **~16 weeks** | Submission-ready paper |

---

## 10. References

### Self-Correction and Review
- Huang et al. "Large Language Models Cannot Self-Correct Reasoning Yet." ICLR 2024. https://arxiv.org/abs/2310.01798
- Kamoi et al. "When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey." TACL 2025. https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713
- Gou et al. "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing." ICLR 2024. https://arxiv.org/abs/2305.11738

### Test-Time Scaling
- Snell et al. "Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Parameters." ICLR 2025.
- Shen et al. "Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction." NeurIPS 2025. https://arxiv.org/abs/2506.07976
- Ruan et al. "On the Role of Feedback in Test-Time Scaling of Agentic AI Workflows." 2025. https://arxiv.org/abs/2504.01931

### Budget-Aware Agents
- Jiang et al. "Budget-Aware Tool-Use Enables Effective Agent Scaling (BATS)." 2025. https://arxiv.org/abs/2511.17006
- "Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents (BAVT)." 2026. https://arxiv.org/abs/2603.12634
- "Phase Transition for Budgeted Multi-Agent Synergy." 2026. https://arxiv.org/abs/2601.17311

### Multi-Agent Systems
- Tran & Kiela. "Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets." 2026. https://arxiv.org/abs/2604.02460
- Cemri et al. "Why Do Multi-Agent LLM Systems Fail?" ICLR 2025. https://arxiv.org/abs/2503.13657
- "MAR: Multi-Agent Reflexion Improves Reasoning Abilities in LLMs." 2025. https://arxiv.org/abs/2512.20845

### Planner-Executor
- "Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks." 2025. https://arxiv.org/abs/2503.09572

### Environment Feedback for Code/Visual Generation
- Gehring et al. "RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning." ICML 2025. https://arxiv.org/abs/2410.02089
- "WebGen-Agent: Enhancing Interactive Website Generation with Multi-Level Feedback." 2025. https://arxiv.org/abs/2509.22644
- "ScreenCoder: Advancing Visual-to-Code Generation for Front-End Automation." 2025. https://arxiv.org/abs/2507.22827
- "Scaling Agentic Verifier for Competitive Coding." 2026. https://arxiv.org/abs/2602.04254
- "Vision-Guided Iterative Refinement for Frontend Code Generation." 2026. https://arxiv.org/abs/2604.05839

### Agent Benchmarking
- "Benchmark Test-Time Scaling of General LLM Agents." 2026. https://arxiv.org/abs/2602.18998
- "Agentic Test-Time Scaling for WebAgents (CATTS)." 2026. https://arxiv.org/abs/2602.12276
