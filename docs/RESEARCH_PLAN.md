# Agents that check their work: interaction scaling through environmental feedback

## Research Plan

---

## 1. Core Thesis

Current test-time scaling research focuses on **reasoning scaling** — generating longer chains of thought, sampling more candidates, or searching over reasoning trees. These approaches scale compute within the LLM's own token space, operating on the same fixed information available at generation time. Models that can only reason cannot reliably check their own work: without a ground-truth signal from outside the weights, self-critique has no anchor.

We propose **Interaction Scaling** — a fundamentally different dimension of test-time compute where agents improve their outputs by iteratively interacting with an external environment: executing code and observing test results, rendering artifacts and obtaining visual feedback, verifying claims against external sources. Each interaction cycle introduces **environmental feedback** — information new to the agent and reliable because it originates outside the model's weights — breaking the information-theoretic ceiling that limits reasoning-only scaling. An agent that checks its work is an agent that queries its environment, not one that talks to itself.

The **proposer-reviewer** architecture is an effective primitive for interaction scaling. The proposer generates artifacts; the reviewer grounds its evaluation in environmental feedback (execution results, visual rendering, fact verification) and provides structured improvement signals. We formalize and systematically study how to allocate a fixed compute budget across three phases — proposal (think), execution (do), and review — and demonstrate that budget-aware interaction scaling consistently outperforms both reasoning scaling and naive (budget-unaware) interaction scaling across code generation, web page generation, slide generation, video editing, and deep research tasks.

Finally, we ask whether this externally scaffolded behavior can be **internalized**: can a small student model, distilled from multi-turn teacher trajectories, learn to check its own work at inference time? We evaluate on strictly held-out tasks and find that distillation transfers the *format* of interaction scaling (budget-aware output, structured self-critique) but that the *gains* are bounded by the quality of the environmental feedback signal — interaction scaling distills only as well as the environment that provided it.

---

## 2. Contributions

### Contribution 1: The Grounded Feedback Framework

We propose an information-theoretic framework that explains when multi-agent review helps and when it does not. The key insight: feedback is valuable not merely because it is "new" (LLM sampling noise is technically new but unreliable), but because it is **externally grounded** — originating from systems outside the model that provide reliable signal about artifact quality.

#### Feedback Taxonomy

- **Type 0 — Self-Review (no grounded information):** The same model re-reads its own output. The reviewer operates on the same weights and context as the proposer. Empirically, this often decreases accuracy (Huang et al., ICLR 2024). Analogous to re-reading your own essay without showing it to anyone else.

- **Type 1 — LLM Cross-Review (no grounded information):** A different LLM (or the same LLM with a different prompt) critiques the output. Although a different model technically brings different priors (different weights encode different training distributions), the reviewer still operates on the same textual representation of the artifact with no external verification. The IAD paper (Ruan et al., 2025) confirms that such feedback "becomes repetitive and uninformative." Tran & Kiela (2026) show that under equal compute budgets, multi-agent debate provides no architectural advantage over a single agent for reasoning tasks. Analogous to re-reading your own essay wearing a different hat.

- **Type 2 — Static Tool Feedback (grounded, pre-execution):** External tools analyze the artifact without executing it — linters, type checkers, static analyzers, code search, structural validators. These introduce genuinely new information: a type error, an unused variable warning, or a security vulnerability flag are all signals the proposer did not have at generation time. The information is grounded in formal language specifications and rule sets, though it is limited to properties verifiable without runtime behavior. Analogous to having a copy editor check your manuscript for grammatical errors before publication.

- **Type 3 — Dynamic Environment Feedback (grounded, post-execution):** The artifact is executed, rendered, or verified against external sources, and the results are fed to the reviewer. Each cycle injects information that was physically impossible for the proposer to possess at generation time — a test failure revealing a logic error, a visual rendering exposing layout overflow, a search engine result contradicting a factual claim. RLEF (Gehring et al., ICML 2025), WebGen-Agent (2025), and CRITIC (Gou et al., ICLR 2024) all demonstrate strong gains from this feedback type. Analogous to running your experiment and looking at the actual data.

  Type 3 feedback has several sub-modalities:
  - **Type 3a — Execution feedback:** Test pass/fail, error messages, coverage reports, runtime behavior
  - **Type 3b — Visual rendering feedback:** Screenshots of rendered artifacts analyzed by VLMs
  - **Type 3c — Temporal feedback:** Keyframe sequences from video/animation, analyzed for temporal coherence
  - **Type 3d — Factual verification feedback:** Search engine results, database queries, or knowledge base lookups that verify or contradict specific claims

This taxonomy unifies and explains results from multiple prior papers:
- Why self-correction fails (Huang et al.) → Type 0
- Why debate gains collapse under equal compute (Tran & Kiela, 2026) → Type 1
- Why CRITIC works with tools but not without → Type 3 vs Type 0
- Why WebGen-Agent nearly doubles accuracy → Type 3b
- Why static analysis catches bugs that self-review misses → Type 2

#### Information-Theoretic Formalization

We formalize the framework using an explicit graphical model. Let $T$ denote the task specification, $A$ the generated artifact, and $F$ the reviewer's feedback.

**Type 0/1 (no grounding):** The information flow follows $T \to A \to F$, forming a Markov chain. The reviewer's feedback is derived solely from $A$ (and $T$, which both proposer and reviewer share). While LLM sampling introduces stochastic noise — making $F$ technically a random variable — this noise is uncorrelated with artifact quality. By the Data Processing Inequality, $I(T; F) \leq I(T; A)$: the feedback cannot contain more task-relevant information than the artifact itself. The reviewer may rephrase or reorganize observations about $A$, but cannot surface facts about whether $A$ actually works.

**Type 2/3 (grounded):** The information flow introduces an external variable $E$ representing the environment's response (execution result, rendering, search result, lint output). The graphical model becomes $T \to A \to E$ with $F = f(T, A, E)$. Crucially, $E$ carries information about the **actual behavior or correctness** of $A$ that is not deducible from $A$'s surface form: a program that "looks correct" may fail tests; a web page that "looks reasonable" in code may overflow visually; a factual claim that "sounds right" may be contradicted by evidence. Since $E \not\perp T \mid A$ in general (the environment's response reveals task-relevant information beyond the artifact text), the DPI no longer bounds us: $I(T; F \mid E) > I(T; A)$ becomes achievable. The environment breaks the Markov chain by introducing a genuinely new, reliable information source.

The practical significance: this explains why simply allocating more compute to reasoning (deeper chain-of-thought) hits diminishing returns for artifact-producing tasks — the model cannot reason its way to information it does not possess. Only external grounding can supply that information.

### Contribution 2: Interaction Scaling — Formalizing a Dimension of Test-Time Compute

We formalize **Interaction Scaling** as increasing test-time compute through additional cycles of environment interaction (execute → observe → revise), as opposed to:
- **Reasoning Scaling:** more thinking tokens (chain-of-thought, tree search, self-consistency)
- **Sampling Scaling:** more independent attempts (best-of-N)

Shen et al. (NeurIPS 2025) introduced the distinction between thinking and doing (environment interaction). We refine this with a third component: **reviewing** — a structured evaluation step grounded in external feedback that converts raw execution results into actionable improvement signals. We acknowledge that the think-do-review pattern is already implicit in many agentic systems (SWE-bench agents, coding assistants); our contribution is to formalize these as tunable, budgeted components and systematically study their scaling properties. The three components are:

| Component | What scales | Grounded information? | Example |
|---|---|---|---|
| Think | Reasoning tokens per step | No | Longer chain-of-thought |
| Do | Environment interaction steps | Yes (raw) | More tool calls, test runs, searches |
| Review | Evaluation cycles with structured feedback | Yes (interpreted) | Execute code → analyze failures → provide revision plan |

The key insight: "Doing" alone (more interactions without structured review) hits a ceiling — General AgentBench (2026) identifies a "context ceiling" where accumulated interaction history degrades performance. "Reviewing" extracts and structures the information from interactions, preventing context overload while preserving the information gain.

We note that interaction scaling inherently involves reasoning (the proposer thinks while generating; the reviewer reasons about feedback). The meaningful comparison is not "reasoning vs. interaction" but rather "reasoning without environmental grounding vs. reasoning with environmental grounding." We design our experiments to make this distinction precise.

### Contribution 3: Budget-Aware Proposer-Reviewer Allocation

Given a total budget of $B$ tokens (measured as total LLM tokens consumed across all API calls), how should an agent allocate across:
- $b_1$ tokens for proposal (generation/revision)
- $b_2$ tokens for processing execution/rendering results
- $b_3$ tokens for review (analyzing results, producing structured feedback)

where $b_1 + b_2 + b_3 \leq B$. We additionally report wall-clock time and API cost to capture the non-token costs of environment interaction (test execution time, rendering latency, search API calls).

We test the following hypotheses:
1. **The optimal allocation is task-dependent.** Code generation benefits from more execution cycles (run more tests); visual generation benefits from more review (VLM analysis is expensive but critical); deep research benefits from more fact-verification cycles.
2. **Naive equal allocation is suboptimal.** Adaptive allocation (more review early when uncertainty is high, more generation later when the solution is nearly correct) outperforms fixed ratios.
3. **Does the marginal value of review decrease faster than the marginal value of execution?** We test this empirically rather than assuming it. If review exhibits diminishing returns after 2-3 cycles while each new execution still introduces fresh information, this predicts a specific shape for the scaling curve. If review instead shows increasing returns (the reviewer accumulates understanding), the optimal strategy changes qualitatively.
4. **Budget-awareness enables qualitatively different strategies.** With a 30-step budget, the optimal strategy is generate-execute-submit. With 300 steps, it becomes plan-generate-execute-review-revise-retest-review. This echoes the "phase transition" finding of Xu et al. (2026).

### Contribution 4: Cross-Modal Generalization

Prior work studies individual modalities in isolation. We demonstrate that the proposer-reviewer pattern with grounded feedback works as a **unified architectural pattern** across five tasks spanning four distinct feedback modalities:

1. **Code generation** — execution feedback (Type 3a) + static analysis (Type 2): test pass/fail, error messages, lint warnings
2. **Web page generation** — visual feedback (Type 3b): rendered screenshots compared to design spec via VLM
3. **Slide generation** — visual feedback (Type 3b): rendered slide images evaluated for layout, density, readability
4. **Video editing** — temporal visual feedback (Type 3c): keyframe sequences analyzed for scene accuracy, temporal coherence, effect correctness
5. **Deep research** — factual verification feedback (Type 3d): independent fact-checking of each claim via search engine and knowledge base queries

The same architecture (proposer generates → environment provides grounded feedback → reviewer analyzes and structures feedback → proposer revises) applies across all five tasks. We measure whether the grounded feedback framework's predictions hold uniformly across modalities.

### Contribution 5: Internalizing Interaction Scaling via Distillation

Contributions 1–4 treat the agent loop as external scaffolding wrapped around a frozen model. We ask whether the scaffolding can be absorbed: can a small student (Qwen3-8B) distilled from multi-turn teacher trajectories (Claude Sonnet 4) learn to *check its own work* without an explicit outer loop?

We use a two-stage recipe:
1. **SFT on teacher trajectories.** Collect successful `[GENERATE] → [EXECUTE] → [REVIEW] → [SUBMIT]` traces from the teacher running the proposer-reviewer architecture on the training-split tasks. Fine-tune the student with QLoRA on these trajectories.
2. **GRPO with grounded rewards.** Apply Group Relative Policy Optimization with the same environmental feedback that served as the teacher's reviewer signal (test execution for code, VLM scoring for visual tasks, fact-check retrieval for research).

**Evaluation is strictly held-out.** The training-split tasks are disjoint from the Phase 1 benchmark tasks used for evaluation — same categories and difficulty, different instances. This isolates *learned behavior* from *memorization* and lets us answer:

- Does the student's one-shot pass rate exceed the base model's? (Does SFT transfer the priors and format?)
- Does the student's own interaction curve (pass rate vs. budget N=1,2,3,5) have positive slope? (Has it internalized when to iterate, not just how?)
- How much of the teacher's multi-turn ceiling does the student recover at matched budget? (Distillation efficiency.)
- Does GRPO improve over SFT, or does weak reward signal cause policy drift away from correct SFT answers? (Pathology that we observe on an in-distribution pilot.)

The claim is not that one-shot inference replaces multi-turn agents — a one-shot model has no test suite at inference. The claim is that interaction scaling has a *distillable component* (priors, format, self-critique habits) and a *non-distillable component* (the feedback itself), and that the ceiling of internalization is set by the quality of the environmental signal the teacher had access to.

---

## 3. Related Work

### 3.1 Self-Correction and Self-Review in LLMs

Huang et al. (ICLR 2024) demonstrated that LLMs cannot self-correct reasoning without external feedback — accuracy often decreases after self-review. Kamoi et al. (TACL 2025, "When Can LLMs Actually Correct Their Own Mistakes?") confirmed this across a broad range of settings: self-correction works only when (1) reliable external feedback is available, or (2) the model has been specifically fine-tuned for correction. Gou et al. (ICLR 2024, CRITIC) provided a controlled comparison: tool-grounded verification yields substantial gains, but removing the tool and relying on self-evaluation eliminates most improvement. Our work explains these findings through the grounded feedback framework and builds on them architecturally.

### 3.2 Test-Time Scaling

The foundational result from Snell et al. (ICLR 2025) showed that scaling test-time compute can be more effective than scaling model parameters for reasoning tasks. Subsequent work has explored multiple scaling strategies:

- **Reasoning scaling:** Chain-of-thought (Wei et al., 2022), tree-of-thought (Yao et al., 2023), self-consistency (Wang et al., 2023)
- **Sampling scaling:** Best-of-N, majority voting, reward-model reranking
- **Budget-aware scaling:** Jiang et al. (2025, BATS) introduced budget tracking for tool-augmented agents; BAVT (2026) proposed step-level value evaluation with budget-conditioned node selection; Xu et al. (2026, CATTS) use confidence-aware compute allocation for web agents

All of these operate primarily in the "Think" or "Sample" dimensions. Our work adds "Do" and "Review" as formalized components and studies their scaling properties.

### 3.3 Thinking vs. Doing

Shen et al. (NeurIPS 2025) introduced the distinction between scaling per-step reasoning ("thinking") and scaling environment interaction ("doing"), showing that "given a fixed token budget, acting longer yields larger gains than thinking longer" for web agents. Their TTI approach uses curriculum-based RL to train agents with adaptive rollout lengths. Our work extends this with "reviewing" as a third component, proposes an inference-time architectural pattern (proposer-reviewer) rather than a training approach, and generalizes beyond web browsing to code, visual, temporal, and fact-verification tasks.

### 3.4 Multi-Agent Architectures

The multi-agent debate literature (Du et al., 2023; Liang et al., 2023) shows that multiple LLMs debating can improve factual accuracy, but Tran & Kiela (2026) demonstrated that these gains collapse under equal compute budgets — a single agent is as good as debate when tokens are normalized. Cemri et al. (ICLR 2025, "Why Do Multi-Agent LLM Systems Fail?") catalogued 14 failure modes across 7 MAS frameworks, identifying inter-agent misalignment and task verification failure as fundamental challenges. Our work shows that multi-agent (proposer-reviewer) provides genuine architectural advantage specifically when the reviewer introduces Type 2/3 (externally grounded) feedback — the case where the DPI argument does not apply.

### 3.5 Feedback in Agentic Scaling

Ruan et al. (2025, IAD) is the most directly related work. It studies feedback's role in inference-time alignment through iterative agent decoding, demonstrating ~10% gains across Sketch2Code, Text2SQL, and WebShop. However, IAD is a single-agent approach, uses fixed allocation ratios, does not formalize feedback types, and does not study budget-optimal allocation. Our work provides the missing theoretical framework (why some feedback works), the missing optimization (how to allocate budget), and the missing architectural pattern (proposer-reviewer).

The Scaling Agentic Verifier (2026) uses a separate verifier agent for competitive coding — the closest existing work to our proposer-reviewer architecture. However, it is limited to competitive coding verification (finding counterexamples) and does not study budget allocation or cross-modal generalization.

### 3.6 Planner-Executor Architectures

Yang et al. (2025, Plan-and-Act) separate planning from execution, achieving SOTA on WebArena-Lite. Their key finding — weak planners are the critical bottleneck — informs our architecture design. Our proposer-reviewer extends the planner-executor pattern with an explicit review phase grounded in environment feedback, closing the loop that Plan-and-Act leaves open (plan → execute, but no structured review of execution results).

### 3.7 Visual Feedback for Code/Visual Generation

WebGen-Agent (2025) uses multi-level visual feedback (screenshots + VLM descriptions) for iterative web page refinement, improving accuracy from 26.4% to 51.9%. ScreenCoder (2025) decomposes front-end generation into grounding, planning, and generation with screenshot comparison. Vision-Guided Iterative Refinement (2026) uses a multimodal LLM as a visual critic. These works validate the effectiveness of visual feedback for specific tasks; we unify them under the interaction scaling framework and study their scaling properties systematically.

### 3.8 Factual Verification and Deep Research

Min et al. (2023, FActScore) introduced atomic fact decomposition and verification for long-form text generation. Wei et al. (2024, VERIFY) demonstrated iterative search-grounded verification for report generation. Recent deep research systems (OpenAI, 2025; Gemini, 2025) perform multi-step web research with iterative refinement but do not study the scaling properties of their verification loops. Our deep research task formalizes fact verification as Type 3d feedback and studies how budget allocation between research, writing, and fact-checking affects report quality.

---

## 4. Framing

### Paper Narrative (Introduction Flow)

**Opening (the problem):** LLM agents are increasingly used for artifact-producing tasks — writing code, generating web pages, creating presentations, editing videos, producing research reports. Test-time scaling has emerged as a powerful lever for improving these agents. But current scaling approaches focus overwhelmingly on "thinking more" — longer reasoning traces, more samples, deeper search trees.

**The gap:** We observe that for artifact-producing tasks, the most valuable test-time compute is not additional thinking but additional *interaction with the real world*. When a coding agent executes its code and sees test failures, it gains information that no amount of reasoning could have produced. When a web generation agent renders its HTML and sees a screenshot, it obtains visual feedback that was physically impossible to derive from the code alone. When a research agent fact-checks its claims against search results, it discovers errors invisible to pure reasoning. This "interaction compute" has fundamentally different scaling properties than reasoning compute.

**The framework:** We formalize this observation through an information-theoretic lens. We define feedback types by whether they provide externally grounded information (Type 0/1: no grounding; Type 2: static tool grounding; Type 3: dynamic environment grounding) and show that only grounded feedback can break through the information ceiling that bounds reasoning-only scaling. We formalize **Interaction Scaling** as a dimension of test-time compute that scales performance through cycles of environment interaction and structured review.

**The architecture:** The proposer-reviewer pattern is an effective primitive for interaction scaling. The proposer generates artifacts ("Think"); the environment provides grounded feedback ("Do"); the reviewer analyzes the results with multimodal feedback and provides structured revision guidance ("Review"). We study how to allocate a fixed budget across these three phases and demonstrate that budget-aware allocation significantly outperforms both fixed-ratio allocation and reasoning-only scaling.

**The results:** Across five tasks spanning four feedback modalities (execution, visual, temporal, factual verification), interaction scaling with budget-aware proposer-reviewer consistently outperforms reasoning-only, sampling-only, and naive interaction approaches. The gains are largest at moderate budgets (50-200 steps), and the optimal allocation strategy varies predictably by task modality.

---

## 5. Experimental Plan

### 5.1 Tasks and Benchmarks

We select five tasks that span four distinct feedback modalities:

| Task | Artifact | Grounded Feedback | Review Signal | Benchmark |
|---|---|---|---|---|
| Code generation | Python code | Test execution + static analysis | Pass/fail, errors, lint warnings, coverage | HumanEval+, MBPP+, SWE-bench Lite |
| Web page generation | HTML/CSS/JS | Browser rendering | Screenshot + VLM comparison to spec | Sketch2Code, Design2Code |
| Slide generation | Presentation (Slidev) | Slide rendering | Rendered slide images + VLM evaluation | Custom benchmark (100 slide specs with reference designs) |
| Video editing | Blender Python script | Keyframe rendering | Keyframe sequence + VLM temporal analysis | Custom benchmark (100 editing tasks with source videos) |
| Deep research | Research report | Search engine fact-checking | Per-claim verification verdicts + sources | FActScore-adapted benchmark (100 research topics) |

For the custom benchmarks:
- **Slide generation:** 100 slide generation specifications (academic papers, product briefs, tutorials) with reference designs. Evaluation via automated metrics (layout IoU, text overflow detection, visual similarity) and human evaluation (50-example subset, 3 annotators).
- **Video editing:** 100 editing tasks using Creative Commons source videos with natural language instructions (scene selection, trimming, effects, transitions). Evaluation via scene boundary accuracy (IoU), keyframe visual quality, temporal coherence metrics. Human evaluation for a subset.
- **Deep research:** 100 research topics across science, technology, history, and current events, with manually verified claim sets. Evaluation via FActScore (decompose report into atomic claims, verify each against authoritative sources), completeness (coverage of key aspects), and coherence.

All custom benchmarks will be publicly released. We seek external validation of evaluation criteria before running experiments.

### 5.2 Baselines

**B1 — Single-Agent, No Review (Think only):**
Standard generation with chain-of-thought. The agent generates the artifact in one shot, using all budget for reasoning and generation. No execution, no review.

**B2 — Single-Agent, Self-Review (Think + Type 0):**
The agent generates, then re-reads its output and self-critiques without external feedback. Budget split: 50% generation, 50% self-review and revision.

**B3 — Cross-Model Review (Think + Type 1):**
A separate LLM reviews the artifact and provides structured feedback, but without any external tool use or execution. The reviewer reads the code/HTML/script/report and critiques based solely on its own knowledge. This models the common "LLM code review" or "LLM editorial review" pattern. Budget split: 50% generation, 50% cross-model review and revision.

**B4 — Best-of-N Sampling (Sample):**
Generate $N$ independent candidates, select the best using a reward model or self-consistency. Pure sampling scaling.

**B5 — Single-Agent Agentic Loop (Do, no separate reviewer):**
A single agent generates, executes, reads output, and revises in a natural loop — the pattern used by most existing coding agents (Claude Code, Cursor, SWE-bench agents). No separate reviewer; the same agent processes execution feedback and decides on revisions. Budget is spent organically as the agent works. This is the critical baseline: it isolates whether architectural separation (proposer vs. reviewer) adds value beyond simply having environment feedback.

**B6 — IAD Reproduction (IAD baseline):**
Reproduce the Iterative Agent Decoding approach (Ruan et al., 2025) with fixed $N=K$ ratios.

**Ours — Budget-Aware Proposer-Reviewer (Think + Do + Review, optimized):**
Proposer generates → environment provides grounded feedback → reviewer analyzes with structured evaluation → proposer revises. Budget allocation is dynamic, optimized through the mechanisms described in 5.4. The key architectural distinction from B5: the reviewer is a separate agent with its own context, receiving only the latest artifact and grounded feedback (not the full generation history), enabling more focused evaluation and preventing context overload.

### 5.3 Core Experiments

**Experiment 1: Feedback Type Ablation**

Hold the architecture constant (proposer-reviewer with budget $B$ = 200K tokens) and vary the feedback type:
- Type 0: Reviewer sees only the generated artifact (no execution)
- Type 1: A different LLM critiques without execution or tools
- Type 2: Reviewer receives static analysis output (lint warnings, type errors) but no execution
- Type 3a: Reviewer receives execution results (test pass/fail, error messages)
- Type 3b: Reviewer receives rendered visual output (screenshot)
- Type 3c: Reviewer receives temporal output (keyframe sequence)
- Type 3d: Reviewer receives fact-verification results (search engine verdicts)
- Type 2+3: Reviewer receives both static analysis and execution/rendering results

Prediction: Type 3 >> Type 2 > Type 1 $\approx$ Type 0. Type 2+3 $\geq$ Type 3 alone (static analysis catches issues complementary to execution). The gap is largest at smaller budgets where each feedback cycle's grounded information matters most.

This directly tests Contribution 1 (the grounded feedback framework).

**Experiment 2: Scaling Curves by Dimension**

Fix total budget $B \in \{50K, 100K, 200K, 500K, 1M, 2M\}$ tokens and measure performance for:
- Reasoning-only scaling (all budget to chain-of-thought, self-consistency, extended thinking)
- Sampling-only scaling (all budget to best-of-N)
- Single-agent interaction (B5: one agent with environment feedback)
- Proposer-reviewer interaction (ours: separate proposer and reviewer with environment feedback)
- Combined (optimized allocation across reasoning, interaction, and review)

Plot performance vs. $B$ for each strategy and each task.

Prediction: Both forms of interaction scaling dominate reasoning and sampling scaling for artifact-producing tasks. The gap widens at moderate budgets (200K-1M tokens) where interaction scaling enables qualitatively different strategies (planning + testing + review), while reasoning scaling saturates. The proposer-reviewer separation provides additional gains over the single-agent loop, especially for tasks where context management is critical (video editing, deep research).

This directly tests Contribution 2 and the value of architectural separation.

**Experiment 3: Budget Allocation Optimization**

For fixed $B$ = 500K tokens, sweep the allocation ratio across the proposer-reviewer system:
- Vary $b_1$ (proposal tokens) from 10% to 80%
- Vary $b_2$ (environment processing tokens) from 10% to 80%
- Vary $b_3$ (review tokens) from 10% to 80%
- Subject to $b_1 + b_2 + b_3 = B$

Plot performance as a heatmap over the allocation simplex. Identify the optimal allocation for each task.

Then test adaptive allocation: a meta-controller that adjusts ratios dynamically based on:
- Iteration number (explore early, exploit late)
- Current quality estimate (derived from pass rate or visual similarity)
- Remaining budget

Compare adaptive vs. fixed-optimal vs. equal allocation.

This directly tests Contribution 3 (budget-aware allocation).

**Experiment 4: Cross-Modal Generalization**

Run the full proposer-reviewer system with Type 3 feedback on all five tasks. Compare:
- Task-specific hand-tuned reviewer prompts
- Generic reviewer prompt ("analyze the grounded feedback and provide structured improvement suggestions")
- Does the same architectural pattern work across code, web, slides, video, and research?

This directly tests Contribution 4 (cross-modal generalization).

**Experiment 5: Verification Gap Analysis**

Inspired by General AgentBench's "verification gap" finding:
- For each task, generate $K$ candidates with the proposer
- Measure pass@$K$ (best candidate exists among $K$)
- Measure choose@$K$ with self-selection (proposer picks best — Type 0)
- Measure choose@$K$ with cross-model selection (different LLM picks — Type 1)
- Measure choose@$K$ with grounded reviewer selection (Type 3)
- Quantify how much of the verification gap each approach closes

Prediction: The reviewer with Type 3 feedback closes significantly more of the verification gap than self-selection or cross-model selection. The gap is largest for visual tasks (where "code looks right" vs. "rendering looks right" diverge most) and deep research (where "claim sounds right" vs. "claim is factually correct" diverge most).

### 5.4 Budget-Aware Allocation Mechanism

We propose and compare three allocation strategies:

**Strategy A — Fixed Ratio:**
Pre-determined split (e.g., 40% propose, 30% execute, 30% review). Baseline approach.

**Strategy B — Phase-Adaptive:**
Rule-based adaptation:
- Round 1: Heavy on proposal (60/20/20) — establish initial solution
- Round 2+: Heavy on review (20/40/40) — identify and fix issues
- Final round: Heavy on execution (20/60/20) — thorough testing/verification

**Strategy C — Confidence-Conditioned:**
Use the reviewer's confidence score (derived from execution pass rate, visual similarity metric, or fact-check accuracy) to dynamically allocate:
- Low confidence → more review cycles (need to understand what's wrong)
- Medium confidence → more execution (need to verify fixes)
- High confidence → stop early (save budget, solution is good)

The meta-controller is a simple rule-based system (not a learned model), to keep the contribution focused on the architectural pattern rather than the optimization algorithm. We leave learned allocation as future work.

### 5.5 Models

- **Proposer:** Claude Sonnet 4 (primary), GPT-4.1 (secondary, for generalization)
- **Reviewer:** Claude Sonnet 4 with multimodal input (screenshots + text)
- **Ablation:** Test with weaker reviewer models (Haiku 4) to study whether the reviewer needs to be as capable as the proposer

We use models from two different providers to mitigate perception of vendor bias. All results are reported for both model families.

### 5.6 Evaluation Metrics

| Task | Primary Metric | Secondary Metrics |
|---|---|---|
| Code generation | pass@1 (test execution) | Lint pass rate, coverage, code quality |
| Web page generation | Visual similarity (SSIM + CLIP) | Structural accuracy (DOM comparison), layout IoU |
| Slide generation | Visual fidelity (VLM score) | Content accuracy, layout consistency, text overflow rate |
| Video editing | Scene boundary IoU + keyframe quality | Temporal coherence, effect correctness, audio-visual sync |
| Deep research | FActScore (claim-level accuracy) | Coverage (key aspects addressed), coherence, source quality |

All experiments report:
- Performance vs. total budget $B$ in tokens (scaling curves)
- Performance vs. allocation ratio (allocation sensitivity)
- Wall-clock time and API dollar cost (practical efficiency)
- Token-efficiency: performance gain per 100K tokens spent

### 5.7 Ablation Studies

1. **Reviewer architecture:** Separate reviewer agent vs. reviewer-as-tool (reviewer is a function call within the proposer's loop) — does architectural separation matter? (Connect to B5 comparison)
2. **Feedback granularity:** Full execution output vs. pass/fail only vs. error message only — how much feedback detail is needed?
3. **Number of review cycles:** 1 vs. 2 vs. 3 vs. adaptive — where do diminishing returns kick in?
4. **Proposer-reviewer model asymmetry:** Strong proposer + weak reviewer vs. weak proposer + strong reviewer — which matters more? (Connect to Yang et al.'s "weak planner" finding)
5. **Cross-modal review transfer:** Can a code reviewer help with web generation and vice versa? How modality-specific must the reviewer be?
6. **Static + dynamic feedback interaction:** For code generation, does adding linting (Type 2) on top of test execution (Type 3a) provide additive gains, or is execution alone sufficient?

### 5.8 Compute Cost Estimate

| Component | Estimated token cost | Estimated API cost |
|---|---|---|
| Experiment 1 (feedback ablation): 8 conditions × 5 tasks × ~100 problems × 200K tokens | ~80B tokens | ~$8K |
| Experiment 2 (scaling curves): 6 budgets × 5 strategies × 5 tasks × ~100 problems | ~150B tokens | ~$15K |
| Experiment 3 (allocation sweep): ~30 allocation points × 5 tasks × ~100 problems × 500K tokens | ~75B tokens | ~$7.5K |
| Experiments 4-5 + ablations | ~50B tokens | ~$5K |
| **Total estimate** | **~350B tokens** | **~$35K** |

This estimate assumes Claude Sonnet pricing. Actual costs may be lower with prompt caching (applicable to shared system prompts and repeated benchmark problems). We budget $50K total to account for debugging runs, retries, and additional ablations.

If budget is constrained, we prioritize: Experiments 1 > 2 > 5 > 3 > 4, and reduce custom benchmark sizes to 50 examples each.

---

## 6. Expected Results and Story

### The scaling curve story

We expect to produce a figure showing scaling curves (performance vs. total budget $B$ in tokens) for each task:

```
Performance
    |          ___________  <- Proposer-Reviewer + budget-aware (ours)
    |        /
    |      /    ________   <- Single-agent agentic loop (B5)
    |    /    /
    |   /   /   ______    <- Sampling Scaling (best-of-N)
    |  /  /   /
    | / /   /  ______     <- Reasoning Scaling (more CoT)
    |//   /  /
    |   /  /
    |  / /
    | //
    |/
    +-------------------------> Budget B (tokens)
```

Key narrative: Reasoning scaling saturates early (the model can't think its way to correctness without seeing execution results). Sampling scaling is more efficient but still limited (generating more candidates doesn't help if you can't evaluate them reliably). The single-agent agentic loop helps significantly (executing and observing is valuable). Budget-aware proposer-reviewer helps most (architectural separation enables focused review without context overload, and smart allocation maximizes information gain per token).

The gap between the single-agent loop (B5) and the proposer-reviewer (ours) is a key novel finding: it quantifies the value of architectural separation beyond just having environment feedback.

### The allocation heatmap story

For code generation, we expect the optimal allocation to favor execution (run many tests):
- Optimal: ~30% propose, ~45% execute, ~25% review

For visual generation (web pages, slides), we expect the optimal allocation to favor review (VLM analysis is expensive but critical):
- Optimal: ~35% propose, ~25% execute (rendering is fast), ~40% review

For video editing, we expect heavier allocation to execution (coarse-to-fine scene localization is multi-step):
- Optimal: ~25% propose, ~50% execute (keyframe extraction + rendering), ~25% review

For deep research, we expect the optimal allocation to favor review (fact-checking is the bottleneck):
- Optimal: ~30% propose (writing), ~25% execute (search queries), ~45% review (fact verification)

This modality-dependent optimal allocation is itself a finding: there is no one-size-fits-all ratio, and budget-aware adaptation is necessary.

### The grounded feedback story

We expect the feedback type ablation to show a clear hierarchy:
- Type 3 (dynamic grounded) >> Type 2 (static grounded) > Type 1 (cross-model) $\approx$ Type 0 (self-review)
- With Type 2+3 providing modest gains over Type 3 alone (static analysis catches complementary issues)
- With the gap largest at small budgets (where each feedback cycle's information content matters most)

This validates the grounded feedback framework as a predictive theory, not just a post-hoc explanation.

---

## 7. Paper Outline

1. **Introduction** (1.5 pages) — Problem, gap, thesis, contributions. Frames the paper as: agents that check their work do so by querying an environment, and that environmental feedback loop both drives test-time scaling and is partly distillable into a student model.
2. **Background and Related Work** (2 pages)
   - 2.1 Test-time scaling: reasoning, sampling, and interaction
   - 2.2 Self-correction, multi-agent debate, and their limitations
   - 2.3 Environment feedback in prior work (RLEF, WebGen-Agent, CRITIC, deep research)
   - 2.4 Agent distillation and on-policy RL (trajectory-level SFT, GRPO, budget-conditioning)
3. **The Grounded Feedback Framework** (1.5 pages)
   - 3.1 Feedback taxonomy (Type 0/1/2/3)
   - 3.2 Information-theoretic analysis (graphical model, DPI argument)
   - 3.3 Predictions derived from the framework
4. **Method: Budget-Aware Proposer-Reviewer** (1.5 pages)
   - 4.1 Architecture (proposer, environment, reviewer)
   - 4.2 Budget definition and allocation strategies
   - 4.3 Cross-modal instantiation (code, web, slides, video, research)
5. **Experiments: Scaling with External Scaffolding** (2.5 pages)
   - 5.1 Feedback type ablation (validates framework)
   - 5.2 Scaling curves by dimension (validates interaction scaling)
   - 5.3 Budget allocation optimization (validates budget awareness)
   - 5.4 Cross-modal generalization (validates universality)
   - 5.5 Verification gap analysis
6. **From Scaffolding to Capability: Distillation and Internalization** (2 pages)
   - 6.1 Training setup: teacher-student distillation with held-out splits
     - Training split: 60–100 new bug-fix tasks per modality, disjoint from evaluation
     - Teacher: frontier model running the Section 4 proposer-reviewer loop
     - Student: Qwen3-8B with QLoRA, SFT on teacher trajectories, then GRPO with grounded rewards
   - 6.2 Evaluation matrix: {base, SFT, GRPO, teacher} × {N=1, 2, 3, 5 interaction turns} on the held-out benchmark
   - 6.3 Four quantitative questions
     - Does the student's one-shot pass rate exceed the base model's? (Priors + format)
     - Does the student's interaction curve have positive slope? (Policy, not just format)
     - How much of the teacher's multi-turn performance does the student recover at matched budget? (Distillation efficiency)
     - Does GRPO improve over SFT or cause policy drift under weak reward signal?
   - 6.4 What distills and what doesn't — the bounded-internalization result
7. **Analysis and Discussion** (1 page)
   - 7.1 When does interaction scaling fail? (Noisy feedback channels, unverifiable tasks, saturated budgets)
   - 7.2 The value of architectural separation (single-agent loop vs. proposer-reviewer)
   - 7.3 Why some scaling distills and some doesn't — environment signal quality as the load-bearing variable
   - 7.4 Connection to human work patterns (drafting, testing, revising)
8. **Conclusion** (0.5 page)
9. **Appendix** — Ablation studies, prompt templates, custom benchmark details, distillation hyperparameters, held-out-split construction, compute costs

Target: 10 pages + references + appendix.

---

## 8. Key Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Type 3 feedback doesn't consistently beat Type 2 across all tasks | Low (strong prior evidence from RLEF, WebGen-Agent, CRITIC) | If some tasks show weaker gaps, characterize boundary conditions — this is itself a finding |
| Single-agent loop (B5) matches proposer-reviewer performance | Medium | If true, the contribution shifts to: the grounded feedback framework and scaling analysis remain valid; the architectural separation finding becomes "separation helps only for context-heavy tasks" |
| Budget allocation shows flat sensitivity (any reasonable ratio works) | Medium | If true, reframe: "interaction scaling is robust to allocation" is also useful; focus contribution on framework and cross-modal generalization |
| Custom benchmarks (slides, video, research) criticized for quality | Medium | 100 examples each (not 50); public release; external validation; use FActScore (established metric) for deep research |
| Reviewers argue this is "just engineering" not research | Low-Medium | The information-theoretic framework and scaling analysis provide the conceptual contribution; the graphical model formalization makes specific, testable predictions |
| A concurrent paper publishes similar results | Medium (active field) | Move fast; differentiate on the unified framework across 4 feedback modalities (most concurrent work is modality-specific) |
| Compute budget insufficient for full experimental plan | Medium | Prioritized experiment ordering (1 > 2 > 5 > 3 > 4); prompt caching; reduced benchmark sizes as fallback |

---

## 9. Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Literature deep-dive and framework formalization | 2 weeks | Section 2-3 draft, formal definitions, graphical model |
| Infrastructure: proposer-reviewer for code + web | 3 weeks | Working system for code generation and web page generation |
| Infrastructure: slides + video + deep research | 4 weeks | Working system for all 5 modalities |
| Custom benchmark construction + external validation | 3 weeks | Slide, video, and deep research benchmarks (100 examples each) |
| Experiment 1 (feedback type ablation) | 2 weeks | Key framework validation |
| Experiment 2 (scaling curves) | 3 weeks | Core scaling results |
| Experiment 3 (budget allocation) | 2 weeks | Allocation optimization results |
| Experiments 4-5 + ablations | 3 weeks | Cross-modal + verification gap + ablation studies |
| Writing and iteration | 3 weeks | Complete paper draft |
| **Total** | **~25 weeks** | Submission-ready paper |

Note: Phases overlap where possible (e.g., benchmark construction runs in parallel with infrastructure for later modalities; writing begins during final experiments). Critical path is ~20 weeks with ideal parallelization.

---

## 10. References

### Self-Correction and Review
- Huang et al. "Large Language Models Cannot Self-Correct Reasoning Yet." ICLR 2024. https://arxiv.org/abs/2310.01798
- Kamoi et al. "When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey." TACL 2025. https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713
- Gou et al. "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing." ICLR 2024. https://arxiv.org/abs/2305.11738

### Test-Time Scaling
- Snell et al. "Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Parameters." ICLR 2025.
- Shen et al. "Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction." NeurIPS 2025. https://arxiv.org/abs/2506.07976
- Ruan et al. "On the Role of Feedback in Test-Time Scaling of Agentic AI Workflows (IAD)." 2025. https://arxiv.org/abs/2504.01931

### Budget-Aware Agents
- Jiang et al. "Budget-Aware Tool-Use Enables Effective Agent Scaling (BATS)." 2025. https://arxiv.org/abs/2511.17006
- "Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents (BAVT)." 2026. https://arxiv.org/abs/2603.12634
- Xu et al. "Phase Transition for Budgeted Multi-Agent Synergy." 2026. https://arxiv.org/abs/2601.17311

### Multi-Agent Systems
- Tran & Kiela. "Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets." 2026. https://arxiv.org/abs/2604.02460
- Cemri et al. "Why Do Multi-Agent LLM Systems Fail?" ICLR 2025. https://arxiv.org/abs/2503.13657
- "MAR: Multi-Agent Reflexion Improves Reasoning Abilities in LLMs." 2025. https://arxiv.org/abs/2512.20845

### Planner-Executor
- Yang et al. "Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks." 2025. https://arxiv.org/abs/2503.09572

### Environment Feedback for Code/Visual Generation
- Gehring et al. "RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning." ICML 2025. https://arxiv.org/abs/2410.02089
- "WebGen-Agent: Enhancing Interactive Website Generation with Multi-Level Feedback." 2025. https://arxiv.org/abs/2509.22644
- "ScreenCoder: Advancing Visual-to-Code Generation for Front-End Automation." 2025. https://arxiv.org/abs/2507.22827
- "Scaling Agentic Verifier for Competitive Coding." 2026. https://arxiv.org/abs/2602.04254
- "Vision-Guided Iterative Refinement for Frontend Code Generation." 2026. https://arxiv.org/abs/2604.05839

### Agent Benchmarking
- "Benchmark Test-Time Scaling of General LLM Agents (General AgentBench)." 2026. https://arxiv.org/abs/2602.18998
- Xu et al. "Agentic Test-Time Scaling for WebAgents (CATTS)." 2026. https://arxiv.org/abs/2602.12276

### Factual Verification
- Min et al. "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation." EMNLP 2023. https://arxiv.org/abs/2305.14251
