# Experiment Log — Hard Benchmarks

## All 6 Categories Have Results

| Category | Tasks | Feedback | Single-shot | Reviewed | Delta |
|----------|-------|----------|-------------|----------|-------|
| **Code (SWE)** | 15 | Type 3a: Execution | 67% pass | **100%** pass | **+33pp** |
| **Video editing** | 5 | Type 3c: Temporal+Exec | 0.24 quality | **0.58** quality | **+0.34** |
| **Research** | 5 | Type 3d: Factual | 0.76 accuracy | **0.92** accuracy | **+0.16** |
| **Web pages** | 2 | Type 3b: Visual | 0.57 quality | **0.72** quality | **+0.15** |
| **Slides** | 5* | Type 3b: Visual | 0.76 quality | **0.85** quality | **+0.09** |
| **Animations** | 8 | Type 3c: Temporal | 0.29 quality | **0.37** quality | **+0.07** |

*Slides: 5 tasks with correct prompt (batch 2 pending for tasks 6-10)

## Key Findings

### 1. Interaction Scaling Works Across All Modalities
Every single category shows improvement from grounded feedback. The
improvement ranges from +0.07 (animations) to +33pp (code), but ALL
are positive.

### 2. Feedback Actionability Determines Magnitude
```
Execution (+33pp) >> Video keyframes (+0.34) > Factual (+0.16)
> Visual (+0.09-0.15) > Animation frames (+0.07)
```

### 3. Video Editing Is a Strong New Finding
Video editing shows +0.34 quality improvement — the second-largest
delta after code. This is because video feedback combines execution
errors (code crashes) with visual verification (keyframes), getting
the best of both worlds.

Specific wins:
- video_003 (loop extraction): 0.00 → 1.00 — completely broken code
  fixed by seeing the execution error then keyframe verification
- video_004 (speed change): 0.00 → 0.70 — execution error fixed,
  then keyframe review caught timing issue

### 4. Some Tasks Resist Review
- video_002 (reverse video): review couldn't fix (corrupted output)
- video_005 (rotated watermark): ffmpeg drawtext rotation too tricky
- research_003 (GDP rankings): too many interconnected facts
- Animations generally show smallest gains — temporal bugs are
  hardest to diagnose from static frame screenshots

## Configuration
- Model: Claude Sonnet 4 (claude-sonnet-4-20250514), temperature 0.0
- Budget: 500K tokens per problem
- Max iterations: 5 (code), 3 (visual/video), 2 (research)
- Visual review: VLM via Anthropic vision API
- Video: moviepy/ffmpeg + keyframe extraction + VLM
- Code: sandboxed subprocess execution
