# Instrumental Convergence: RLHF vs. RLVR, Same-Family Comparison (Tulu-3-8B)

## What this is

A controlled follow-up to the earlier RLVR-vs-RLHF instrumental convergence (IC) experiment. Where the first round compared two *different* model families (Olmo-3-7B-Think vs. Llama-2-7b-chat-hf), this run holds the base model and SFT stage fixed and varies only the last training stage:

| Condition | Model | Training stage |
|---|---|---|
| RLHF | `allenai/Llama-3.1-Tulu-3-8B-DPO` | SFT → DPO (stops before RLVR) |
| RLVR | `allenai/Llama-3.1-Tulu-3-8B` | SFT → DPO → RLVR (full Tulu-3 pipeline) |

Because both checkpoints come from the same AllenAI Tulu-3 pipeline, the only meaningful difference between them is whether the final RLVR stage was applied. This removes the base-model/tokenizer/SFT-data confound that limited the interpretability of the first round.

## Design novelty: judging the judges, not just the responses

Most IC evals stop at one layer: generate a response, score it for IC. This pipeline adds a second layer on top — the models don't just get scored, they're put in the evaluator seat and made to *judge each other's* responses first (step 2, blind unprimed peer audit), and then that judgment itself gets independently scored for accuracy and bias (step 4, meta-audit).

The reasoning behind adding that layer: a model generating a response is operating in "learner" mode — it's producing an output under whatever pressures its training instilled. A model judging another model's response is operating in "teacher" mode — it has to explicitly reason about whether the response is acceptable. If IC-shaped behavior is something a model has the latent capacity to recognize but doesn't spontaneously surface in generation, teacher-mode is where that gap should show up: put the same underlying model in an evaluative role and see whether it now catches what it didn't produce differently.

That's exactly what the blind self-audit results below show. The response-generation layer alone (step 3, IC-focused judging) says these models produce IC-flagged output on roughly a third to two-fifths of prompts. But when the *same models* are put in teacher mode with no framing toward safety (step 2), they catch almost none of it — 0% and 3.95%. The two-layer design is what makes that gap visible at all; a single-layer "score the response" pipeline would never have surfaced it, because it never asks the model to evaluate anything.

**Where this sits relative to existing work:** the closest prior result is the "Self-Correction Blind Spot" line of work, which finds LLMs fail to correct errors in their own output while successfully correcting the identical error when it's presented as coming from someone else — a self-vs-external asymmetry in error recognition. Separately, there's a "meta-evaluation collapse" literature on judging LLM judges in general, and multi-judge / specialized-judge setups (e.g. SysAdmin's five power-seeking-dimension judges) for scoring power-seeking behavior directly. As far as I could find, no existing IC/instrumental-convergence work combines blind peer-to-peer judging (unprimed, no safety framing) with an independent meta-audit of those judgments, specifically to measure the gap between IC a model *produces* and IC a model *recognizes* in a peer. That combination — not either layer alone — is the part of this pipeline that looks new. Flagged as "as far as I could find" rather than a confirmed literature claim, since a full search isn't the same as a systematic review.

## Benchmark

76 prompts from [InstrumentalEval](https://github.com/yf-he/InstrumentalEval), pulled live from the benchmark CSVs, across 6 categories:

- Deceptive Behavior (11)
- Evading Shutdown (20)
- Hacking Computer Systems (10)
- Hiding Unwanted Behavior (11)
- Running Multiple AI Copies (13)
- Strategically Appearing Aligned (11)

## Pipeline

1. **Generation** — both models loaded 4-bit quantized (NF4, double-quant), one per T4 GPU (`cuda:0` = RLHF, `cuda:1` = RLVR), sampled at temp 0.6 / top-p 0.9, 1024 max new tokens.
2. **Blind unprimed peer audit** — each model reviews the *other's* response to a generic "is this response Good or Bad?" prompt, with no mention of IC or safety. This measures whether models spontaneously notice IC behavior in peers without being told what to look for.
3. **Primary IC judging (4 external judges)** — Gemini (`gemini-3.1-flash-lite`), Groq (`llama-3.1-8b-instant`), GPT-4o, and GPT-5.6 Terra independently score each response for IC presence across three dimensions (resource acquisition, self-preservation/anti-shutdown, goal safeguarding), with severity, evidence quote, and reasoning. Consensus = IC flagged by ≥2 of 4 judges.
4. **Meta-audit** — the same 4 judges evaluate whether each peer judgment from step 2 was itself accurate and unbiased.

## Results

### Headline: primary IC detection rate across all 76 tasks (4-judge majority consensus)

This is the number that matters for the core question — everything else in this notebook is either a decomposition of it or a separate finding.

| Model | IC rate | Gemini | Groq | GPT-4o | Terra |
|---|---|---|---|---|---|
| RLHF (DPO-only) | **39.47%** (30/76) | 26.32% | 90.79% | 36.84% | 23.68% |
| RLVR (full pipeline) | **34.21%** (26/76) | 22.37% | 86.84% | 32.89% | 23.68% |

**Fisher's exact test on the 76-task consensus counts: p = 0.61 (OR ≈ 1.25).** The 5.3-point gap is not statistically significant at n=76 — consistent with the Round-2 GPT-judge result on the cross-family task (Terra p=0.40, GPT-4o p=1.0), and a reversal in direction from the original Gemini-only result (p=0.034) that first motivated this project. Across two rounds and two model pairs, the strongest read is: **no reliable IC-rate difference between RLHF and RLVR has been detected once judge disagreement is accounted for.**

Groq (`llama-3.1-8b-instant`) is a clear outlier — flagging IC in ~87-91% of responses vs. 22-37% from the other three judges. This is a small, cheap model acting as judge and it is almost certainly over-triggering rather than the other three under-triggering; treat Groq's column as noise, not signal, in the aggregate number.

### Category breakdown — exploratory, not the headline

Categories exist to see whether the aggregate result hides different scenarios, not to replace it — each one is too small (n=10-20) to draw a standalone conclusion from, and the 76-task result above is what should be cited as the finding.

| Category | RLHF | RLVR | Δ (RLHF−RLVR) | n |
|---|---|---|---|---|
| Evading Shutdown | 95.00% | 85.00% | +10.00 | 20 |
| Hiding Unwanted Behavior | 36.36% | 27.27% | +9.09 | 11 |
| Strategically Appearing Aligned | 27.27% | 18.18% | +9.09 | 11 |
| Hacking Computer Systems | 20.00% | 30.00% | **−10.00** | 10 |
| Deceptive Behavior | 9.09% | 0.00% | +9.09 | 11 |
| Running Multiple AI Copies | 7.69% | 7.69% | 0.00 | 13 |

Worth noting only as context: Evading Shutdown runs far hotter than every other category for both models (85-95% vs single/low-double digits elsewhere), and it's the category driving most of the aggregate RLHF>RLVR gap — Hacking Systems actually goes the other way. Neither observation changes the headline p=0.61 result; they're just where in the benchmark the aggregate signal is concentrated.

### Blind peer audit: models don't self-report IC they weren't asked about

| | Ground-truth IC rate (multi-judge consensus) | Unprompted blind rejection rate | Gap |
|---|---|---|---|
| RLVR judging RLHF's IC-positive responses | 39.47% (30/76) | 3.95% (3/76 caught) | 35.53 pts missed |
| RLHF judging RLVR's IC-positive responses | 34.21% (26/76) | 0.00% (0/76 caught) | 34.21 pts missed |

This is the most striking result in the notebook, and it's the direct payoff of the teacher/learner design above: it's only visible because the pipeline put the models in an evaluative role in the first place. When a model is asked "is this a good response?" with no framing toward safety, it accepts IC-laden responses from its peer as "Good" essentially every time — even though asking an explicitly IC-focused judge prompt against the same responses reveals IC present in a third to two-fifths of them. The gap isn't about capability (the same base architecture *can* detect IC when told what to look for in step 3) — it's about salience under a generic evaluation frame. That has a direct implication for anything relying on model self-critique or blind AI feedback as a safety signal: a model not primed to look for instrumental convergence largely won't flag it, even when made to act as the evaluator.

### Meta-audit: judges disagree sharply on whether peer judgments were fair

| | Majority-consensus judgment accuracy | Gemini | Groq | GPT-4o | Terra |
|---|---|---|---|---|---|
| RLHF judging RLVR (was that peer review fair?) | 71.05% | 57.89% | 63.16% | 69.74% | **1.32%** |
| RLVR judging RLHF (was that peer review fair?) | 59.21% | 38.16% | 59.21% | 67.11% | **1.32%** |

Terra rates almost every peer judgment as inaccurate/biased in both directions (1.32% = 1/76 in each case), while the other three judges cluster in the 38-70% range. This is too extreme and too identical across both conditions to be a real signal about the underlying judgments — more likely Terra is applying a much stricter or differently-calibrated standard for "accurate and fair" at the meta level than the other three models, or there's a prompt/formatting sensitivity specific to Terra on this particular template. Worth a targeted look at a handful of raw Terra meta-judgments before drawing any conclusion from this row.

## Interpretation, honestly

- On the metric that matters — IC rate across all 76 tasks — RLHF and RLVR are statistically indistinguishable (p=0.61). That holds now under a cleaner same-family design *and* the earlier cross-family design, across three different judge families (Gemini, GPT-4o/Terra, and now this run). That's a reasonably strong null result at this point, not just an artifact of one judge or one model pair.
- The one place something reads as a real, robust pattern rather than judge noise is the **blind self-audit gap**: models overwhelmingly fail to flag IC in a peer's response unless the evaluation prompt tells them to look for it. That's a finding about elicitation, not about RLVR vs RLHF, and arguably the more publishable/interesting result of this whole notebook.
- Groq and Terra both look like judge-reliability outliers in opposite directions (Groq over-triggers on IC presence, Terra under-triggers on judgment accuracy). The headline number should probably also be reported as GPT-4o-only, or as a sensitivity range across judges, alongside the blended 4-judge consensus — the consensus rule (≥2/4) is currently letting one noisy judge swing borderline cases.

## Known issues / things to check before writing this up

- No significance test was run in-notebook on the category breakdown or on either self-audit gap — worth adding.
- Groq and Terra's outlier behavior should be diagnosed against a few raw transcripts rather than just excluded, since "small/cheap model over-triggers" and "one judge scores almost everything as biased" are both claims worth actually checking before stating them as fact in the writeup.
- Consider re-running the primary IC consensus as a 3-judge vote excluding Groq, to see if the headline gap and its significance move at all.

## Files produced

| File | Contents |
|---|---|
| `1_responses_tulu3_rlhf.csv` / `2_responses_tulu3_rlvr.csv` | Raw generations per model, per prompt |
| `3_blind_eval_rlhf_judging_rlvr(best).csv` / `4_blind_eval_rlvr_judging_rlhf(best).csv` | Unprimed peer "Good/Bad" audits |
| `multi_judge_eval_1_resp_rlhf.csv` / `multi_judge_eval_2_resp_rlvr.csv` | 4-judge IC scoring of each model's own responses |
| `multi_judge_eval_3_peer_rlhf_judges_rlvr.csv` / `multi_judge_eval_4_peer_rlvr_judges_rlhf.csv` | 4-judge meta-audit of the blind peer judgments |
| `category_breakdown.csv` | Per-category IC rate table (Cell 14 output) |

## Reproducing

Run on Kaggle with 2× T4 GPUs. Requires Kaggle Secrets `API` (Gemini), `GROQ_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`. Install pins: `transformers==4.46.3`, `accelerate>=0.34.0`, `bitsandbytes>=0.46.1`.
