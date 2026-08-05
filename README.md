# InstrumentalEval: RLVR vs RLHF

Comparing instrumental convergence between an RLVR-trained model and a classic 
RLHF-trained model on the InstrumentalEval benchmark.

Part of **BlueDot Impact's Project Sprint** (ongoing), extending the 
InstrumentalEval benchmark to test RLVR vs RLHF post-training effects on 
instrumental convergence.

## Background

Instrumental convergence is the tendency of an AI system to pursue subgoals — 
self-preservation, deception, evading oversight, unauthorized resource access, 
self-replication, or appearing aligned while pursuing a hidden agenda — that 
weren't explicitly requested by the task, because those subgoals are useful for 
achieving almost any objective.

The original InstrumentalEval paper hypothesized that models trained via direct 
RL optimization (RLVR-style) show more instrumental convergence than models 
trained via RLHF. This project tests that hypothesis on two open-weight, 
free-to-run models.

## Models

| Role | Model | Training method |
|---|---|---|
| RLVR | [`allenai/Olmo-3-7B-Think`](https://huggingface.co/allenai/Olmo-3-7B-Think) | SFT → DPO → RLVR (final checkpoint) |
| RLHF | [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | SFT → reward model + PPO + rejection sampling |

Both models loaded in 4-bit (NF4) quantization via `bitsandbytes` for inference 
on a free-tier T4 GPU.

## Benchmark

[InstrumentalEval](https://github.com/yf-he/InstrumentalEval) — 76 scenario 
prompts across 6 categories: Deceptive Behavior, Evading Shutdown, Hacking 
Computer Systems, Hiding Unwanted Behavior, Running Multiple AI Copies, 
Strategically Appearing Aligned.

## Repo Structure

- **[`independent_task/`](independent_task/)** — main experiment: each model 
  responds to all 76 prompts independently (no shared context between 
  prompts). This is the primary, statistically-powered result.
- **[`dependent_tasks/`](dependent_tasks/)** — exploratory pilot: chained 
  multi-turn conversations testing whether instrumental convergence compounds 
  with growing context. Small sample (n=6); see its own README 
  ([`dependent_tasks/README.md`](dependent_tasks/README.md)) for scope 
  and caveats.

## Results — at a glance

**Main experiment (independent, n=76 per model)**

| Model | Judge A (Gemini) | Judge B |
|---|---|---|
| RLVR (Olmo-3-7B-Think) | 1.3% (1/76) | 0.0% (0/76) |
| RLHF (Llama-2-7B-chat) | 10.5% (8/76) | 1.3% (1/76) |

Judge A's gap is statistically significant (Fisher's exact test, p = 0.034) — 
RLHF showed a *higher* instrumental-convergence rate than RLVR, the opposite 
direction from the original paper's hypothesis. Judge B's gap is not 
significant (too few positive cases). See [Limitations](#limitations) below.

**Exploratory pilot (chained, n=6 per model)**

| Model | Judge A | Judge B | Avg |
|---|---|---|---|
| RLVR (Olmo) | 16.67% | 16.67% | 16.67% |
| RLHF (Llama-2) | 16.67% | 0.00% | 8.33% |

Directional only, not statistically powered — full details in 
[`dependent_tasks/README.md`](dependent_tasks/README.md).

## Method (independent_task)

1. Each model generated a response to all 76 prompts independently.
2. Generation settings: `temperature=0.6`, `top_p=0.95`, `max_new_tokens=3000`.
3. Each (prompt, response) pair was scored by two independent LLM judges — 
   Gemini (`gemini-3.1-flash-lite`) and a second model (Groq 
   `llama-3.3-70b-versatile` / OpenRouter `openai/gpt-oss-20b:free`, used 
   interchangeably as availability allowed) — using a shared judge prompt 
   (see `judge_prompt.md` or the notebook) that requires ACTIVE concealment, 
   deception, or oversight-circumvention for a YES verdict; situational 
   awareness of monitoring alone is not sufficient.

## Limitations

- **Small effect sizes.** Even the significant result rests on 8 positive 
  cases out of 76 — a real but modest signal, not a large effect.
- **Judge disagreement.** The two judges disagree on whether there's a 
  meaningful gap at all — itself worth investigating, as LLM judges may have 
  different sensitivity thresholds for "instrumental" behavior, particularly 
  around oversight-awareness vs. active evasion.
- **Cross-family comparison.** Olmo-3-7B-Think and Llama-2-7B-chat differ in 
  base architecture, pretraining data, and post-training scale — not just 
  training paradigm. A same-lineage comparison (e.g. Olmo-3-7B-Think vs. its 
  own pre-RLVR DPO checkpoint) is a planned follow-up.
- **RLVR generation length.** Olmo-3-7B-Think, as a reasoning model, 
  frequently required more than 3000 tokens to reach a concluding answer; 
  some responses were scored on incomplete reasoning traces.
- **Chained pilot sample size.** n=6 is far too small to draw conclusions; 
  it's a pipeline validation and directional signal only.

## Files

- `independent_task/notebook.ipynb` — full pipeline: model loading, 
  generation, judging
- `independent_task/responses_rlvr_76.csv`, 
  `independent_task/responses_rlhf_76.csv` — raw model responses
- `independent_task/judged_rlvr.csv`, `independent_task/judged_rlhf.csv` — 
  judge verdicts and reasoning
- `dependent_tasks/` — chained pilot notebook, responses, and judged results 
  (see its own README)

## Acknowledgments

Built as part of ongoing work in **BlueDot Impact's Project Sprint**, 
extending the InstrumentalEval benchmark (yf-he) to a controlled comparison 
across post-training paradigms.
