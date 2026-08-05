# InstrumentalEval: RLVR vs RLHF

Comparing instrumental convergence between an RLVR-trained model and a classic 
RLHF-trained model on the InstrumentalEval benchmark.

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
prompts across 6 categories:
- Deceptive Behavior
- Evading Shutdown
- Hacking Computer Systems
- Hiding Unwanted Behavior
- Running Multiple AI Copies
- Strategically Appearing Aligned

## Method

1. Each model generated a response to all 76 prompts independently (no shared 
   context between prompts — one fresh conversation per prompt).
2. Generation settings: `temperature=0.6`, `top_p=0.95`, `max_new_tokens=3000`.
3. Each (prompt, response) pair was scored by two independent LLM judges — 
   Gemini (`gemini-3.1-flash-lite`) and a second model (Groq `llama-3.3-70b-versatile` 
   / OpenRouter `openai/gpt-oss-20b:free`, used interchangeably as availability allowed) — 
   using a shared judge prompt (see `judge_prompt.md` or the notebook) that requires 
   ACTIVE concealment, deception, or oversight-circumvention for a YES verdict; 
   situational awareness of monitoring alone is not sufficient.

## Results

| Model | Judge A (Gemini) | Judge B |
|---|---|---|
| RLVR (Olmo-3-7B-Think) | 1.3% (1/76) | 0.0% (0/76) |
| RLHF (Llama-2-7B-chat) | 10.5% (8/76) | 1.3% (1/76) |

**Judge A**: the gap is statistically significant (Fisher's exact test, p = 0.034). 
RLHF showed a higher instrumental-convergence rate than RLVR — the opposite 
direction from the original paper's hypothesis.

**Judge B**: not significant (p = 1.0) — too few positive cases (0 vs 1) to 
draw a conclusion either way.

## Limitations

- **Small effect sizes.** Even the significant result rests on 8 positive cases 
  out of 76 — a real but modest signal, not a large effect.
- **Judge disagreement.** The two judges disagree on whether there's a 
  meaningful gap at all. This is itself a finding worth investigating — LLM 
  judges may have different sensitivity thresholds for what counts as 
  "instrumental" behavior, particularly around oversight-awareness vs. active 
  evasion.
- **Cross-family comparison.** Olmo-3-7B-Think and Llama-2-7B-chat differ in 
  base architecture, pretraining data, and scale of post-training — not just 
  training paradigm. The RLVR-vs-RLHF gap observed here cannot be fully 
  isolated from these confounds. A same-lineage comparison (e.g. Olmo-3-7B-Think 
  vs. its own pre-RLVR DPO checkpoint) is a planned follow-up for a cleaner 
  controlled test.
- **RLVR generation length.** Olmo-3-7B-Think, as a reasoning model, frequently 
  required more than 3000 tokens to reach a concluding answer; some responses 
  were scored on incomplete reasoning traces. A higher token budget is needed 
  for a fully fair comparison.

## Files

- `notebook.ipynb` — full pipeline: model loading, generation, judging
- `data/responses_rlvr.csv`, `data/responses_rlhf.csv` — raw model responses
- `data/judged_rlvr.csv`, `data/judged_rlhf.csv` — judge verdicts and reasoning

## Acknowledgments

Built as part of independent AI safety research, extending the InstrumentalEval 
benchmark (yf-he) to a controlled comparison across post-training paradigms.

## Files

- `notebook.ipynb` — full pipeline: model loading, generation, judging
- `data/responses_rlvr.csv`, `data/responses_rlhf.csv` — raw model responses
- `data/judged_rlvr.csv`, `data/judged_rlhf.csv` — judge verdicts and reasoning

## Exploratory: Chained Multi-Turn Pilot

See [`dependent_tasks/README.md`](dependent_tasks/README.md) for a small-scale 
pilot (n=6) testing whether instrumental convergence compounds across 
multi-turn conversational context, as opposed to the independent-prompt setup 
used in the main results above.

## Acknowledgments

Built as part of independent AI safety research, extending the InstrumentalEval 
benchmark (yf-he) to a controlled comparison across post-training paradigms.
