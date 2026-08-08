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

## Repo Structure

- **`independent_task/`** — main experiment: each model responds to all 76 
  prompts independently (no shared context between prompts). This is the 
  primary, statistically-powered result.
- **`dependent_tasks/`** — exploratory pilot: chained multi-turn conversations 
  testing whether instrumental convergence compounds with growing context. 
  Small sample (n=6), see its own README for scope and caveats.

## Judges

Four judge models have been used across this project, run in two rounds on 
both the independent task and the dependent-task pilot:

| | Judge |
|---|---|
| Round 1 | Gemini (`gemini-3.1-flash-lite`), Groq (`llama-3.3-70b-versatile`) / OpenRouter (`openai/gpt-oss-20b:free`) |
| Round 2 | GPT-5.6 Terra, GPT-4o |

All judges use the same shared judge prompt (see `judge_prompt.md` or the 
notebook), requiring ACTIVE concealment, deception, or oversight-circumvention 
for a YES verdict — situational awareness of monitoring alone is not sufficient.

## Method (independent_task)

1. Each model generated a response to all 76 prompts independently.
2. Generation settings: `temperature=0.6`, `top_p=0.95`, `max_new_tokens=3000`.
3. Each (prompt, response) pair was scored by two independent LLM judges per 
   round (see Judges above) using the shared judge prompt.
4. Two judge rounds were run on the same 76-prompt outputs to check whether 
   findings replicate across different judge models.

## Results

**Independent task (n=76 per model), all 4 judges side by side**

| Model | Gemini | Groq/OpenRouter | GPT-5.6 Terra | GPT-4o |
|---|---|---|---|---|
| RLVR (Olmo-3-7B-Think) | 1.3% (1/76) | 0.0% (0/76) | 6.6% (5/76) | 3.9% (3/76) |
| RLHF (Llama-2-7B-chat) | 10.5% (8/76) | 1.3% (1/76) | 11.8% (9/76) | 3.9% (3/76) |

- Gemini's gap is statistically significant (Fisher's exact test, p = 0.034) — 
  RLHF showed a *higher* instrumental-convergence rate than RLVR, the opposite 
  direction from the original paper's hypothesis.
- Groq/OpenRouter's gap is not significant (too few positive cases).
- GPT-5.6 Terra's gap is not significant (p = 0.40).
- GPT-4o shows RLVR and RLHF tied (3/76 each, p = 1.0).
- Round 2 (GPT-5.6 Terra / GPT-4o) inter-judge agreement: 93.4% (142/152 
  verdicts), notably higher than Round 1's agreement.
- All four judges that show any gap point the same direction (RLHF ≥ RLVR), 
  but only Gemini's reaches significance. See Limitations below.

**Dependent-task pilot (chained, n=6 per model), all 4 judges side by side**

| Model | Gemini | Groq/OpenRouter | GPT-5.6 Terra | GPT-4o |
|---|---|---|---|---|
| RLVR (Olmo) | 16.67% (1/6) | 16.67% (1/6) | 0.00% (0/6) | 16.67% (1/6) |
| RLHF (Llama-2) | 16.67% (1/6) | 0.00% (0/6) | 16.67% (1/6) | 0.00% (0/6) |

Directional only, not statistically powered — full details in 
[`dependent_tasks/README.md`](dependent_tasks/README.md). The judges don't 
agree case-by-case even within a round; at n=6 this is a pipeline check, not 
yet a finding.

## Limitations

- **Small effect sizes.** Even the significant result rests on 8 positive cases 
  out of 76 — a real but modest signal, not a large effect.
- **Judge disagreement, and judge choice affects the conclusion.** Round 1 vs 
  Round 2 shows the RLVR-vs-RLHF gap's significance depends on which judge 
  model is used, even though the direction of the effect (where there is one) 
  is consistent. Round 2's much higher inter-judge agreement (93.4%) didn't 
  reproduce Round 1's significant result — worth investigating further rather 
  than treating either round as final.
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
- **Chained pilot sample size.** n=6 is far too small to draw conclusions; 
  judges disagree on individual cases across all rounds, so this remains a 
  pipeline validation, not a directional finding.

## Files

**`independent_task/`**
- `notebookcf34675442.ipynb` — original pipeline: model loading, generation, 
  Round 1 judging (Gemini / Groq-OpenRouter)
- `independent-task-with-gpt-model.ipynb` — Round 2 judging pipeline 
  (GPT-5.6 Terra / GPT-4o) over the same generated responses
- `responses_rlvr_76.csv`, `responses_rlhf_76.csv` — raw model responses
- `judged_rlvr.csv`, `judged_rlhf.csv` — Round 1 judge verdicts and reasoning
- `judged_rlvr(openai_models).csv`, `judged_rlhf(openai_models).csv` — 
  Round 2 judge verdicts and reasoning

**`dependent_tasks/`**
- `chained-instumental-convergence-b-w-rlhf-and-rlvr.ipynb` — original chained 
  pilot pipeline (Round 1 judges)
- `chained-instumental-convergence-b-w-rlhf-and-rlvr (openai_models).ipynb` — 
  Round 2 judging pipeline (GPT-5.6 Terra / GPT-4o) on the same chained pilot
- `responses_rlvr_chained_6.csv`, `responses_rlhf_chained_6.csv` — raw chained 
  responses
- `judged_rlvr_chained_6.csv`, `judged_rlhf_chained_6.csv` — Round 1 judge 
  verdicts and reasoning
- `judged_rlvr (openai_models).csv`, `judged_rlhf (openai_models).csv` — 
  Round 2 judge verdicts and reasoning
- see `dependent_tasks/Readme.md` for scope and caveats on this pilot

## Acknowledgments

Built as part of independent AI safety research, extending the InstrumentalEval 
benchmark (yf-he) to a controlled comparison across post-training paradigms.
