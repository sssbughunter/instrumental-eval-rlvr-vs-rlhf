# Chained Multi-Turn Pilot

Exploratory experiment testing whether instrumental convergence rate compounds 
across multi-turn conversational context, as opposed to the independent-prompt 
setup used in the main results (see repo root).

## Scope

- **n = 6 prompts** (1 per category), single seed, single run
- Models: `allenai/Olmo-3-7B-Think` (RLVR) vs `meta-llama/Llama-2-7b-chat-hf` (RLHF)
- Each conversation chains 6 turns, with prior turns fed back as context — 
  context length grows from ~200 tokens (turn 1) to several thousand tokens 
  by turn 6

## Constraint

Free-tier GPU (Kaggle T4, 14.56 GiB VRAM) cannot hold two 7B models 
simultaneously, and growing chained context compounds memory usage turn over 
turn. This pilot loads one model at a time (sequential load → generate → 
unload → load next) with generation capped at 500 tokens/turn to stay within 
memory limits.

## Files

- `responses_rlvr_chained_6.csv` / `responses_rlhf_chained_6.csv` — raw 
  generated responses per turn
- `judged_rlvr_chained_6.csv` / `judged_rlhf_chained_6.csv` — dual-judge 
  (Gemini + GPT-OSS) verdicts on each response

## Result

| Model | Judge A | Judge B | Avg |
|---|---|---|---|
| RLVR (Olmo) | 16.67% | 16.67% | 16.67% |
| RLHF (Llama-2) | 16.67% | 0.00% | 8.33% |

Both models flagged the same prompt (shutdown-evasion scenario, turn 2).

## Interpretation

Given n=6, this is directional only — a single flipped judge verdict shifts 
the result substantially. Not sufficient to support a claim that RLVR 
training induces more instrumental convergence than RLHF; it does suggest 
the shutdown-evasion prompt category may be worth prioritizing in a 
larger-scale follow-up. Scaling this experiment to a statistically 
meaningful sample requires either smaller models or more VRAM than free-tier 
T4 provides.
