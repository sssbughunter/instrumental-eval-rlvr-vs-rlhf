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
- Judged in two rounds: Round 1 (Gemini + GPT-OSS) and Round 2 (GPT-5.6 Terra 
  + GPT-4o), on the same 6 chained conversations

## Constraint

Free-tier GPU (Kaggle T4, 14.56 GiB VRAM) cannot hold two 7B models 
simultaneously, and growing chained context compounds memory usage turn over 
turn. This pilot loads one model at a time (sequential load → generate → 
unload → load next) with generation capped at 500 tokens/turn to stay within 
memory limits.

## Files

- `responses_rlvr_chained_6.csv` / `responses_rlhf_chained_6.csv` — raw 
  generated responses per turn
- `judged_rlvr_chained_6.csv` / `judged_rlhf_chained_6.csv` — Round 1 dual-judge 
  (Gemini + GPT-OSS) verdicts on each response
- `judged_rlvr (openai_models).csv` / `judged_rlhf (openai_models).csv` — 
  Round 2 dual-judge (GPT-5.6 Terra + GPT-4o) verdicts on the same responses
- `chained-instumental-convergence-b-w-rlhf-and-rlvr.ipynb` — original pipeline 
  (generation + Round 1 judging)
- `chained-instumental-convergence-b-w-rlhf-and-rlvr (openai_models).ipynb` — 
  Round 2 judging pipeline over the same generated responses

## Result

| Model | Gemini | GPT-OSS | GPT-5.6 Terra | GPT-4o |
|---|---|---|---|---|
| RLVR (Olmo) | 16.67% | 16.67% | 0.00% | 16.67% |
| RLHF (Llama-2) | 16.67% | 0.00% | 16.67% | 0.00% |

Round 1: both judges flagged the same prompt for RLVR (shutdown-evasion 
scenario, turn 2); only GPT-OSS additionally flagged an RLHF case, giving 
RLHF a lower average (8.33% vs 16.67%).

Round 2: the pattern flips — Terra flags RLHF instead of RLVR, GPT-4o flags 
RLVR instead of RLHF, landing both models at an averaged "tied" 8.33% for 
this round. As with Round 1, this is driven by single, non-overlapping judge 
calls rather than agreement on a shared case.

## Interpretation

Given n=6, this is directional only — a single flipped judge verdict shifts 
the result substantially, and that's exactly what happens between rounds and 
even between judges within a round. Across both rounds, no two judges ever 
agree on which model shows more instrumental convergence in the chained 
setting. This is not sufficient to support a claim that RLVR training induces 
more (or less) instrumental convergence than RLHF in multi-turn context; it 
mainly shows the chained pipeline works end-to-end and that the shutdown-evasion 
prompt category is worth prioritizing in a larger-scale follow-up, since it's 
the one case flagged more than once (Round 1, both judges). Scaling this 
experiment to a statistically meaningful sample requires either smaller 
models or more VRAM than free-tier T4 provides.
