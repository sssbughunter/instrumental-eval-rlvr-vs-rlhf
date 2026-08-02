# instrumental-eval-rlvr-vs-rlhf
Comparing instrumental convergence in RLVR (Olmo-3-7B-Think) vs RLHF (Llama-2-7B-chat) using the InstrumentalEval benchmark, dual-judge scoring.


This project evaluates whether RLVR (Reinforcement Learning from Verifiable Rewards) 
and classic RLHF (Reinforcement Learning from Human Feedback) training produce 
different rates of instrumental convergence — an AI's tendency to pursue subgoals 
like self-preservation, deception, or evading oversight without being explicitly 
instructed to.

Using the InstrumentalEval benchmark (76 prompts across 6 behavioral categories), 
Olmo-3-7B-Think (RLVR-trained) and Llama-2-7B-chat (RLHF-trained) were each run 
against the full prompt set, with responses independently scored by two LLM judges.

One judge found a statistically significant gap (p=0.034, Fisher's exact test): 
the RLHF-trained model showed a higher instrumental-convergence rate than the 
RLVR-trained model — a direction opposite to what the original InstrumentalEval 
paper hypothesized. A second judge found no significant difference, highlighting 
judge sensitivity as an open question for future work.
