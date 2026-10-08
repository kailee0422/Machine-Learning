# Experiment Report

## Introduction

This study explores fine-tuning **LLaMA-3 (8B, 4-bit quantized)** using the lightweight
`unsloth` framework, applying **LoRA (Low-Rank Adaptation)** to improve the model's
Chinese instruction-following ability without training a large model from scratch.

## Materials and Methods

- **Model**: `unsloth/llama-3-8b-bnb-4bit`.
- **Datasets**: `kigner/ruozhiba-llama3-tt` (Chinese dialogue/domain-specific
  conversation) and `hibing624/alpaca-zh` (Chinese Alpaca-style self-instruct data from
  GPT-4), trained separately to compare their effect on responses.
- **LoRA config**: applied to `q_proj`, `k_proj`, etc., rank 16, dropout 0.
- **Training**: gradient checkpointing enabled; batch size 2, learning rate 2e-4, max 60
  steps, AdamW optimizer with 8-bit precision.

## Results

- **Before fine-tuning**: the base model mostly echoed the input or failed to answer
  (e.g. asked what it would do if someone stole bread, it replied "I'll tell him I don't
  have bread" or simply repeated the question).
- **After fine-tuning** on either dataset, the model gave an actually relevant answer
  (e.g. "I'd report it to the police since that's a criminal act").
- **Unfamiliar-concept test** (a nonsense term, 吉伊卡哇): the base model confidently
  fabricated a plausible-sounding but entirely made-up definition rather than admitting
  it didn't know — a clear hallucination under an out-of-distribution prompt.

## Conclusion

Fine-tuning LLaMA-3 with LoRA on a small custom dataset measurably improved its
instruction-following behavior without needing to train a large model from scratch,
confirming LoRA as an efficient way to adapt an LLM to a specific domain or style. The
model still hallucinates confidently on genuinely unfamiliar terms — a limitation that
targeted domain data could address in future work.

## Code

You can run on colab or local

<a target="_blank" href="https://colab.research.google.com/github/kailee0422/Machine-Learning/blob/main/HW4/ML_HW4_alpaca.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
