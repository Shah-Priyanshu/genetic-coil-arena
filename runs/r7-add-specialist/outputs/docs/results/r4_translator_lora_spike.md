# R4 spike report: translator and shared-base LoRA

Folder: `C:\Users\priya\AppData\Local\Temp\claude\D--Projects-genetic-coil-arena\bbe42ee6-ac23-453f-a943-a4838eb9c1dc\scratchpad\spikes`. The repo and `D:\venvs\gca` were not touched.

## Part 1: translator via Ollama structured outputs
120 items (40 templated, 80 paraphrases with typos), gold hand-written. 36 are held-out: their (tempo, hold, has-avoid) combination and region appear in none of the 4 few-shots. Temperature 0, `think:false`, num_ctx 2048, num_thread 4; the prompt states my label conventions. The server was down; I started `ollama serve` and stopped it at the end.

| model | exact | held-out | valid | eaters/region/avoid/tempo/hold | p50 / p95 | first call (load) | size |
|---|---|---|---|---|---|---|---|
| gemma4:e2b | 93.3% | 91.7% | 100% | 98/100/98/100/97 | 0.29 / 0.36 s | 15.0 s (14.2) | 1.7 GB |
| gemma4:e4b | 95.8% | 91.7% | 100% | 100/100/98/98/100 | 0.41 / 0.45 s | 110.9 s (51.3 + 59 prompt) | 3.2 GB |
| qwen2.5-coder:14b | **100%** | 100% | 100% | all 100 | 0.61 / 0.67 s | 11.0 s (10.1) | 9.1 GB |
| qwen3-coder:30b | 99.2% | 97.2% | 100% | 99/100/99/100/100 | 2.52 / 4.09 s | 72.3 s (69.1) | 19 GB (10.5 on GPU) |

Exact-match 95% CIs: 87-97, 91-98, 97-100, 95-100. The 30b ran (p50 well under 10 s) despite CPU offload. All models were unloaded. A cold e2b load took 125 s in a smoke run (during a download), so first-call numbers depend on the file cache.

Caveats:
- **Ceiling effect.** I wrote labels and conventions, so the set cannot separate the 14b from the 30b.
- **Misses.** e2b missed mostly `hold` and `avoid`; e4b had 3 tempo misses on soft cues ("safe", "don't rush", "no rush"; debatable labels) and 2 missed `avoid`.
- **Validity** was checked afterwards with `jsonschema`; unknown whether Ollama itself enforces minItems/uniqueItems.

**Recommendation: qwen2.5-coder:14b** (120/120, 0.6 s). Load it on demand (about 10 s) and unload it with keep_alive 0, since at 9.1 GB it cannot sit beside training (8 GB) on 12 GB. If the translator must stay resident, use gemma4:e4b (3.2 GB, 0.41 s, 96%). Skip the 30b: no gain, 4x slower.

## Part 2: Qwen3-1.7B + LoRA
Qwen/Qwen3-1.7B was available. `venv_llm` reuses torch 2.10.0+cu128 and adds transformers 5.20.2, peft 0.21.2, accelerate 1.15.0. Download: **4.08 GB in 69 s** into `...\spikes\hf\hub\models--Qwen--Qwen3-1.7B` (I redirected HF_HOME there).

**Timing caveat:** the CPU was about 78% busy with the other experiment, so HF eager is launch-bound (batch 8 gives 8x the tok/s of batch 1; forward time is flat from 128 to 512 tokens). Treat latencies as upper bounds.

| item | measured |
|---|---|
| VRAM after bf16 load | 3282 MiB allocated (1.72B params); +4385 MiB at driver level; 9.9 s load |
| generation, 128 tokens | 13.3 tok/s (batch 1); 105 tok/s aggregate (batch 8) |
| hidden states | 29 tensors (embeddings + 28 layers), each [1, L, 2048] bf16; last is post-final-norm; forward 91 ms (L=128), 95 ms (L=512). Norms: embedding rows 1.5, layer-14 last token 137-156, final layer 98-117 |
| KV cache | DynamicCache, 28 layers, [1, 8, L, 128] bf16; **112 KiB/token**; 56 MiB at 512 tokens |
| 2 LoRA adapters (r=8, q/v) | 1.6M params each, fp32: **+6.1 MiB each**. Base `data_ptr` identical across adapters, so the base is shared (a copy costs 3.3 GiB) |
| `set_adapter` | 4.2 ms/call; forward at L=128 107 ms same adapter vs 111 ms alternating; mixed-adapter batches (`adapter_names`) work but are slow (178 vs 119 ms) |
| latent soft prompt | sender state [1, 2048] (layer 14 or 28) becomes [1,1,2048], prepended to receiver embeds ([1, 9, 2048]); forward and generate run. Sender pass 101-122 ms; the extra token is below noise; full handoff 235 ms. Raw states are 76-100x larger than embeddings and shift next-token probabilities strongly (KL 5.5 at layer 14) |
| KV prefix to receiver | **works with PEFT** for forward and `generate`. fp32: max logit diff 5e-5, identical text. bf16: diff 0.22, greedy flips at token 3 (numerical noise). The cache is mutated in place: deepcopy (62 ms at 512 tokens) or `crop(-n)` |
| LoRA training | 50 steps, batch 8 x 128, AdamW lr 5e-4: loss 14.45, 3.05 (step 10), 0.0005 (step 50). **0.279 s/step**, peak 8.0 GiB. Accuracy on 12 pairs 0/12 to 12/12; untrained receiver stayed 0/12 |

**Training memory is dominated by full-vocab logits.** With the default loss, 8 x 128 is the maximum without checkpointing (12 x 128 hit the 10.2 GiB cap) and with checkpointing 16 x 128 fits (7.5 GiB). Computing `lm_head` only on supervised positions allows 16 x 128 without checkpointing (8.8 GiB, 0.32 s/step) and 64 x 256 with it (7.3 GiB, 4.1 s/step).

## What worked and what didn't
- **Windows memory spill.** WDDM spills to system RAM instead of raising OOM: my first training run sat over 10 minutes at 100% GPU with a 16 GB working set until I killed it (which sweep config caused it is unknown). Fix: `torch.cuda.set_per_process_memory_fraction(0.85)`; later sweeps ran capped.
- **Environment.** pip warns that litellm pins tokenizers/click; installs went into the venv only.
- **API changes.** transformers 5 uses `dtype=`, and `DynamicCache.crop` takes negative ints.
- **Disk.** HF cache 4.08 GB, venv 167 MB, results under 1 MB.

## Recommended defaults
- **Base:** Qwen3-1.7B, bf16, sdpa. One resident base, LoRA r=8 / alpha 16 on q and v, switched with `set_adapter`.
- **Latent messages:** start with `hidden_states[14]` as 1-4 soft tokens behind a learned projector (RMSNorm or learned scale); never inject raw states. Mid-versus-last is a prior, not measured (nothing was trained): ablate it next.
- **Training batches:** 16 x 128 with selective logits and no checkpointing; 32-64 x 128-256 with checkpointing. Cap VRAM at 85%.
