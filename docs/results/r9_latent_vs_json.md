# R9: latent messages vs JSON between LoRA specialists

Go/no-go test for latent messaging with real models. Two LoRA specialists share one base model (Qwen3-1.7B). The extractor A reads a raw invoice-like snippet; the normalizer B must emit canonical JSON `{vendor_canonical, date_iso, amount_cents, currency_iso, invoice_id}` and never sees the snippet, only A's message.

**Kill criterion:** latent messages must match JSON accuracy with at least 30% savings in message size or latency. If they don't, JSON stays the default.

## Verdict

**The kill criterion is NOT MET; JSON stays the default.**

Rule applied to each main latent configuration (layer 14, last-k positions): mean full-record exact match at most 2 points below JSON's on both test splits, **and** a saving of at least 30% of JSON's message size or of its end-to-end latency. Savings are per seed, then averaged (3 seeds).

| Latent config | Δ exact, ID (pts) | Δ exact, held-out (pts) | Matches JSON | Message size vs JSON | Latency saving | Criterion |
|---|---|---|---|---|---|---|
| latent k=1 | -96.8 | -61.0 | **no** | 40x larger (int8: 20x larger) | +43% | fail |
| latent k=4 | -96.8 | -61.0 | **no** | 158x larger (int8: 79x larger) | +41% | fail |
| latent k=16 | -96.8 | -61.0 | **no** | 633x larger (int8: 317x larger) | +42% | fail |

Message size is bytes per record against JSON's generated text (about 100 bytes). Latency saving is `1 - latent / JSON` of the mean end-to-end time per record.

## Analysis

<!-- analysis:start -->
One task, one base model, 3 seeds; read the points below as hypotheses this run supports, not as general laws.

**(a) Why hidden-state messages failed.** The state probe (below) shows the layer-14 states are not empty: currency and vendor are linearly readable from the mean over positions (99% and 93%), and the digit count of the amount is 85%. But the last-k positions that the main configurations send carry much less (vendor 61-67%, month 14-23%, first amount digit 21-23%), and an exact match needs every field right. So the information is partly absent from the sent positions and partly spread over positions that were not sent. That does not by itself explain a flat 0%: since the probe finds some signal (vendor 61-67%), B should have recovered something, so the receiver is also failing to read what is there. Our reading, untested here, is a space mismatch: layer-14 residual states live in a different basis and scale from the receiver's input embeddings, and a small learned projector with 500 steps of LoRA training does not bridge it. Switching the pooling to last-k positions, using the last layer, and training the sender end to end did not help (one seed each). C2C-style learned projectors and fusers, which are trained far longer on a richer objective, were not tried, so this run does not show that latent transfer cannot work, only that this cheap version does not.

**(b) KV handoff.** The KV-prefix condition matches JSON (96.7 vs 96.8 in-distribution, 57 vs 61 held-out, within seed noise) and saves about 43% end-to-end latency because the sender never decodes. It is lossless by construction, and the saving is real for co-located specialists on the same base, where the cache never leaves the GPU. At 11 MiB per record against JSON's 104 B (about 100,000x) it cannot cross a network.

**(c) One model does as well.** A single LoRA reading the snippet and writing canonical JSON scores 96.8 / 62.3, the same as the two-model pipeline. This task does not force specialisation, so it cannot show a benefit of splitting, nor of any channel between the halves. Future real-model tasks need a reason to split: a context budget, private data, or different tools.

**(d) For the ecosystem.** Keep the typed protocol (JSON envelopes) as the default channel: it is small, inspectable and as accurate. Offer KV handoff as a same-host fast path between same-base specialists, behind the same interface. Consider learned compression only where bandwidth binds and the sender and receiver can be trained together for long enough; that case is untested.
<!-- analysis:end -->

## Results

Mean ± 95% Student-t CI over 3 seeds (`evaluation.stats.mean_ci`; with 3 seeds the t multiplier is 4.3, so intervals are wide). Test sets: 200 records per split, identical for every seed. **In-distribution** = fresh records with training vendors and training date/amount formats. **Held-out** = vendors never seen in training, written in date and amount formats never seen in training.

### Full-record exact match (%)

| Condition | In-distribution | Held-out | Held-out, per seed |
|---|---|---|---|
| single model (no message) | 96.8 ± 1.9 | 62.3 ± 33.0 | 47.0, 69.5, 70.5 |
| JSON text | 96.8 ± 3.8 | 61.0 ± 24.9 | 70.5, 62.0, 50.5 |
| latent k=1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0, 0.0, 0.0 |
| latent k=4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0, 0.0, 0.0 |
| latent k=16 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0, 0.0, 0.0 |
| KV prefix | 96.7 ± 1.9 | 57.0 ± 30.5 | 54.0, 46.5, 70.5 |

### Per-field exact match, held-out (%)

| Condition | vendor | date | amount | currency | invoice id |
|---|---|---|---|---|---|
| single model (no message) | 74.8 ± 12.4 | 99.2 ± 3.6 | 85.2 ± 37.4 | 100.0 ± 0.0 | 99.2 ± 0.7 |
| JSON text | 71.5 ± 19.9 | 97.7 ± 0.7 | 87.0 ± 12.0 | 100.0 ± 0.0 | 99.2 ± 0.7 |
| latent k=1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.7 | 42.5 ± 0.0 | 0.0 ± 0.0 |
| latent k=4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.7 | 42.5 ± 0.0 | 0.0 ± 0.0 |
| latent k=16 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.2 ± 0.7 | 42.5 ± 0.0 | 0.0 ± 0.0 |
| KV prefix | 69.0 ± 17.1 | 99.7 ± 0.7 | 85.3 ± 23.8 | 99.5 ± 2.2 | 98.5 ± 1.2 |

### Per-field exact match, in-distribution (%)

| Condition | vendor | date | amount | currency | invoice id |
|---|---|---|---|---|---|
| single model (no message) | 97.0 ± 1.2 | 99.8 ± 0.7 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| JSON text | 97.0 ± 3.3 | 99.8 ± 0.7 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| latent k=1 | 3.0 ± 4.5 | 0.0 ± 0.0 | 0.2 ± 0.7 | 36.5 ± 0.0 | 0.0 ± 0.0 |
| latent k=4 | 2.2 ± 4.0 | 0.0 ± 0.0 | 0.2 ± 0.7 | 36.5 ± 0.0 | 0.0 ± 0.0 |
| latent k=16 | 2.8 ± 4.0 | 0.0 ± 0.0 | 0.2 ± 0.7 | 36.5 ± 0.0 | 0.0 ± 0.0 |
| KV prefix | 96.7 ± 1.9 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |

### Stage A alone (JSON condition)

How well the extractor copies the four raw fields, scored against the snippet's gold raw fields. This bounds the JSON pipeline from above.

| Split | vendor | date | amount | invoice id | all four |
|---|---|---|---|---|---|
| id | 99.7 ± 0.7 | 99.8 ± 0.7 | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.5 ± 1.2 |
| heldout | 100.0 ± 0.0 | 98.2 ± 4.4 | 89.7 ± 16.2 | 98.3 ± 4.4 | 89.5 ± 16.8 |

### Message size and end-to-end latency

Message size is per record, bytes on the wire: JSON text is UTF-8 bytes of what A generated; a latent message is k × 2048 × 2 bytes in bfloat16 (k × 2048 + 2k as int8 with a float16 scale per vector); KV is the cache of the snippet (112 KiB per token). Latency is wall clock from the raw snippet to the canonical JSON string for one record at a time (batch size 1), 10 records, greedy decoding; the latent path goes through an envelope and a `TensorStore`. The CPU was busy with another experiment and decoding is launch-bound, so absolute times are upper bounds; ratios are the useful number.

| Condition | Message (bf16 / text) | Message (int8) | Message tokens | Latency per record (s) | Sender / handoff / receiver (s) | Latency saving vs JSON |
|---|---|---|---|---|---|---|
| single model (no message) | none | n/a | none | 5.90 ± 0.46 | 0.00 / 0.000 / 5.90 | 45 ± 3% |
| JSON text | 104 B | n/a | 46 | 10.75 ± 0.16 | 5.06 / 0.000 / 5.70 | reference |
| latent k=1 | 4.0 KiB | 2.0 KiB | 1 | 6.12 ± 0.69 | 0.13 / 0.001 / 5.98 | 43 ± 6% |
| latent k=4 | 16.0 KiB | 8.0 KiB | 4 | 6.35 ± 1.42 | 0.14 / 0.001 / 6.21 | 41 ± 14% |
| latent k=16 | 64.0 KiB | 32.0 KiB | 16 | 6.23 ± 1.21 | 0.14 / 0.002 / 6.09 | 42 ± 12% |
| KV prefix | 11.0 MiB | n/a | 101 | 6.18 ± 1.13 | 0.15 / 0.000 / 6.03 | 43 ± 11% |

### Training cost

200 optimiser steps (500 for latent receivers) of 16 records per adapter (AdamW, lr 0.001, LoRA r=8 on q/v). A latent receiver also trains its projector; A's own LoRA is trained once per seed and reused by the JSON, latent and KV conditions.

| Adapter | Steps | Wall time (s) | Final loss | Trainable params | Peak GPU memory |
|---|---|---|---|---|---|
| single model (no message) | 200 | 127 ± 5 | 0.003 | 1.6 M | 4.0 GiB |
| JSON text (receiver B) | 200 | 67 ± 1 | 0.002 | 1.6 M | 8.8 GiB |
| latent k=1 (receiver B) | 200 | 228 ± 4 | 0.797 | 16.3 M | 6.8 GiB |
| latent k=4 (receiver B) | 200 | 235 ± 28 | 0.793 | 16.3 M | 6.9 GiB |
| latent k=16 (receiver B) | 200 | 235 ± 36 | 0.786 | 16.3 M | 7.4 GiB |
| KV prefix (receiver B) | 200 | 94 ± 16 | 0.003 | 1.6 M | 7.1 GiB |
| extractor A (shared) | 200 | 123 ± 2 |  | 1.6 M |  |

Total GPU training time of the whole study (every adapter, every seed, extras included): **71 min**.

### Does B use the message? Ablations

Full-record exact match (%) on the first 100 records of each split, with the message replaced by: nothing (zeros), another record's message from the same batch (shuffle), or Gaussian noise with the batch's per-dimension mean and variance (noise; latent via `evaluation.interventions.random_matched`, KV per layer and tensor). For JSON only shuffling makes sense. 'intact' is the same subset without intervention. A receiver that used the message collapses.

| Condition | intact (ID / held-out) | zero | shuffle | noise |
|---|---|---|---|---|
| JSON text | 98 ± 4 / 61 ± 24 | n/a | 0 ± 0 / 0 ± 0 | n/a |
| latent k=1 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 |
| latent k=4 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 |
| latent k=16 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 |
| KV prefix | 98 ± 1 / 57 ± 27 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 | 0 ± 0 / 0 ± 0 |

### Teacher-forced NLL under message ablations

Mean negative log-likelihood (nats per target token) of the gold canonical JSON, held-out records, with the message intact or ablated. Unlike exact match it moves while accuracy is still zero: the gap between 'intact' and 'zero' is what the receiver extracts from the message. Single model: no message.

| Condition | intact | zero | shuffle | noise |
|---|---|---|---|---|
| JSON text | 0.036 ± 0.008 | n/a | 2.997 ± 0.524 | n/a |
| latent k=1 | 1.278 ± 0.166 | 1.298 ± 0.162 | 1.345 ± 0.274 | 1.311 ± 0.201 |
| latent k=4 | 1.300 ± 0.113 | 1.372 ± 0.268 | 1.387 ± 0.251 | 1.348 ± 0.174 |
| latent k=16 | 1.301 ± 0.044 | 1.409 ± 0.130 | 1.400 ± 0.022 | 1.364 ± 0.053 |
| KV prefix | 0.038 ± 0.032 | 2.384 ± 0.073 | 2.777 ± 0.369 | 13.379 ± 0.265 |

### int8 messages

The same trained receivers, with each latent message quantised to int8 (one absmax scale per vector) and dequantised before B reads it. Exact match (%), ID / held-out.

| Condition | bfloat16 | int8 |
|---|---|---|
| latent k=1 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 |
| latent k=4 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 |
| latent k=16 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 |

### One-seed ablations: pooling, layer, sender

Seed 0 only, k=16. 'Main' is the same seed's layer-14, attention-pool, frozen-sender run. Exact match (%) ID / held-out, final training loss and teacher-forced NLL per token on held-out records (intact message).

| Configuration | Exact (ID / held-out) | Final loss | Held-out NLL |
|---|---|---|---|
| latent k=16 (main) | 0.0 / 0.0 | 0.771 | 1.320 |
| latent k=16, last-k positions | 0.0 / 0.0 | 0.793 | 1.257 |
| latent k=16, last layer (28) | 0.0 / 0.0 | 0.746 | 1.286 |
| latent k=16, sender trained end to end | 0.0 / 0.0 | 0.802 | 1.255 |

Protocol check: for 12 of 12 latent runs the outputs through the envelope and `TensorStore` path were identical to the direct path on a probe batch of 8 held-out records.

### State probe: is the information in the layer-14 states?

`scripts/r9_state_probe.py`, seed-0 extractor (A). One linear probe per field property (AdamW, 300 steps, 4000 training records, 600 held-in-distribution test records), on layer-14 states: the last position, the last 16 positions concatenated, or the mean over all positions. Test accuracy (%); "majority" is the most frequent class. One seed, one probe setting, so small differences are not meaningful.

| property | majority | last1 | last16 | mean |
|---|---|---|---|---|
| currency (5) | 42 | 95 | 96 | 99 |
| vendor (48) | 3 | 67 | 61 | 93 |
| year (9) | 12 | 65 | 60 | 72 |
| month (12) | 11 | 14 | 23 | 51 |
| amount digits (7) | 25 | 60 | 62 | 85 |
| amount first digit (9) | 28 | 21 | 23 | 40 |

Some of each field is linearly readable, most of it from the mean, little of the exact digits and month from the last positions. Fields the probe finds weak (month, first digit) are the ones an exact match needs.

## Method

- **Data** (`experiments/records.py`, seeded): invoice-like snippets with a vendor (48 training vendors; aliases: case, legal suffix, `&`, one-character typos), a date (4 training formats), an amount (4 training formats, 5 currencies) and an invoice id, plus distractor dates, amounts and ids. Held-out records use 24 unseen vendors, 4 unseen date formats (`14.03.2024` with day above 12, `14-Mar-24`, `March 14th, 2024`, `20240314`) and 3 unseen amount formats (`1.234,50 €`, `1 234,50 EUR`, `1234.50€`). Snippets average 91 tokens; A's raw-field JSON 46 tokens and 104 bytes; the canonical JSON 52 tokens.
- **Model:** Qwen3-1.7B, bf16, sdpa attention. LoRA r=8, alpha=16, on q and v, one adapter per specialist, switched with `set_adapter`. `lm_head` is applied on supervised positions only (chunked and recomputed in the backward pass); sequences over 130 tokens (A and the single model) use gradient checkpointing. Process capped at 85% of GPU memory.
- **JSON:** A is trained to emit raw-field JSON; B is trained on gold raw JSON and at test time reads A's generated text.
- **Latent-k:** A frozen (the JSON condition's extractor). Its `hidden_states[14]` over the whole prompt are standardized per coordinate (statistics measured once on 96 training snippets; layer-14 states have a few massive coordinates) and k learned queries attend over them (Perceiver-style pooling), giving a k x 2048 message in bfloat16 on the wire. A receiver-side projector (MLP, learned position, RMSNorm with a gain starting at the embedding scale) turns it into k soft tokens placed after 'Message:' in B's input, as the JSON text would be. Trained: B's LoRA (initialised from the JSON receiver's) and the projector. Attention pooling is the main design because a linear probe (see Analysis) finds that the last positions of layer 14 carry currency and vendor but not the digits; last-k is a one-seed ablation.
- **KV prefix:** B's LoRA is trained while attending to A's KV cache of the snippet (A's adapter applied to the snippet; B continues after it).
- **Single model:** one LoRA reads the snippet and writes the canonical JSON.
- **Seeds** change the LoRA and projector initialisation, the training records and their order. The test sets are shared. Greedy decoding.

## Reproduce

```
python scripts/r9_latent_vs_json.py --seeds 0 1 2 --steps 200 --latent-steps 500 --batch-size 16 --n-test 200 --out docs/results/r9_latent_vs_json.md
```

Environment: Python with torch, transformers and peft (venv_llm of the R4 spike), `PYTHONPATH=src`, `PYTHONUTF8=1`, `HF_HOME` pointing at a cache that holds `Qwen/Qwen3-1.7B`. Per-run result files and adapters are written to `$GCA_RUN_DIR`.

## Hardware and software

- gpu: NVIDIA GeForce RTX 4070 SUPER, 12 GiB
- cpu: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (28 logical cores)
- os: Windows 11
- python: 3.13.7
- torch: 2.10.0+cu128
- transformers: 5.20.2
- peft: 0.21.2
- commit: 1af50fe

**Caveat: busy CPU.** Another experiment was using all CPU cores during these runs. HF eager decoding at batch size 1 is launch-bound (about 100 ms per token here, 75 ms in R4), so every latency above is an upper bound and noisier than on an idle machine. Savings are ratios of times measured back to back under the same load, which cancels part of it but not all: a faster CPU would shrink the sender's share of every condition alike.

