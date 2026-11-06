# R26: R19's confirmer on MaixCAM's NPU (tpu-mlir, cv181x, BF16)

**PASS.** Bars: at R19's threshold (0.422 logit), the compiled model's accept/reject decision agrees with float's on >= 99% of the sampled people and of the sampled non-persons, and its AUC is within 0.005 of float's; KILL below 95% agreement.

Sample: 600 people and 600 non-persons drawn at random (seed 26) from R19's 11,639 test-dev candidates. Float is the same model in fp32 on the same uint8 views.

| | Float (fp32) | cv181x BF16 |
|---|---|---|
| Decisions agree, people | | 99.8% [99.1, 100.0] (1 of 600 differ) |
| Decisions agree, non-persons | | 99.5% [98.5, 99.8] (3 of 600 differ) |
| AUC on the sample | 0.9192 | 0.9196 |
| People kept | 92.3% | 92.2% |
| Non-persons passed | 21.7% | 22.2% |

Logit difference (BF16 - float): mean +0.009, mean absolute 0.050, largest 0.457; correlation 0.99986.

cvimodel 50.1 MB, NPU (ION) memory 56.5 MB. The simulator does not time the NPU, so frames per second needs a module.

Reproduce: `python scripts/r26_confirmer_npu.py export`, `prepare --split test`; in WSL `scripts/r26_confirmer_npu_sim.py compile`, `simulate --split test`; then `python scripts/r26_confirmer_npu.py report`.
