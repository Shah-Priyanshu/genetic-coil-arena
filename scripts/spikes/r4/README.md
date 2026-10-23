# R4 spike scripts

Scripts from the R4 feasibility spike (translator via Ollama structured outputs; Qwen3-1.7B + LoRA). Report: docs/results/r4_translator_lora_spike.md.

They were run from a scratch folder with a separate venv (transformers 5.20.2, peft 0.21.2, accelerate 1.15.0 on top of torch 2.10+cu128) and HF_HOME pointing at a local cache, so paths inside may need adjusting. Per-item translator outputs (about 130 KB per model) were not copied; p1_testset.jsonl holds the 120 instructions with gold specs.
