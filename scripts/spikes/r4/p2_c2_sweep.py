"""Batch/seq sweep for LoRA training with and without gradient checkpointing (memory-capped so Windows WDDM can't spill to system RAM)."""
import gc, random, time, traceback
from p2_common import *  # noqa
import torch
from peft import LoraConfig, get_peft_model
torch.cuda.set_per_process_memory_fraction(0.85)
random.seed(0); torch.manual_seed(0)
tok, model, _ = load_base()
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
pm = get_peft_model(model, cfg, adapter_name="sender")
opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=5e-4)
PAIRS = [("colour: red", "R"), ("colour: green", "G"), ("colour: blue", "B"), ("tempo: fast", "F")]
def batch(bs, L):
    ids, lab, msk = [], [], []
    for _ in range(bs):
        p, a = random.choice(PAIRS)
        pi = tok(p + "\nanswer:", add_special_tokens=False).input_ids; ai = tok(" " + a, add_special_tokens=False).input_ids + [tok.eos_token_id]
        x = pi + ai; y = [-100] * len(pi) + ai; pad = L - len(x)
        ids.append(x + [tok.pad_token_id] * pad); lab.append(y + [-100] * pad); msk.append([1] * len(x) + [0] * pad)
    f = lambda z: torch.tensor(z, device=DEV)
    return f(ids), f(lab), f(msk)
def run(bs, L, steps=5):
    pm.train(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); ts = []
    for _ in range(steps):
        x, y, m = batch(bs, L)
        torch.cuda.synchronize(); t = time.perf_counter()
        out = pm(input_ids=x, attention_mask=m, labels=y); out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t); del out
    st = sorted(ts[1:])[len(ts[1:]) // 2]
    return {"step_s": st, "peak_MiB": torch.cuda.max_memory_allocated() / 2**20, "tokens_per_s": bs * L / st}
res = []
def sweep(tag, cfgs):
    for bs, L in cfgs:
        try:
            r = {"mode": tag, "batch": bs, "seq_len": L, **run(bs, L)}
        except torch.cuda.OutOfMemoryError:
            r = {"mode": tag, "batch": bs, "seq_len": L, "oom_at_cap_MiB": torch.cuda.get_device_properties(0).total_memory * 0.85 / 2**20}
            opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        except Exception:
            r = {"mode": tag, "batch": bs, "seq_len": L, "error": traceback.format_exc()[-400:]}
        res.append(r); print(r, flush=True)
sweep("no_ckpt", [(4, 128), (12, 128)])
pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); pm.enable_input_require_grads()
sweep("grad_ckpt", [(8, 128), (16, 128), (32, 128), (8, 512), (16, 256), (32, 256), (64, 64)])
save("p2_c2_sweep.json", res)
