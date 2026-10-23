"""Part 2 step 6: LoRA training smoke test (50 steps, batch 8, seq 128, AdamW) + small batch/seq sweep."""
import random, time, traceback
from p2_common import *  # noqa
import torch, peft
from peft import LoraConfig, get_peft_model

random.seed(0); torch.manual_seed(0)
import gc
# Windows/WDDM silently spills to system RAM when VRAM is full (and becomes extremely slow); cap so OOM is raised instead.
torch.cuda.set_per_process_memory_fraction(0.85)
R = {"peft": peft.__version__}
tok, model, load_s = load_base()

PAIRS = [("colour: red", "R"), ("colour: green", "G"), ("colour: blue", "B"),
         ("region: top_left", "TL"), ("region: top_right", "TR"), ("region: bottom_left", "BL"),
         ("region: bottom_right", "BR"), ("tempo: fast", "F"), ("tempo: cautious", "C"), ("tempo: normal", "N"),
         ("hold: true", "H1"), ("hold: false", "H0")]
EOS = tok.eos_token

def encode_batch(pairs, seq_len):
    ids, labels, masks = [], [], []
    for p, a in pairs:
        pi = tok(p + "\nanswer:", add_special_tokens=False).input_ids
        ai = tok(" " + a, add_special_tokens=False).input_ids + [tok.eos_token_id]
        x = (pi + ai)[:seq_len]
        y = ([-100] * len(pi) + ai)[:seq_len]
        pad = seq_len - len(x)
        ids.append(x + [tok.pad_token_id] * pad); labels.append(y + [-100] * pad); masks.append([1] * len(x) + [0] * pad)
    t = lambda z: torch.tensor(z, device=DEV)
    return t(ids), t(labels), t(masks)

cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
pm = get_peft_model(model, cfg, adapter_name="sender")
pm.add_adapter("receiver", cfg)       # present but untrained, to check isolation
pm.set_adapter("sender")
pm.print_trainable_parameters()
R["trainable_params_sender"] = sum(p.numel() for n, p in pm.named_parameters() if p.requires_grad)
R["receiver_requires_grad_any"] = any(p.requires_grad for n, p in pm.named_parameters() if ".receiver." in n)
opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=5e-4, weight_decay=0.0)

def answer_acc(adapter):
    pm.set_adapter(adapter); pm.eval(); ok = 0; outs = {}
    with torch.inference_mode():
        for p, a in PAIRS:
            x = tok(p + "\nanswer:", return_tensors="pt", add_special_tokens=False).to(DEV)
            g = pm.generate(**x, max_new_tokens=4, do_sample=False, temperature=None, top_p=None, top_k=None)
            s = tok.decode(g[0, x.input_ids.shape[1]:], skip_special_tokens=True).strip()
            outs[p] = s; ok += (s == a)
    pm.set_adapter("sender")
    return ok / len(PAIRS), outs

R["acc_before_sender"], R["outs_before_sender"] = answer_acc("sender")

def run_steps(bs, seq_len, steps, train_log=None):
    pm.train()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    for s in range(steps):
        batch = [random.choice(PAIRS) for _ in range(bs)]
        x, y, m = encode_batch(batch, seq_len)
        torch.cuda.synchronize(); t = time.perf_counter()
        out = pm(input_ids=x, attention_mask=m, labels=y)
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        losses.append(float(out.loss))
        del out
        if (s + 1) % 10 == 0: print(f"  step {s+1}/{steps} loss {losses[-1]:.4f} step_s {times[-1]:.3f}", flush=True)
    return times, losses, torch.cuda.max_memory_allocated() / 2**20

try:
    times, losses, peak = run_steps(8, 128, 50)
    R["train"] = {"batch": 8, "seq_len": 128, "steps": 50, "lr": 5e-4, "optimizer": "AdamW",
                  "loss_step1": losses[0], "loss_step5": losses[4], "loss_step10": losses[9], "loss_step25": losses[24], "loss_step50": losses[49],
                  "mean_loss_first5": sum(losses[:5]) / 5, "mean_loss_last5": sum(losses[-5:]) / 5,
                  "losses": losses, "step_time_s_first": times[0], "step_time_s_median_after_warmup": sorted(times[5:])[len(times[5:]) // 2],
                  "step_time_s_mean_after_warmup": sum(times[5:]) / len(times[5:]), "peak_allocated_MiB": peak,
                  "tokens_per_step_padded": 8 * 128}
    print(json.dumps(R["train"], indent=1), flush=True)
except Exception:
    R["train_error"] = traceback.format_exc()
    print(R["train_error"])

R["acc_after_sender"], R["outs_after_sender"] = answer_acc("sender")
R["acc_after_receiver_untrained"], R["outs_after_receiver"] = answer_acc("receiver")
print("acc before/after sender, receiver:", R["acc_before_sender"], R["acc_after_sender"], R["acc_after_receiver_untrained"], flush=True)

# ---- sweep for batch-size defaults (3 steps each, same optimiser) ----
sweep = []
for bs, L in [(8, 128), (16, 128), (32, 128), (8, 512), (16, 256), (64, 64)]:
    try:
        torch.cuda.empty_cache()
        tms, _, pk = run_steps(bs, L, 4)
        sweep.append({"batch": bs, "seq_len": L, "step_s": sorted(tms[1:])[len(tms[1:]) // 2], "peak_MiB": pk, "tokens_per_s": bs * L / sorted(tms[1:])[len(tms[1:]) // 2]})
    except torch.cuda.OutOfMemoryError:
        sweep.append({"batch": bs, "seq_len": L, "oom": True}); opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    except Exception:
        sweep.append({"batch": bs, "seq_len": L, "error": traceback.format_exc()[-400:]})
    print("sweep so far:", sweep[-1], flush=True)
R["sweep"] = sweep
print(json.dumps(sweep, indent=1), flush=True)
R["driver_used_MiB_end"] = driver_used_mib()
save("p2_c_train.json", R)
