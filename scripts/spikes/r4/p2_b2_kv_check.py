"""Is the bf16 mismatch between generate-with-prefix-cache and no-cache generate just numerical noise? Check in bf16 and fp32."""
import copy, gc, traceback
from p2_common import *  # noqa
import torch
from peft import LoraConfig, get_peft_model
R = {}
for dtype in (torch.bfloat16, torch.float32):
    name = str(dtype)
    tok, model, _ = load_base(dtype); model.eval()
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg, adapter_name="sender"); pm.add_adapter("receiver", cfg); pm.eval()
    s_ids = tok("Red eats in the top-left corner, quickly.", return_tensors="pt").input_ids.to(DEV)
    r_ids = tok("Understood. My next move is to", return_tensors="pt").input_ids.to(DEV)
    Ts, Tr = s_ids.shape[1], r_ids.shape[1]
    full_ids = torch.cat([s_ids, r_ids], 1)
    kw = dict(max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None)
    r = {}
    with torch.inference_mode():
        pm.set_adapter("sender"); cache = pm(input_ids=s_ids, use_cache=True).past_key_values
        pm.set_adapter("receiver")
        ro = pm(input_ids=r_ids, past_key_values=copy.deepcopy(cache), attention_mask=torch.ones(1, Ts + Tr, device=DEV, dtype=torch.long), use_cache=True)
        full = pm(input_ids=full_ids).logits[:, Ts:]
        r["max_abs_logit_diff"] = float((full.float() - ro.logits.float()).abs().max())
        r["logit_abs_max"] = float(full.float().abs().max())
        ref1 = pm.generate(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **kw)
        ref2 = pm.generate(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **kw)
        g = pm.generate(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), past_key_values=copy.deepcopy(cache), **kw)
        r["reference_deterministic"] = bool(torch.equal(ref1, ref2))
        r["cache_gen_equals_reference"] = bool(torch.equal(g, ref1))
        a, b = g[0, full_ids.shape[1]:].tolist(), ref1[0, full_ids.shape[1]:].tolist()
        r["first_divergence_idx"] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        r["text_with_cache"] = tok.decode(a); r["text_reference"] = tok.decode(b)
        # now make the SENDER adapter non-trivial (B != 0) so its K/V differ from the receiver's would-be K/V
        for n, p in pm.named_parameters():
            if ".sender." in n and "lora_B" in n:
                p.data.normal_(0, 0.02)
        pm.set_adapter("sender"); cache2 = pm(input_ids=s_ids, use_cache=True).past_key_values
        pm.set_adapter("receiver")
        ro2 = pm(input_ids=r_ids, past_key_values=copy.deepcopy(cache2), attention_mask=torch.ones(1, Ts + Tr, device=DEV, dtype=torch.long)).logits
        r["nonzero_sender_adapter_logit_change_vs_zero_adapter_cache_(max_abs)"] = float((ro2.float() - ro.logits.float()).abs().max())
    R[name] = r
    print(name, json.dumps(r, indent=1), flush=True)
    del pm, model, cache, cache2; gc.collect(); torch.cuda.empty_cache()
save("p2_b2_kv_check.json", R)
