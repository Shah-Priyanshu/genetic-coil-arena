"""Part 2 step 4-5: two LoRA adapters on one shared base; adapter switching; latent-message mechanics."""
import copy, time, traceback
from p2_common import *  # noqa
import torch, peft, transformers
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

R = {"peft": peft.__version__, "transformers": transformers.__version__}
torch.cuda.reset_peak_memory_stats()
tok, model, load_s = load_base()
model.eval()
R["mem_base"] = mem()

cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
pm = get_peft_model(model, cfg, adapter_name="sender")
pm.eval()
R["mem_1_adapter"] = mem()
pm.add_adapter("receiver", cfg)
pm.eval()
R["mem_2_adapters"] = mem()
R["adapter_delta_bytes"] = {
    "first_adapter": R["mem_1_adapter"]["allocated_MiB"] * 2**20 - R["mem_base"]["allocated_MiB"] * 2**20,
    "second_adapter": R["mem_2_adapters"]["allocated_MiB"] * 2**20 - R["mem_1_adapter"]["allocated_MiB"] * 2**20,
}
base_bytes = sum(p.numel() * p.element_size() for n, p in pm.named_parameters() if "lora_" not in n)
lora_params = {a: sum(p.numel() for n, p in pm.named_parameters() if f".{a}." in n) for a in ("sender", "receiver")}
lora_bytes = {a: sum(p.numel() * p.element_size() for n, p in pm.named_parameters() if f".{a}." in n) for a in ("sender", "receiver")}
R["lora_param_counts"] = lora_params
R["lora_param_bytes"] = lora_bytes
R["lora_dtype"] = str(next(p for n, p in pm.named_parameters() if ".sender." in n).dtype)
R["base_weight_bytes"] = base_bytes
R["if_base_were_duplicated_extra_bytes"] = base_bytes
# base weights are literally the same tensors regardless of active adapter:
w = pm.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight
R["base_q_proj_data_ptr_same_after_switch"] = None
p0 = w.data_ptr()
pm.set_adapter("receiver"); p1 = pm.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight.data_ptr()
pm.set_adapter("sender"); p2 = pm.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight.data_ptr()
R["base_q_proj_data_ptr_same_after_switch"] = (p0 == p1 == p2)
print(json.dumps({k: R[k] for k in R if k.startswith(("mem", "adapter", "lora", "base", "if_"))}, indent=1, default=str), flush=True)

# ---- switching latency ----
S = {}
try:
    def toggle():
        pm.set_adapter("receiver"); pm.set_adapter("sender")
    for _ in range(10): toggle()
    n = 300
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): toggle()
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    S["set_adapter_ms_per_call"] = dt / (2 * n) * 1000
    ids = make_ids(tok, 128)
    with torch.inference_mode():
        same = timeit(lambda: pm(input_ids=ids, logits_to_keep=1))
        def sw_fwd():
            pm.set_adapter("receiver"); pm(input_ids=ids, logits_to_keep=1)
            pm.set_adapter("sender"); pm(input_ids=ids, logits_to_keep=1)
        alt = timeit(sw_fwd)
        S["fwd128_same_adapter_ms"] = same["median_ms"]
        S["fwd128_alternating_adapters_ms_per_fwd"] = alt["median_ms"] / 2
        with pm.disable_adapter():
            S["fwd128_adapter_disabled_ms"] = timeit(lambda: pm(input_ids=ids, logits_to_keep=1))["median_ms"]
    # mixed-adapter batch (different adapter per row)
    try:
        ids2 = make_ids(tok, 128, bs=2)
        with torch.inference_mode():
            o = pm(input_ids=ids2, adapter_names=["sender", "receiver"], logits_to_keep=1)
            S["mixed_adapter_batch_works"] = True
            S["mixed_batch_bs2_fwd128_ms"] = timeit(lambda: pm(input_ids=ids2, adapter_names=["sender", "receiver"], logits_to_keep=1))["median_ms"]
            S["plain_batch_bs2_fwd128_ms"] = timeit(lambda: pm(input_ids=ids2, logits_to_keep=1))["median_ms"]
    except Exception:
        S["mixed_adapter_batch_works"] = False
        S["mixed_adapter_batch_error"] = traceback.format_exc()[-800:]
except Exception:
    S["error"] = traceback.format_exc()
R["switching"] = S
print(json.dumps(S, indent=1, default=str), flush=True)

# ---- latent message mechanics ----
M = {}
try:
    s_ids = tok("Red eats in the top-left corner, quickly.", return_tensors="pt").input_ids.to(DEV)
    r_ids = tok("Understood. My next move is to", return_tensors="pt").input_ids.to(DEV)
    Ts, Tr = s_ids.shape[1], r_ids.shape[1]
    M["sender_tokens"], M["receiver_tokens"] = Ts, Tr
    with torch.inference_mode():
        pm.set_adapter("sender")
        so = pm(input_ids=s_ids, output_hidden_states=True, use_cache=True)
        hs = so.hidden_states
        mid = len(hs) // 2
        M["n_hidden_tensors"] = len(hs); M["mid_index"] = mid; M["last_index"] = len(hs) - 1
        h_mid = hs[mid][:, -1, :]; h_last = hs[-1][:, -1, :]
        M["h_mid_shape"], M["h_last_shape"] = list(h_mid.shape), list(h_last.shape)
        emb = pm.get_input_embeddings()
        r_emb = emb(r_ids)
        M["receiver_embeds_shape"] = list(r_emb.shape)
        e_norm = float(emb.weight.float().norm(dim=-1).mean())
        M["norms"] = {"embedding_row_mean": e_norm, "h_mid": float(h_mid.float().norm()), "h_last": float(h_last.float().norm()),
                      "r_emb_tokens_mean": float(r_emb.float().norm(dim=-1).mean()),
                      "h_mid_max_abs_dim": float(h_mid.float().abs().max()), "h_last_max_abs_dim": float(h_last.float().abs().max())}
        pm.set_adapter("receiver")
        mask0 = torch.ones(1, Tr, device=DEV, dtype=torch.long)
        base_logits = pm(inputs_embeds=r_emb, attention_mask=mask0).logits[:, -1].float()
        for name, h in (("mid", h_mid), ("last", h_last)):
            msg = h.unsqueeze(1).to(r_emb.dtype)                       # [1,1,H]
            inp = torch.cat([msg, r_emb], dim=1)                       # [1,1+Tr,H]
            mask = torch.ones(1, Tr + 1, device=DEV, dtype=torch.long)
            o = pm(inputs_embeds=inp, attention_mask=mask)
            gen = pm.generate(inputs_embeds=inp, attention_mask=mask, max_new_tokens=12, do_sample=False,
                              temperature=None, top_p=None, top_k=None)
            lp = F.log_softmax(o.logits[:, -1].float(), -1); lb = F.log_softmax(base_logits, -1)
            kl = float(F.kl_div(lb, lp, log_target=True, reduction="batchmean"))  # KL(with_msg || without)
            # scaled variant: match embedding norm
            msg_s = (h / h.float().norm() * e_norm).unsqueeze(1).to(r_emb.dtype)
            o_s = pm(inputs_embeds=torch.cat([msg_s, r_emb], 1), attention_mask=mask)
            kl_s = float(F.kl_div(lb, F.log_softmax(o_s.logits[:, -1].float(), -1), log_target=True, reduction="batchmean"))
            M[name] = {"msg_shape": list(msg.shape), "inputs_embeds_shape": list(inp.shape),
                       "logits_shape": list(o.logits.shape), "generate_out_shape": list(gen.shape),
                       "generated_text": tok.decode(gen[0], skip_special_tokens=True),
                       "kl_vs_no_message_raw": kl, "kl_vs_no_message_norm_matched": kl_s,
                       "top1_changed_raw": bool(o.logits[:, -1].argmax() != base_logits.argmax())}
        M["no_message_generate_text"] = tok.decode(pm.generate(inputs_embeds=r_emb, attention_mask=mask0, max_new_tokens=12, do_sample=False,
                                                           temperature=None, top_p=None, top_k=None)[0], skip_special_tokens=True)
        # added latency
        msg = h_mid.unsqueeze(1).to(r_emb.dtype); inp = torch.cat([msg, r_emb], 1); mask = torch.ones(1, Tr + 1, device=DEV, dtype=torch.long)
        def sender_pass():
            pm.set_adapter("sender")
            o = pm(input_ids=s_ids, output_hidden_states=True, logits_to_keep=1)
            return o.hidden_states[mid][:, -1, :]
        def recv_with():
            pm.set_adapter("receiver"); pm(inputs_embeds=inp, attention_mask=mask, logits_to_keep=1)
        def recv_without():
            pm.set_adapter("receiver"); pm(inputs_embeds=r_emb, attention_mask=mask0, logits_to_keep=1)
        def full_handoff():
            h = sender_pass()
            pm.set_adapter("receiver")
            e = emb(r_ids)
            pm(inputs_embeds=torch.cat([h.unsqueeze(1).to(e.dtype), e], 1), attention_mask=mask, logits_to_keep=1)
        M["latency_ms"] = {"sender_pass_(Ts_tokens,+hs)": timeit(sender_pass)["median_ms"],
                           "receiver_with_msg_(Tr+1)": timeit(recv_with)["median_ms"],
                           "receiver_without_msg_(Tr)": timeit(recv_without)["median_ms"],
                           "full_handoff_sender+receiver": timeit(full_handoff)["median_ms"]}
        M["latency_ms"]["receiver_added_by_msg"] = M["latency_ms"]["receiver_with_msg_(Tr+1)"] - M["latency_ms"]["receiver_without_msg_(Tr)"]
        # sender with a 128-token message
        s128 = make_ids(tok, 128)
        def sender128():
            pm.set_adapter("sender"); return pm(input_ids=s128, output_hidden_states=True, logits_to_keep=1)
        M["latency_ms"]["sender_pass_128tok"] = timeit(sender128)["median_ms"]
except Exception:
    M["error"] = traceback.format_exc()
R["latent_message"] = M
print(json.dumps(M, indent=1, default=str), flush=True)

# ---- KV prefix from sender into receiver ----
P = {}
try:
    with torch.inference_mode():
        pm.set_adapter("sender")
        so = pm(input_ids=s_ids, use_cache=True)
        cache = so.past_key_values
        P["sender_cache_type"] = type(cache).__name__
        P["sender_cache_seq_len"] = int(cache.get_seq_length()); P["sender_cache_bytes"] = kv_bytes(cache)
        pm.set_adapter("receiver")
        # (a) manual forward with prefix cache
        c2 = copy.deepcopy(cache)
        mask = torch.ones(1, Ts + Tr, device=DEV, dtype=torch.long)
        ro = pm(input_ids=r_ids, past_key_values=c2, attention_mask=mask, use_cache=True)
        P["forward_with_prefix_cache_works"] = True
        P["logits_shape"] = list(ro.logits.shape)
        P["cache_len_after"] = int(ro.past_key_values.get_seq_length())
        # (b) reference: full concat forward under receiver. Adapters are zero-initialised (B=0) so must match.
        full = pm(input_ids=torch.cat([s_ids, r_ids], 1)).logits[:, Ts:]
        P["max_abs_logit_diff_vs_full_concat"] = float((full.float() - ro.logits.float()).abs().max())
        P["argmax_agree_all_positions"] = bool((full.argmax(-1) == ro.logits.argmax(-1)).all())
        # (c) generate with prefix cache
        try:
            c3 = copy.deepcopy(cache)
            ids_full = torch.cat([s_ids, r_ids], 1)
            g = pm.generate(input_ids=ids_full, attention_mask=torch.ones_like(ids_full), past_key_values=c3,
                            max_new_tokens=12, do_sample=False, temperature=None, top_p=None, top_k=None)
            P["generate_with_prefix_cache_works"] = True
            P["generated_text"] = tok.decode(g[0, ids_full.shape[1]:], skip_special_tokens=True)
            gref = pm.generate(input_ids=ids_full, attention_mask=torch.ones_like(ids_full), max_new_tokens=12, do_sample=False,
                               temperature=None, top_p=None, top_k=None)
            P["generate_matches_no_cache_reference"] = bool(torch.equal(g, gref))
        except Exception:
            P["generate_with_prefix_cache_works"] = False
            P["generate_error"] = traceback.format_exc()[-900:]
        # (d) cost: 512-token sender cache handed to a short receiver turn vs recompute
        s512 = make_ids(tok, 512)
        pm.set_adapter("sender")
        c512 = pm(input_ids=s512, use_cache=True).past_key_values
        P["cache512_bytes"] = kv_bytes(c512)
        t = time.perf_counter(); cc = copy.deepcopy(c512); torch.cuda.synchronize(); P["deepcopy_512_ms"] = (time.perf_counter() - t) * 1000
        pm.set_adapter("receiver")
        m512 = torch.ones(1, 512 + Tr, device=DEV, dtype=torch.long)
        def with_cache():
            c = copy.deepcopy(c512)
            pm(input_ids=r_ids, past_key_values=c, attention_mask=m512, use_cache=True, logits_to_keep=1)
        def recompute():
            pm(input_ids=torch.cat([s512, r_ids], 1), logits_to_keep=1)
        P["receiver_with_512_prefix_cache_ms_(incl_deepcopy)"] = timeit(with_cache)["median_ms"]
        P["receiver_recompute_512+Tr_ms"] = timeit(recompute)["median_ms"]
        # (e) cache from sender with a trained-looking adapter would differ; here just record that adapter swap in between is fine
except Exception:
    P["error"] = traceback.format_exc()
R["kv_prefix"] = P
print(json.dumps(P, indent=1, default=str), flush=True)
R["peak"] = mem()
save("p2_b_lora.json", R)
