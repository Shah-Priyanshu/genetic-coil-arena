"""Part 2 step 2-3: load Qwen3-1.7B bf16, VRAM, generation speed, hidden states, KV cache."""
import time, traceback
from p2_common import *  # noqa
import torch, transformers

R = {"torch": torch.__version__, "transformers": transformers.__version__,
     "gpu": torch.cuda.get_device_name(0)}
R["driver_used_before_load_MiB"] = driver_used_mib()
torch.cuda.reset_peak_memory_stats()
tok, model, load_s = load_base()
model.eval()
R["load_seconds"] = load_s
R["vram_after_load"] = mem()
R["driver_delta_after_load_MiB"] = R["vram_after_load"]["driver_used_MiB"] - R["driver_used_before_load_MiB"]
cfg = model.config
R["config"] = {k: getattr(cfg, k, None) for k in [
    "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
    "vocab_size", "intermediate_size", "tie_word_embeddings", "max_position_embeddings",
    "use_sliding_window", "torch_dtype", "dtype"]}
R["attn_implementation"] = getattr(cfg, "_attn_implementation", None)
R["n_params"] = sum(p.numel() for p in model.parameters())
R["weights_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
print(json.dumps(R, indent=1, default=str), flush=True)

# ---- generation speed ----
try:
    G = {}
    prompt = tok("The squad of three snakes cooperates to", return_tensors="pt").to(DEV)
    kw = dict(max_new_tokens=128, min_new_tokens=128, do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.inference_mode():
        model.generate(**prompt, **kw)  # warm-up
        runs = []
        for _ in range(3):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = model.generate(**prompt, **kw)
            torch.cuda.synchronize(); runs.append(time.perf_counter() - t)
        G["bs1_new_tokens"] = int(out.shape[1] - prompt.input_ids.shape[1])
        G["bs1_seconds_runs"] = runs
        G["bs1_tokens_per_s_median"] = 128 / sorted(runs)[1]
        p8 = {k: v.repeat(8, 1) for k, v in prompt.items()}
        model.generate(**p8, **kw)
        runs8 = []
        for _ in range(3):
            torch.cuda.synchronize(); t = time.perf_counter()
            model.generate(**p8, **kw)
            torch.cuda.synchronize(); runs8.append(time.perf_counter() - t)
        G["bs8_seconds_runs"] = runs8
        G["bs8_aggregate_tokens_per_s_median"] = 8 * 128 / sorted(runs8)[1]
    G["sample_text"] = tok.decode(out[0], skip_special_tokens=True)[:200]
    R["generation"] = G
except Exception:
    R["generation_error"] = traceback.format_exc()
print(json.dumps(R.get("generation", R.get("generation_error")), indent=1), flush=True)

# ---- hidden states ----
H = {}
try:
    with torch.inference_mode():
        for L in (128, 512):
            ids = make_ids(tok, L)
            out = model(input_ids=ids, output_hidden_states=True)
            hs = out.hidden_states
            H[f"L{L}"] = {
                "n_hidden_state_tensors": len(hs), "shape_each": list(hs[0].shape), "dtype": str(hs[0].dtype),
                "logits_shape": list(out.logits.shape), "type": type(hs).__name__,
                "fwd_full_logits_no_hs": timeit(lambda: model(input_ids=ids)),
                "fwd_full_logits_with_hs": timeit(lambda: model(input_ids=ids, output_hidden_states=True)),
                "fwd_logits_to_keep1_with_hs": timeit(lambda: model(input_ids=ids, output_hidden_states=True, logits_to_keep=1)),
                "fwd_backbone_with_hs": timeit(lambda: model.model(input_ids=ids, output_hidden_states=True)),
            }
            H[f"L{L}"]["hs_extra_bytes"] = sum(h.numel() * h.element_size() for h in hs)
            # is the last element the post-norm state? compare with backbone last_hidden_state
            bb = model.model(input_ids=ids, output_hidden_states=True)
            H[f"L{L}"]["last_hs_equals_last_hidden_state"] = bool(torch.equal(bb.hidden_states[-1], bb.last_hidden_state))
            H[f"L{L}"]["second_last_equals_last_hidden_state"] = bool(torch.equal(bb.hidden_states[-2], bb.last_hidden_state))
        # scale of states at last token
        ids = make_ids(tok, 128)
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        emb = model.get_input_embeddings().weight
        H["norms"] = {"embedding_row_norm_mean": float(emb.float().norm(dim=-1).mean()),
                      "per_layer_last_token_norm": [float(h[0, -1].float().norm()) for h in hs]}
except Exception:
    H["error"] = traceback.format_exc()
R["hidden_states"] = H
print(json.dumps(H, indent=1, default=str)[:3500], flush=True)

# ---- KV cache ----
K = {}
try:
    with torch.inference_mode():
        for L in (128, 512):
            ids = make_ids(tok, L)
            torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated()
            out = model(input_ids=ids, use_cache=True)
            cache = out.past_key_values
            kvs = kv_tensors(cache)
            b = kv_bytes(cache)
            K[f"L{L}"] = {"cache_type": type(cache).__name__, "len(cache)": len(cache),
                          "seq_length": int(cache.get_seq_length()) if hasattr(cache, "get_seq_length") else None,
                          "layer_type": type(cache.layers[0]).__name__ if hasattr(cache, "layers") else None,
                          "per_layer_key_shape": list(kvs[0][0].shape), "per_layer_value_shape": list(kvs[0][1].shape),
                          "dtype": str(kvs[0][0].dtype), "n_layers": len(kvs),
                          "total_bytes": b, "bytes_per_token": b / L, "total_MiB": b / 2**20}
            del out, cache
        cfgd = model.config
        hd = getattr(cfgd, "head_dim", cfgd.hidden_size // cfgd.num_attention_heads)
        K["formula_bytes_per_token"] = 2 * cfgd.num_hidden_layers * cfgd.num_key_value_heads * hd * 2
        K["fp8_or_int8_bytes_per_token_if_quantised"] = K["formula_bytes_per_token"] / 2
        # batch 8 x 512 cache memory, for sizing
        K["bs8_L512_MiB"] = 8 * K["L512"]["total_MiB"]
except Exception:
    K["error"] = traceback.format_exc()
R["kv_cache"] = K
print(json.dumps(K, indent=1, default=str), flush=True)
R["peak_vram_whole_script"] = mem()
save("p2_a_base.json", R)
