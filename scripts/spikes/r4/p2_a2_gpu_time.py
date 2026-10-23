"""Separate GPU-busy time from wall time (machine CPU is saturated by another job, so HF eager is launch-bound)."""
import time, traceback
from p2_common import *  # noqa
import torch
from torch.profiler import profile, ProfilerActivity
tok, model, _ = load_base(); model.eval()
R = {}
def gpu_busy_ms(fn, n=5):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            t = time.perf_counter(); fn(); torch.cuda.synchronize(); wall = (time.perf_counter() - t) * 1000
        ev = prof.key_averages()
        from torch.autograd import DeviceType
        busy = sum(e.self_device_time_total for e in ev if e.device_type == DeviceType.CUDA) / 1000
        out.append((wall, busy))
    out.sort()
    w, b = out[len(out) // 2]
    return {"wall_ms_under_profiler": w, "gpu_kernel_busy_ms": b}
with torch.inference_mode():
    for L in (128, 512):
        ids = make_ids(tok, L)
        R[f"fwd_L{L}"] = gpu_busy_ms(lambda: model(input_ids=ids, output_hidden_states=True, logits_to_keep=1))
    # one decode step with a 128-token cache
    ids = make_ids(tok, 128)
    out = model(input_ids=ids, use_cache=True)
    import copy
    nxt = out.logits[:, -1:].argmax(-1)
    def step():
        c = out.past_key_values
        model(input_ids=nxt, past_key_values=c, use_cache=True)
        c.crop(-1)
    R["decode_step_cache128"] = gpu_busy_ms(step)
# real wall: python-side per-token decode
print(json.dumps(R, indent=1))
save("p2_a2_gpu_time.json", R)
