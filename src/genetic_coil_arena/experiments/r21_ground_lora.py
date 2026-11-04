"""R21: editing the weights of the ground station's language model (goal G17).

R16's ground station asks qwen2.5-coder:14b, unedited, to fill one ``swarm.behavior``
command, and code does the rest (``r16_ground_station.GroundTranslator``, v5). On G14b's
fresh sealed set it missed the exact-match bar narrowly (88.9%, bar 90%). R7a's distilled
students failed on hand-written requests because they were trained on template-like
phrasing (G3b). This module fine-tunes a smaller open-weight model of the same family,
Qwen2.5-Coder-7B-Instruct, with QLoRA on requests written like real ones: five writers
with different styles (``train_a..e.jsonl`` in ``DATA_DIR``), each given only the G14b
label guide, never the sealed sets.

The model sees exactly the prompt the 14B sees (system prompt, the nine few-shot
examples, the request) and is trained on the reply alone: ``{"refusal", "clarify",
"params"}`` with only the values the operator stated (``target_reply``). The same 7B
unedited, with the same prompt and the same decoding (``HFChatBackend``: greedy, the
first JSON object in the reply), is the "as is" system. Both run inside the unchanged v5
``GroundTranslator``, so only the weights differ.

Training texts must pass the n-gram leak check against the dev and both sealed sets
(``check_leaks``) and every ``ok`` label must validate with its defaults filled
(``check_labels``).
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any

from ..backends.base import CAP_GENERATE, GenerationResult
from ..envs.swarm import ground as gs
from . import r15_scout_brain as r15
from . import r16_ground_station as r16

DATA_DIR = r15.DATA_DIR / "r21_ground_lora"
HF_CACHE = Path(r"D:\hf_cache")
BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
WRITERS = ("a", "b", "c", "d", "e")
LEAK_THRESHOLD = 0.5  # R7a's: a training text may share at most half a held-out text's 4-grams
MAX_NEW = 160

QUESTIONS = {
    "search_grid": "Which area should the swarm search?",
    "perimeter": "Which place should the swarm circle?",
}
REFUSALS = (
    ("out-of-domain", "Not a swarm command."),
    ("two-behaviours", "One behaviour at a time, please."),
    ("single-drone", "Commands go to the whole swarm, not one drone."),
    ("range", "That value is outside the allowed range."),
    ("wrong-param", "That behaviour has no such setting."),
)


def load_train(writers: tuple[str, ...] = WRITERS) -> list[dict[str, Any]]:
    items = []
    for w in writers:
        items += r16.load_items(DATA_DIR / f"train_{w}.jsonl")
    return items


def target_reply(item: dict[str, Any]) -> str:
    """The reply the model is trained to give, in the few-shot examples' format."""
    tags = set(item.get("tags", ()))
    if item["expect"] == "ok":
        return json.dumps({"refusal": "", "clarify": "", "params": item["params"]})
    if item["expect"] == "clarify":
        q = next((QUESTIONS[t] for t in QUESTIONS if t in tags),
                 "Which behaviour should the swarm fly?")  # fmt: skip
        return json.dumps({"refusal": "", "clarify": q})
    r = next((text for tag, text in REFUSALS if tag in tags), "The swarm cannot do that.")
    return json.dumps({"refusal": r, "clarify": ""})


def check_labels(items: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """``(id, problem)`` for every label that breaks the guide's conventions: an ``ok``
    whose params (defaults filled) fail the spec, an unknown place, or a duplicate text."""
    registry = r16.build_registry()
    pack = r16.pack()
    places = {p.name for p in gs.SITE}
    seen: dict[str, str] = {}
    bad = []
    for it in items:
        key = re.sub(r"\W+", " ", it["text"].lower()).strip()
        if key in seen:
            bad.append((it["id"], f"duplicate of {seen[key]}"))
        seen[key] = it["id"]
        if it["expect"] not in ("ok", "clarify", "refuse"):
            bad.append((it["id"], f"expect {it['expect']!r}"))
            continue
        if it["expect"] != "ok":
            if "params" in it:
                bad.append((it["id"], "params on a non-ok item"))
            continue
        p = dict(it.get("params") or {})
        if "area" in p and p["area"] not in places:
            bad.append((it["id"], f"area {p['area']!r} is not canonical"))
            continue
        try:
            registry.make(pack.spec_name, gs.fill_behavior_defaults(p), version=pack.version)
        except Exception as exc:  # noqa: BLE001 - every failure is a label problem
            bad.append((it["id"], f"{type(exc).__name__}: {str(exc)[:120]}"))
            continue
        loose = gs.unsupported(p, it["text"])
        if loose:
            bad.append((it["id"], "not in the text: " + "; ".join(loose)))
    return bad


def held_out_texts() -> list[str]:
    """Dev and both sealed sets' texts (read by code for the leak check only)."""
    from . import r16_sealed

    items = r16_sealed.load_dev() + r16_sealed.load_sealed() + r16_sealed.load_sealed(b=True)
    return [it["text"] for it in items]


def check_leaks(items: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """``(id, containment)`` of training texts too close to a held-out text."""
    from ..translator.distill import OverlapIndex

    index = OverlapIndex(held_out_texts(), n=4)
    out = []
    for it in items:
        score, _ = index.max_containment(it["text"])
        if score >= LEAK_THRESHOLD:
            out.append((it["id"], score))
    return out


# ----------------------------------------------------------------------------- model


def messages(system: str, examples: Any, prompt: str) -> list[dict[str, str]]:
    """The chat ``OllamaBackend.request_body`` sends: system, few-shot pairs, request."""
    out = [{"role": "system", "content": system}]
    for user, assistant in examples:
        out += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return out + [{"role": "user", "content": prompt}]


def first_json(text: str) -> str:
    """The first balanced ``{...}`` in ``text`` (code fences and chatter dropped), or the
    text itself when there is none."""
    start = text.find("{")
    if start < 0:
        return text
    depth, quoted, escape = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if quoted:
            escape = c == "\\" and not escape
            if c == '"' and not escape:
                quoted = False
            continue
        if c == '"':
            quoted = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def load_model(adapter: Path | None = None, train: bool = False) -> tuple[Any, Any]:
    """The 7B in 4-bit (NF4, bf16 compute), with ``adapter`` loaded if given."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.cuda.set_per_process_memory_fraction(0.85)
    tok = AutoTokenizer.from_pretrained(BASE, cache_dir=HF_CACHE / "hub")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)  # fmt: skip
    net = AutoModelForCausalLM.from_pretrained(BASE, cache_dir=HF_CACHE / "hub",
                                               quantization_config=quant,
                                               dtype=torch.bfloat16, device_map={"": 0})  # fmt: skip
    if adapter is not None:
        from peft import PeftModel

        net = PeftModel.from_pretrained(net, adapter, is_trainable=train)
    return net, tok


class HFChatBackend:
    """``generate`` like ``OllamaBackend`` (system, few-shot pairs, request) on a local
    HF model: greedy decoding, the reply cut to its first JSON object (``first_json``).
    No grammar is applied; the schema argument is accepted and ignored."""

    capabilities = frozenset({CAP_GENERATE})

    def __init__(self, net: Any, tok: Any, name: str, max_new_tokens: int = MAX_NEW):
        self.net, self.tok, self.base_id = net, tok, name
        self.max_new_tokens = max_new_tokens

    def generate(self, prompts: Any, adapter_id: Any = None, *, system: str | None = None,
                 examples: Any = (), schema: Any = None, max_new_tokens: int | None = None,
                 keep_alive: Any = None, options: Any = None) -> list[GenerationResult]:  # fmt: skip
        import torch

        out = []
        for prompt in prompts:
            ids = self.tok.apply_chat_template(messages(system or "", examples, prompt),
                                               add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)  # fmt: skip
            ids = {k: v.to(self.net.device) for k, v in ids.items()}
            start = time.perf_counter()
            with torch.no_grad():
                gen = self.net.generate(**ids, max_new_tokens=max_new_tokens or
                                        self.max_new_tokens, do_sample=False,
                                        pad_token_id=self.tok.eos_token_id)  # fmt: skip
            new = gen[0, ids["input_ids"].shape[1] :]
            text = self.tok.decode(new, skip_special_tokens=True)
            out.append(GenerationResult(text=first_json(text),
                                        n_prompt_tokens=int(ids["input_ids"].shape[1]),
                                        n_new_tokens=int(len(new)),
                                        finish_reason="stop",
                                        latency_s=time.perf_counter() - start,
                                        meta={"full": text}))  # fmt: skip
        return out


def encode(tok: Any, item: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Token ids of the prompt plus the target reply, and labels (-100 on the prompt)."""
    pack = r16.pack()
    shots = [(e.text, e.reply(policy=True)) for e in pack.examples]
    msgs = messages(pack.system, shots, item["text"])
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(prompt, "keys"):  # transformers 5 returns a BatchEncoding
        prompt = prompt["input_ids"]
    full = tok.apply_chat_template(msgs + [{"role": "assistant",
                                            "content": target_reply(item)}],
                                   tokenize=True)  # fmt: skip
    if hasattr(full, "keys"):
        full = full["input_ids"]
    full = list(full)
    while full and full[-1] != tok.convert_tokens_to_ids("<|im_end|>"):
        full.pop()  # drop the template's trailing newline after <|im_end|>
    assert full[: len(prompt)] == list(prompt), "chat template is not prefix-stable"
    return full, [-100] * len(prompt) + full[len(prompt) :]


def train_lora(items: list[dict[str, Any]], dev: list[dict[str, Any]], out: Path,
               seed: int = 0, epochs: int = 3, rank: int = 16, lr: float = 1e-4,
               accum: int = 8, log: Any = print) -> list[dict[str, Any]]:  # fmt: skip
    """QLoRA on every linear projection; after each epoch, the dev set through the v5
    ``GroundTranslator``. Each epoch's adapter is saved under ``out/epoch<k>``; returns
    the per-epoch dev summaries (the caller picks the epoch)."""
    import torch
    from peft import LoraConfig, get_peft_model

    random.seed(seed)
    torch.manual_seed(seed)
    net, tok = load_model()
    # Not peft's prepare_model_for_kbit_training: it upcasts the untied embedding and
    # output layers to fp32 (4.4 GB), which runs the 12 GB card out of memory.
    net.config.use_cache = False
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    net.enable_input_require_grads()
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                     "up_proj", "down_proj"])  # fmt: skip
    net = get_peft_model(net, cfg)
    data = [encode(tok, it) for it in items]
    log(f"{len(data)} training sequences, {max(len(d[0]) for d in data)} tokens at most")
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * (len(data) // accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.05)  # fmt: skip
    hist = []
    for ep in range(epochs):
        net.train()
        order = list(range(len(data)))
        random.shuffle(order)
        start, total = time.perf_counter(), 0.0
        for s in range(len(data) // accum):
            for k in order[s * accum : (s + 1) * accum]:
                loss = reply_loss(net, *data[k]) / accum
                loss.backward()
                total += float(loss)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if s % 10 == 0:
                log(f"[progress] epoch {ep + 1} step {s}/{len(data) // accum}: loss "
                    f"{total / (s + 1):.4f} ({time.perf_counter() - start:.0f} s)")  # fmt: skip
        net.eval()
        net.save_pretrained(out / f"epoch{ep + 1}")
        rows = run_dev(net, tok, dev, f"epoch {ep + 1}", log)
        rec = {"epoch": ep + 1, "loss": total / max(len(data) // accum, 1), **dev_summary(rows)}
        hist.append(rec)
        log(f"epoch {ep + 1}: dev exact {rec['exact']}/{rec['n_ok']}, wrong commands "
            f"{rec['wrong']}, correct {rec['correct']}/{rec['n']}")  # fmt: skip
    return hist


def reply_loss(net: Any, ids: list[int], labels: list[int]) -> Any:
    """Cross-entropy on the reply tokens only, with logits computed only where they are
    needed (``logits_to_keep``): the logit at position t predicts token t + 1."""
    import torch
    import torch.nn.functional as F

    start = next(i for i, v in enumerate(labels) if v != -100)
    x = torch.tensor([ids], device=net.device)
    logits = net(input_ids=x, logits_to_keep=len(ids) - start + 1).logits[0, :-1]
    y = torch.tensor(labels[start:], device=net.device)
    return F.cross_entropy(logits.float(), y, ignore_index=-100)


def translator(net: Any, tok: Any, name: str) -> Any:
    return r16.GroundTranslator(HFChatBackend(net, tok, name), r16.build_registry())


def run_items(tr: Any, items: list[dict[str, Any]], tag: str, log: Any = print) -> list[dict]:
    """R16's per-item rows (``scripts/r16_ground_station.run_items``' fields)."""
    rows = []
    for k, item in enumerate(items):
        res = tr.translate(item["text"], r16.pack())
        outcome = r16.score(res, item)
        rows.append({"id": item["id"], "text": item["text"], "expect": item["expect"],
                     "gold": item.get("params"), "tags": item.get("tags", []),
                     "status": res.status, "params": res.spec.params if res.spec else None,
                     "reason": res.reason, "raw": res.raw, "attempts": res.attempts,
                     "latency_s": res.latency_s, "outcome": outcome,
                     "correct": r16.correct(outcome, item["expect"])})  # fmt: skip
        if (k + 1) % 25 == 0:
            log(f"[progress] {tag}: {k + 1}/{len(items)}, "
                f"{sum(r['correct'] for r in rows)} correct")  # fmt: skip
    return rows


def run_dev(net: Any, tok: Any, dev: list[dict[str, Any]], tag: str, log: Any = print) -> list:
    return run_items(translator(net, tok, BASE), dev, tag, log)


def dev_summary(rows: list[dict]) -> dict[str, int]:
    ok = [r for r in rows if r["expect"] == "ok"]
    return {"n": len(rows), "n_ok": len(ok),
            "exact": sum(r["outcome"] == "exact" for r in ok),
            "wrong": sum(r["outcome"] == "wrong command" for r in rows),
            "correct": sum(r["correct"] for r in rows)}  # fmt: skip


def pick_epoch(hist: list[dict[str, Any]]) -> int:
    """Most exact on dev, then fewest wrong commands, then the earlier epoch."""
    return min(hist, key=lambda h: (-h["exact"], h["wrong"], h["epoch"]))["epoch"]
