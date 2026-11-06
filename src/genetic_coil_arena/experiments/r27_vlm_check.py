"""R27 (goal G23): can a ground-station vision-language model reject the clear non-persons
the confirmer chain passes, keeping the people?

R17 step 6b found the operator's false-accept rate on empty views to be the search's most
sensitive assumption, and R17 step 6h asks the chain to pass about a third fewer empty
candidates. R20b and R25c showed that most labelled false alarms past a confirmer are
people VisDrone did not label, and labelled the clear non-persons blind. This step asks
Qwen3-VL-8B-Instruct (4-bit NF4, on the ground station's GPU), unchanged, whether a person
is at the centre of each passed candidate's view, as a final automatic check before the
operator.

- **View:** R19's eval view of the 144-px crop (the centre 132 px, bilinear to 448 px).
  Two prompt variants: ``plain`` (R19's prompt, "at the exact centre of the image") and
  ``marked`` (a magenta cross at the centre, as on the label sheets, and the prompt names
  it). Score = logit "Yes" minus logit "No" on the first answer token (R19's).
- **Dev:** R20's and R20b's blind-labelled spots (744 test-dev false alarms of R20's chains,
  native "closer look" crops) plus ``N_PEOPLE_DEV`` VisDrone-labelled people drawn at random
  from R20's candidates. The variant and the threshold are chosen here: the threshold keeps
  ``KEEP`` of the dev people (labelled and VisDrone pooled); the variant rejects the most
  dev clear non-persons at its threshold.
- **Test:** R25c's 1,600 blind-labelled spots (R25b's four chains) plus
  ``N_PEOPLE_TEST`` VisDrone-labelled people passed by each chain, drawn at random; spots
  within ``GAP`` px of a dev spot in the same frame are dropped. Scored once.

Bars, fixed before any VLM score:

- **PASS** if, at the dev threshold, the chosen variant rejects >= 50% of the test clear
  non-persons (Wilson lower bound >= 30%) and keeps >= 93% of the test people (labelled and
  VisDrone pooled), each group >= 90%.
- **KILL** if it rejects < 25% of the test clear non-persons.
- Measured with no bar: unclear spots rejected, AUC people against clear non-persons, the
  same variant with Qwen3-VL-2B-Instruct (R19's model, as is).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import r15_scout_brain as r15
from . import r19_confirmer as rc
from . import r20_closer_look as rl
from . import r25_camera as r25

OUT_DIR = r15.DATA_DIR / "r27_vlm_check"
MODELS = {"8b": "Qwen/Qwen3-VL-8B-Instruct", "2b": rc.QWEN}
SIZE = 448
KEEP = 0.95
GAP = 24.0  # native px: a test spot this close to a dev spot in the same frame is dropped
N_PEOPLE_DEV, N_PEOPLE_TEST, PEOPLE_SEED = 200, 100, 27
PASS_REJECT, PASS_REJECT_LO, PASS_KEEP, PASS_KEEP_GROUP, KILL_REJECT = 0.5, 0.3, 0.93, 0.9, 0.25
PROMPTS = {
    "plain": rc.PROMPT,
    "marked": "A drone camera looks down at this scene. Is there a person (standing, walking, "
    "sitting, or riding a bike, scooter or motorbike) where the magenta cross marks the "
    "centre? Answer Yes or No.",
}
CLASSES = ("person", "not_person", "ambiguous")


def view(crop: np.ndarray, marked: bool):
    """R19's eval view at ``SIZE`` px as a PIL image, with a magenta cross if ``marked``."""
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw

    w = 2 * rc.HALF - 12
    x = torch.from_numpy(np.ascontiguousarray(crop[6 : 6 + w, 6 : 6 + w])).permute(2, 0, 1)
    x = F.interpolate(x[None].float(), size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    img = Image.fromarray(x[0].round().clamp(0, 255).byte().permute(1, 2, 0).numpy())
    if marked:
        d, c = ImageDraw.Draw(img), SIZE // 2
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.line((c + 40 * dx, c + 40 * dy, c + 70 * dx, c + 70 * dy), fill=(255, 0, 255),
                   width=4)  # fmt: skip
    return img


def _r20_labels() -> dict[int, str]:
    """R20 test candidate index -> blind label (R20's first, then R20b's)."""
    a = json.loads(Path("docs/results/r20_visual_check.json").read_text("utf-8"))
    out = {a["index"][k]: c for c in CLASSES for k in a[c]}
    b = json.loads(Path("docs/results/r20b_visual_check.json").read_text("utf-8"))
    for n, i in enumerate(b["index"]):
        out.setdefault(int(i), b["labels"][str(n)])
    return out


def dev_set() -> dict[str, np.ndarray]:
    crops, meta = rl.load("test", "close")
    labels = _r20_labels()
    idx = np.array(sorted(labels))
    rng = np.random.default_rng(PEOPLE_SEED)
    pool = np.setdiff1d(np.flatnonzero(meta["person"]), idx)
    people = np.sort(rng.choice(pool, N_PEOPLE_DEV, replace=False))
    allx = np.concatenate([idx, people])
    return {"crops": np.asarray(crops[allx]),
            "label": np.array([labels[i] for i in idx] + ["person"] * len(people)),
            "source": np.array(["blind"] * len(idx) + ["visdrone"] * len(people)),
            "frame": meta["frame"][allx], "x": rl.FACTOR * meta["fx"][allx],
            "y": rl.FACTOR * meta["fy"][allx]}  # fmt: skip


def test_set(dev: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    rec = json.loads(r25.LABELS_C.read_text("utf-8"))
    first: dict[tuple[str, int], str] = {}
    for n, (tag, i) in enumerate(rec["spots"]):
        first.setdefault((tag, int(i)), rec["labels"][str(n)])
    rng = np.random.default_rng(PEOPLE_SEED)
    rows: list[tuple[str, int, str, str]] = [(t, i, lab, "blind") for (t, i), lab in first.items()]
    for tag, variant in r25.C_CHAINS:
        c = r25.chain(tag, variant)
        passed = np.flatnonzero(c["keep"] & c["person"])
        rows += [(tag, int(i), "person", "visdrone")
                 for i in np.sort(rng.choice(passed, N_PEOPLE_TEST, replace=False))]  # fmt: skip
    data = {tag: rc.load("test", r25.scout_dir(tag)) for tag, _ in r25.C_CHAINS}
    crops = np.stack([np.asarray(data[t][0][i]) for t, i, _, _ in rows])
    frame = np.array([data[t][1]["frame"][i] for t, i, _, _ in rows])
    x = np.array([data[t][1]["fx"][i] for t, i, _, _ in rows])
    y = np.array([data[t][1]["fy"][i] for t, i, _, _ in rows])
    keep = np.ones(len(rows), bool)
    if dev is not None:
        for k in range(len(rows)):
            m = dev["frame"] == frame[k]
            keep[k] = not np.any(np.hypot(dev["x"][m] - x[k], dev["y"][m] - y[k]) < GAP)
    return {"crops": crops[keep], "label": np.array([r[2] for r in rows])[keep],
            "source": np.array([r[3] for r in rows])[keep], "frame": frame[keep],
            "chain": np.array([r[0] for r in rows])[keep], "dropped": int((~keep).sum())}  # fmt: skip


class VLMCheck:
    """A Qwen3-VL model asked ``PROMPTS[variant]``; R19's yes-minus-no score."""

    def __init__(self, model: str) -> None:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        torch.cuda.set_per_process_memory_fraction(0.85)
        name = MODELS[model]
        self.proc = AutoProcessor.from_pretrained(name, cache_dir=rc.HF_CACHE / "hub")
        self.proc.tokenizer.padding_side = "left"
        kw: dict[str, Any] = {"cache_dir": rc.HF_CACHE / "hub", "dtype": torch.bfloat16}
        if model == "8b":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)  # fmt: skip
            kw["device_map"] = {"": 0}
        self.net = AutoModelForImageTextToText.from_pretrained(name, **kw)
        if model != "8b":
            self.net = self.net.to("cuda")
        self.net.eval()
        tok = self.proc.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]

    def score(self, crops: np.ndarray, variant: str, batch: int = 8,
              log: Any = print) -> np.ndarray:  # fmt: skip
        import torch

        out = []
        px = SIZE * SIZE
        with torch.no_grad():
            for a in range(0, len(crops), batch):
                imgs = [view(c, variant == "marked") for c in crops[a : a + batch]]
                msgs = [[{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": PROMPTS[variant]}]}]
                    for _ in imgs]  # fmt: skip
                text = [self.proc.apply_chat_template(m, tokenize=False,
                                                      add_generation_prompt=True)
                        for m in msgs]  # fmt: skip
                inp = self.proc(text=text, images=imgs, padding=True, return_tensors="pt",
                                size={"shortest_edge": px, "longest_edge": px}).to("cuda")  # fmt: skip
                lg = self.net(**inp, logits_to_keep=1).logits[:, -1].float()
                s = torch.logsumexp(lg[:, self.yes], 1) - torch.logsumexp(lg[:, self.no], 1)
                out.append(s.cpu().numpy())
                if a % (batch * 40) == 0:
                    log(f"[progress] {variant}: {a + len(imgs)}/{len(crops)}")
        return np.concatenate(out)


def score_path(model: str, variant: str, split: str) -> Path:
    return OUT_DIR / f"{model}_{variant}_{split}.npy"


def threshold_at(s: np.ndarray, label: np.ndarray, keep: float = KEEP) -> float:
    """Keeps ``keep`` of the people (``rc.threshold_at``)."""
    return rc.threshold_at(s[label == "person"], keep)


def evaluate(s: np.ndarray, data: dict[str, np.ndarray], thr: float) -> dict[str, Any]:
    lab, src = data["label"], data["source"]
    ok = s >= thr
    person, clear = lab == "person", lab == "not_person"

    def w(mask: np.ndarray, kept: bool) -> list[float]:
        hits = ok[mask] if kept else ~ok[mask]
        return list(rc.wilson(int(hits.sum()), int(mask.sum())))

    return {"threshold": float(thr), "n": {c: int((lab == c).sum()) for c in CLASSES},
            "clear_rejected": w(clear, False), "unclear_rejected": w(lab == "ambiguous", False),
            "people_kept": w(person, True),
            "people_kept_blind": w(person & (src == "blind"), True),
            "people_kept_visdrone": w(person & (src == "visdrone"), True),
            "auc": rc.auc(s[person], s[clear])}  # fmt: skip


def verdict(t: dict[str, Any]) -> str:
    rej = t["clear_rejected"]
    if rej[0] < KILL_REJECT:
        return "KILL"
    groups = min(t["people_kept_blind"][0], t["people_kept_visdrone"][0])
    ok = (rej[0] >= PASS_REJECT and rej[1] >= PASS_REJECT_LO
          and t["people_kept"][0] >= PASS_KEEP and groups >= PASS_KEEP_GROUP)  # fmt: skip
    return "PASS" if ok else "NOT PASSED"


def choose(dev_scores: dict[str, np.ndarray], dev: dict[str, np.ndarray]) -> tuple[str, dict]:
    """The 8B variant that rejects the most dev clear non-persons at its 95%-keep threshold."""
    res = {v: evaluate(s, dev, threshold_at(s, dev["label"])) for v, s in dev_scores.items()}
    return max(res, key=lambda v: res[v]["clear_rejected"][0]), res


def pct(w: list[float]) -> str:
    return f"{100 * w[0]:.1f}% [{100 * w[1]:.0f}, {100 * w[2]:.0f}]"


def render(rep: dict[str, Any]) -> str:
    t = rep["test"]["8b"]
    lines = ["# R27: a ground-station vision-language model as the final check", "",
             f"**{rep['verdict']}.** Qwen3-VL-8B-Instruct (4-bit), variant `{rep['variant']}` "
             f"chosen on dev, at the dev threshold that keeps {KEEP:.0%} of the dev people: "
             f"rejects {pct(t['clear_rejected'])} of the test clear non-persons and keeps "
             f"{pct(t['people_kept'])} of the test people (bars: rejects >= 50% with the lower "
             "bound >= 30%, keeps >= 93% and each group >= 90%; KILL below 25% rejected).", "",
             f"Test: R25c's blind-labelled false alarms of R25b's four chains plus "
             f"{N_PEOPLE_TEST} VisDrone people passed per chain; {rep['test_dropped']} spots "
             f"near a dev spot dropped. Counts: {t['n']}.", "",
             "| Model | Split | Variant | Clear non-persons rejected | Unclear rejected | People "
             "kept (blind-labelled) | People kept (VisDrone) | AUC people vs clear |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for split in ("dev", "test"):
        for model, by in rep[split].items():
            for var, r in (by.items() if split == "dev" else [(rep["variant"], by)]):
                lines.append(f"| {model} | {split} | {var} | {pct(r['clear_rejected'])} | "
                             f"{pct(r['unclear_rejected'])} | {pct(r['people_kept_blind'])} | "
                             f"{pct(r['people_kept_visdrone'])} | {r['auc']:.3f} |")  # fmt: skip
    lines += ["", "Reproduce: `python scripts/r27_vlm_check.py dev`, `test`, `report`.", ""]
    return "\n".join(lines)
