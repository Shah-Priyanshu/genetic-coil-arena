"""R9 diagnostic: what can a linear probe read from the extractor's hidden states?

Run after ``r9_latent_vs_json.py``: loads the seed-0 extractor from the run directory
and fits one linear probe per field property on A's layer-14 states, taken from the last
position, the last 16 positions, or the mean over all positions. It explains why the
last-k message fails (currency and vendor are readable, digits are not) and prints a
Markdown table.

    python scripts/r9_state_probe.py --run-dir outputs/r9_latent_vs_json
"""

import argparse
import os
from pathlib import Path

import torch

from genetic_coil_arena.backends import HFPeftBackend
from genetic_coil_arena.experiments.latent_vs_json import A_PROMPT
from genetic_coil_arena.experiments.records import TRAIN_VENDORS, make_records

CURRENCIES = ("USD", "EUR", "GBP", "CAD", "AUD")


def features(be, records, layer, mode):
    out = []
    for i in range(0, len(records), 100):
        batch = records[i : i + 100]
        hs = be.hidden_states([A_PROMPT.format(text=r.text) for r in batch], layer, "A")
        x, m = hs.states.float(), hs.mask.float()
        if mode == "last1":
            f = x[:, -1]
        elif mode == "last16":
            f = x[:, -16:].flatten(1)
        else:
            f = (x * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        out.append(f.cpu())
    return torch.cat(out)


def targets(records):
    gold = [r.gold for r in records]
    return {
        "currency (5)": [CURRENCIES.index(g["currency_iso"]) for g in gold],
        "vendor (48)": [TRAIN_VENDORS.index(g["vendor_canonical"]) for g in gold],
        "year (9)": [int(g["date_iso"][:4]) - 2019 for g in gold],
        "month (12)": [int(g["date_iso"][5:7]) - 1 for g in gold],
        "amount digits (7)": [len(str(g["amount_cents"])) - 3 for g in gold],
        "amount first digit (9)": [int(str(g["amount_cents"])[0]) - 1 for g in gold],
    }


def probe(xtr, ytr, xte, yte, classes, device):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-3
    xtr, xte = ((xtr - mu) / sd).to(device), ((xte - mu) / sd).to(device)
    ytr, yte = torch.tensor(ytr, device=device), torch.tensor(yte, device=device)
    torch.manual_seed(0)
    head = torch.nn.Linear(xtr.shape[1], classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(300):
        loss = torch.nn.functional.cross_entropy(head(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (head(xte).argmax(-1) == yte).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=os.environ.get("GCA_RUN_DIR"), required=False)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--n-train", type=int, default=4000)
    args = parser.parse_args()
    be = HFPeftBackend(local_files_only=True).load()
    be.load_adapter("A", Path(args.run_dir) / "ckpt" / "A_s0.pt")
    train, test = make_records(args.n_train, "train", 123), make_records(600, "id_test", 7)
    ytr, yte = targets(train), targets(test)
    classes = {"currency (5)": 5, "vendor (48)": 48, "year (9)": 9, "month (12)": 12,
               "amount digits (7)": 7, "amount first digit (9)": 9}
    rows = {name: [] for name in ytr}
    modes = ("last1", "last16", "mean")
    for mode in modes:
        xtr, xte = features(be, train, args.layer, mode), features(be, test, args.layer, mode)
        for name in ytr:
            rows[name].append(probe(xtr, ytr[name], xte, yte[name], classes[name], be.device))
    chance = {n: max(torch.bincount(torch.tensor(yte[n])).tolist()) / len(test) for n in yte}
    print(f"| property | majority | {' | '.join(modes)} |")
    print("|---|---|" + "---|" * len(modes))
    for name, accs in rows.items():
        print(f"| {name} | {chance[name] * 100:.0f} | " + " | ".join(f"{a * 100:.0f}" for a in accs) + " |")


if __name__ == "__main__":
    main()
