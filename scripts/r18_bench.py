"""R18: the desk bench for the OpenSwarm brains (three ESP32-S3 boards on USB).

Drives ``hardware/esp32/bench`` firmware over serial and appends every measurement to
``D:/Projects/genetic-coil-arena-data/r18_bench/log.jsonl``; ``report`` summarises the
log into ``docs/results/r18_bench.{md,json}``.

python scripts/r18_bench.py ports                         # which COM port is which board
python scripts/r18_bench.py flash --ports COM3 COM4 COM5  # build once, upload to each
python scripts/r18_bench.py radio --ports COM3 COM4 COM5  # brain check and timing, ESP-NOW pings
python scripts/r18_bench.py ftm --ap COM3 --port COM4 --distance-m 2.0
python scripts/r18_bench.py gas --port COM3               # SGP40: baseline, then alcohol nearby
python scripts/r18_bench.py report
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

LOG = Path(r"D:\Projects\genetic-coil-arena-data\r18_bench\log.jsonl")
FIRMWARE = Path("hardware/esp32/bench")
PIO = Path(r"D:\venvs\pio\Scripts\pio.exe")
PINGS = ((500, 32, 20), (500, 200, 20), (500, 32, 5))  # count, bytes, interval ms


def board(port: str):
    import serial

    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, 115200, 0.5
    s.dtr = s.rts = False  # opening must not hold the board in reset
    s.open()
    time.sleep(1.5)
    s.reset_input_buffer()
    return s


def command(s, line: str, timeout: float = 120.0) -> list[list[str]]:
    """Send one command; the ``#TAG,...`` lines it printed, up to ``#DONE``."""
    s.write((line + "\n").encode())
    out, end = [], time.time() + timeout
    while time.time() < end:
        raw = s.readline().decode(errors="replace").strip()
        if raw == "#DONE":
            return out
        if raw.startswith("#"):
            out.append(raw[1:].split(","))
    raise TimeoutError(f"{s.port}: no #DONE after {line!r}")


def log(kind: str, **rec) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def cmd_ports(_args) -> None:
    from serial.tools import list_ports

    for p in list_ports.comports():
        print(p.device, p.description, p.hwid)


def cmd_flash(args) -> None:
    env = {**os.environ, "PLATFORMIO_CORE_DIR": r"D:\platformio"}
    for port in args.ports:
        subprocess.run([str(PIO), "run", "-t", "upload", "--upload-port", port],
                       cwd=FIRMWARE, env=env, check=True)  # fmt: skip


def cmd_radio(args) -> None:
    boards = {p: board(p) for p in args.ports}
    macs = {}
    for p, s in boards.items():
        info = command(s, "info")
        macs[p] = info[0][1]
        command(s, "hello")
    time.sleep(0.5)
    for p, s in boards.items():
        rows = command(s, "brain 20000")
        golden = next(r for r in rows if r[0] == "GOLDEN")
        timing = {r[1]: float(r[3]) for r in rows if r[0] == "BRAIN"}
        log("brain", port=p, mac=macs[p], golden_n=int(golden[1]), gas_ok=int(golden[2]),
            cam_ok=int(golden[3]), worst=int(golden[4]), us_per_call=timing)  # fmt: skip
        print(p, "golden", golden[1:], "us/call", timing)
    for p, s in boards.items():
        for count, size, ms in PINGS:
            rows = command(
                s, f"ping {count} {size} {ms}", timeout=count * ms / 1000 + 60
            )
            for peer in {r[2] for r in rows if r[0] == "GOT"}:
                rtt = [int(r[4]) for r in rows if r[0] == "RTT" and r[2] == peer]
                got = next(int(r[3]) for r in rows if r[0] == "GOT" and r[2] == peer)
                log("ping", sender=macs[p], peer=peer, count=count, bytes=size,
                    interval_ms=ms, replies=len(rtt), one_way=got, rtt_us=rtt,
                    distance_m=args.distance_m)  # fmt: skip
                print(f"{macs[p]} -> {peer}: {count} x {size} B every {ms} ms, one way "
                      f"{got}, back {len(rtt)}, median {np.median(rtt) if rtt else 0:.0f} us")  # fmt: skip


def cmd_ftm(args) -> None:
    ap, s = board(args.ap), board(args.port)
    command(ap, "ap")
    time.sleep(2)
    rows = command(s, f"ftm {args.count}", timeout=args.count * 4 + 30)
    cm = [int(r[3]) for r in rows if r[0] == "FTM" and r[2] == "ok"]
    log("ftm", ap=args.ap, port=args.port, distance_m=args.distance_m, count=args.count,
        ok=len(cm), dist_cm=cm, status=[r[2] for r in rows if r[0] == "FTM"])  # fmt: skip
    print(f"{len(cm)}/{args.count} ok; median {np.median(cm) if cm else 0:.0f} cm "
          f"at a measured {100 * args.distance_m:.0f} cm")  # fmt: skip


def cmd_gas(args) -> None:
    s = board(args.port)
    print(f"Baseline: keep alcohol away for {args.baseline} s.")
    base = command(s, f"gas {args.baseline} 1000", timeout=args.baseline * 2 + 30)
    input(
        "Now put a pad with a few drops of rubbing alcohol 10 cm from the sensor, then press Enter."
    )
    near = command(s, f"gas {args.exposure} 1000", timeout=args.exposure * 2 + 30)
    for name, rows in (("baseline", base), ("near", near)):
        log("gas", port=args.port, phase=name, raw=[int(r[3]) for r in rows if r[0] == "GAS"],
            voc=[int(r[4]) for r in rows if r[0] == "GAS"])  # fmt: skip
    print("logged; remove the pad")


def cmd_report(args) -> None:
    recs = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines()]
    rep: dict = {"brain": [], "ping": [], "ftm": [], "gas": []}
    for r in recs:
        if r["kind"] == "brain":
            rep["brain"].append({k: r[k] for k in ("mac", "golden_n", "gas_ok", "cam_ok",
                                                   "us_per_call")})  # fmt: skip
        elif r["kind"] == "ping":
            rtt = np.array(r["rtt_us"] or [0])
            rep["ping"].append({"link": f"{r['sender'][-5:]} -> {r['peer'][-5:]}",
                                "bytes": r["bytes"], "interval_ms": r["interval_ms"],
                                "one_way": r["one_way"] / r["count"],
                                "round_trip": r["replies"] / r["count"],
                                "median_us": float(np.median(rtt)),
                                "p95_us": float(np.percentile(rtt, 95))})  # fmt: skip
        elif r["kind"] == "ftm" and r["dist_cm"]:
            d = np.array(r["dist_cm"]) / 100
            rep["ftm"].append({"distance_m": r["distance_m"], "ok": r["ok"] / r["count"],
                               "median_m": float(np.median(d)), "sd_m": float(d.std())})  # fmt: skip
        elif r["kind"] == "gas":
            rep["gas"].append({"phase": r["phase"], "voc_mean": float(np.mean(r["voc"])),
                               "voc_max": int(max(r["voc"])),
                               "raw_mean": float(np.mean(r["raw"]))})  # fmt: skip
    lines = ["# R18: the desk bench (ESP32-S3)", "", "## Brains on the chip", "",
             "| Board | Golden (gas / cam) | µs per call (gas, cam) |", "|---|---|---|"]  # fmt: skip
    for b in rep["brain"]:
        lines.append(f"| {b['mac'][-5:]} | {b['gas_ok']}/{b['golden_n']} / "
                     f"{b['cam_ok']}/{b['golden_n']} | {b['us_per_call'].get('gas', 0):.1f}, "
                     f"{b['us_per_call'].get('cam', 0):.1f} |")  # fmt: skip
    lines += ["", "## ESP-NOW broadcast (as OpenSwarm sends)", "",
              "| Link | Bytes | Every | Delivered one way | Round trip | Median RTT | 95th |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for p in rep["ping"]:
        lines.append(f"| {p['link']} | {p['bytes']} | {p['interval_ms']} ms | "
                     f"{100 * p['one_way']:.1f}% | {100 * p['round_trip']:.1f}% | "
                     f"{p['median_us'] / 1000:.2f} ms | {p['p95_us'] / 1000:.2f} ms |")  # fmt: skip
    lines += ["", "## Wi-Fi FTM ranging", "", "| Tape | Success | Median estimate | SD |",
              "|---|---|---|---|"]  # fmt: skip
    for f in rep["ftm"]:
        lines.append(f"| {f['distance_m']:.2f} m | {100 * f['ok']:.0f}% | "
                     f"{f['median_m']:.2f} m | {f['sd_m']:.2f} m |")  # fmt: skip
    lines += ["", "## SGP40 gas sensor", "", "| Phase | VOC index mean | max | raw mean |",
              "|---|---|---|---|"]  # fmt: skip
    for g in rep["gas"]:
        lines.append(f"| {g['phase']} | {g['voc_mean']:.0f} | {g['voc_max']} | "
                     f"{g['raw_mean']:.0f} |")  # fmt: skip
    md = "\n".join(lines) + "\n"
    out = Path("docs/results/r18_bench.md")
    out.write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(md)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ports")
    f = sub.add_parser("flash")
    f.add_argument("--ports", nargs="+", required=True)
    r = sub.add_parser("radio")
    r.add_argument("--ports", nargs="+", required=True)
    r.add_argument(
        "--distance-m", type=float, default=1.0, help="spacing of the boards"
    )
    t = sub.add_parser("ftm")
    t.add_argument("--ap", required=True)
    t.add_argument("--port", required=True)
    t.add_argument("--distance-m", type=float, required=True, help="tape-measured")
    t.add_argument("--count", type=int, default=50)
    g = sub.add_parser("gas")
    g.add_argument("--port", required=True)
    g.add_argument("--baseline", type=int, default=120)
    g.add_argument("--exposure", type=int, default=60)
    sub.add_parser("report")
    args = p.parse_args()
    {"ports": cmd_ports, "flash": cmd_flash, "radio": cmd_radio, "ftm": cmd_ftm,
     "gas": cmd_gas, "report": cmd_report}[args.cmd](args)  # fmt: skip


if __name__ == "__main__":
    main()
