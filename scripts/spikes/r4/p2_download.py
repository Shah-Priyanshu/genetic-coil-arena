import os, time, json, sys
from pathlib import Path
HERE = Path(__file__).parent
os.environ["HF_HOME"] = str(HERE / "hf")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from huggingface_hub import snapshot_download
repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
t = time.time()
p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
dt = time.time() - t
tot = sum(f.stat().st_size for f in Path(os.environ["HF_HOME"]).rglob("*") if f.is_file())
out = {"repo": repo, "snapshot_path": p, "seconds": dt, "cache_bytes": tot, "cache_gb": tot / 1e9, "hf_home": os.environ["HF_HOME"]}
print(json.dumps(out, indent=2))
(HERE / "results" / "p2_download.json").write_text(json.dumps(out, indent=2))
