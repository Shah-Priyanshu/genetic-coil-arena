"""Download and verify the Braatz-group Tennessee Eastman data into a cache.

The data lives outside the repo (``tep.default_cache_dir()``) and is never committed.
Every file is checked against a pinned SHA-256 and shape before it is accepted, so a
truncated or altered download fails loudly. The primary source is the Braatz group's
archive; if that is unreachable, the same files are fetched one by one from a public
mirror and held to the same checksums.

``SOURCE.json`` in the cache records the source URL, licence, retrieval date and the
checksums; ``LICENSE.txt`` is the archive's ``readme.txt`` (it carries the licence).
"""

import datetime
import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from .tep import SOURCE_LICENSE, SOURCE_URL, default_cache_dir

ARCHIVE_SHA256 = "3ae9d9578cd0883ed81ee3c2b911f85e57e367016e08bd5f1bde963694e4c111"
ARCHIVE_BYTES = 4925905
MIRROR_URL = (
    "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/"
)
README_SHA256 = "db068740d80b6f4883d9ef1e1e526b41accd9ac7421b1b11bb2a147bb171d7ee"

# file -> (sha256, (rows, columns) as stored); d00.dat is stored transposed
MANIFEST: dict[str, tuple[str, tuple[int, int]]] = {
    "d00.dat": ("4c3c0b11eefacf93e5d2529e2866d7a625e37aa9153d0f95fae17a6c9e560deb", (52, 500)),
    "d00_te.dat": ("57d56da4199e3d73582855d810be1d13d931386fd728bf1694798c7b0a03678c", (960, 52)),
    "d01.dat": ("b0da22ac7561d82b0ba0b48bf87d2b3a8655cb23a3dc37330dd03cdd126c387c", (480, 52)),
    "d01_te.dat": ("e7e6992fb39664e739fccbc001a1c5d5f7cb21829544f734ca3f282c26f680e5", (960, 52)),
    "d02.dat": ("af1a46ba3a5482be3e7b143606f22ae509138544af2a2d0eb7389459257f8179", (480, 52)),
    "d02_te.dat": ("a34bd9c3693fa96bd3ee533fb7fbe29a9d226a4e3250b585ef1308e1136dbfb0", (960, 52)),
    "d03.dat": ("df39c57bdac7ea476ee236c289e0f59b0cbf868adfbbbaf213f692d8cd96e52a", (480, 52)),
    "d03_te.dat": ("b7bf6f2ed7cff0a5577898b009edd252b9bdf6ee9ff7dbeacf9af9aa4c6abace", (960, 52)),
    "d04.dat": ("346702fce33d9707f0b4ec5b8f533ccb08be60313607d5156d7e237b80a87b3f", (480, 52)),
    "d04_te.dat": ("1ccb488dd5feac11aad54623451168abef066229e685872597897800363916f9", (960, 52)),
    "d05.dat": ("3a47bfce4010ce0c634ae828409cce657ecd03bf0c4d59f8dd4a3d4f792f493f", (480, 52)),
    "d05_te.dat": ("1def7258a2865aeead699d52c8cf926fbf34c1467aaf298121987ab858fd0888", (960, 52)),
    "d06.dat": ("ff0882a44bb74f5d2d1a864425ac7e5ae82420391c7060232cefc08ad5bf7331", (480, 52)),
    "d06_te.dat": ("8a3adc7121eaa9b9300c27f9c57a8ab356315401cb964310a4a8244dd34bf79d", (960, 52)),
    "d07.dat": ("e3fd272ed53f899ab0c8278c7eede55418b01a2339b5fcff14a87fefaa3d181d", (480, 52)),
    "d07_te.dat": ("e01b7c81daf9fccc4495cbde5658296918276b3c078044893f6e945dd91d3446", (960, 52)),
    "d08.dat": ("4817735e339449bec909d1cbe77e421acc79c7357b8c6b5bb82c6e5788b72449", (480, 52)),
    "d08_te.dat": ("c0637f02f23b79890b43cfee9604818ad6d7f81532223d76acf1da0d6e2a2cc1", (960, 52)),
    "d09.dat": ("cb9146da760cdd52cad575d745413a5cfb4b447af2db9fb6a424cf6881ff0fcd", (480, 52)),
    "d09_te.dat": ("513df6c3a4fd48e0eaddb0002cd8cdab3a78a5c83c47883bb52d2dc6c2472428", (960, 52)),
    "d10.dat": ("e08697b5d074aea2345434bc4d26296f5dbdbb7efbf37a1bbee292b2ce765e8f", (480, 52)),
    "d10_te.dat": ("3a278be79d8d111fc60be3cfb4f961c28e2dc632f17eb519886aa0f95864ffc6", (960, 52)),
    "d11.dat": ("8294734d6f6e8f6b8643b9dc72f38ce40afd868d30dee9368a0d26b963a0af9f", (480, 52)),
    "d11_te.dat": ("6d60a4c932d918e1ab26ae5d16320936a665f3aad3f05482d23f5c4550b9f29a", (960, 52)),
    "d12.dat": ("62fcb3c71c0eda8eaaaf707f35296e60b17baaf9a9aef588687532cf4c560d38", (480, 52)),
    "d12_te.dat": ("2180fcecb01d1b110b429ba2969d6b18dcc625599e3640e54f69afa32dccbe50", (960, 52)),
    "d13.dat": ("f755d4ab6dc1d30e36c2e4604eeb0baff4b91032f487398b97fb1220eda861dd", (480, 52)),
    "d13_te.dat": ("2e34e0c427d6e41d77d6cae76550b7e95004f9f13bc093278ecf5c282a1c9f03", (960, 52)),
    "d14.dat": ("29fda7d5cae58cb7240e569e2d2d436371d077d282187d9205c7dc01df8755af", (480, 52)),
    "d14_te.dat": ("e8976d056c0769cebbcec3bcba89d2468040ccae7acbea55ebc3b2e56b217396", (960, 52)),
    "d15.dat": ("4c82fe0864cce2521e91a2c3e59d8630627c5e633cffe83f5e34ebc9901b5bca", (480, 52)),
    "d15_te.dat": ("a7242b956837bec8ca7f224d01805247f2aea6ea868eaa064e068b5c0761b1e5", (960, 52)),
    "d16.dat": ("ca1dfc5e31292052e83dcf6b4f60a73da197eb8a47e65d908ed7ac3cbcccb007", (480, 52)),
    "d16_te.dat": ("8381d7371aac49b498e7bf93149b77a0a6bbeffe73520881dcbe8ae96277244d", (960, 52)),
    "d17.dat": ("596aad6421a64d257bfa098cd5c4664d46dc49d88d1bad86cbc9518e6cd69e42", (480, 52)),
    "d17_te.dat": ("27d2d72052064e3cbb7a45df8073f15bb83b77403f532616e73c8869b043e7c9", (960, 52)),
    "d18.dat": ("b4d182c070036104cf150bc33f7aa879805f19943eb298bc05bdc8647dc70291", (480, 52)),
    "d18_te.dat": ("1db5b93a7daa58ed1b641845b58ad818546ced7f3de866c086da18a7ebef5041", (960, 52)),
    "d19.dat": ("ee3d2a50285cb3ba5e52ec05ea7cb39dc2b574bec592af9d44c192be132aeea6", (480, 52)),
    "d19_te.dat": ("c24dd77f787f89347273addc8a453d79f0b99edd64d43be11a5e0907bd515c5f", (960, 52)),
    "d20.dat": ("acdfe6c256fa6de74a811d6c7df1b710a9f0ed60f48242be92f3fa30db3df6b3", (480, 52)),
    "d20_te.dat": ("e54035eda95311f13dd095f87607d97a89b7541afa2a3f68af098f33a5e2e6e5", (960, 52)),
    "d21.dat": ("f34a9fefffef04ef1f49b3fb74e3def0010cda27a8577d814dd50604e2f8dc09", (480, 52)),
    "d21_te.dat": ("b6823aa09638cd59cb168bc24f991b6d4d557d8aea62b0b2a87ee67c21ba5a83", (960, 52)),
}  # fmt: skip


def sha256(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def check_file(name: str, data: bytes) -> None:
    """Verify one data file's checksum and row/column count.

    Raises:
        ValueError: On a checksum or shape mismatch, or an unknown file name
    """
    if name not in MANIFEST:
        raise ValueError(f"Unexpected TEP file {name!r}")
    want_hash, want_shape = MANIFEST[name]
    got = sha256(data)
    if got != want_hash:
        raise ValueError(f"{name}: sha256 {got} != expected {want_hash}")
    rows = [line for line in data.decode("ascii").splitlines() if line.strip()]
    cols = len(rows[0].split())
    if (len(rows), cols) != want_shape:
        raise ValueError(f"{name}: shape {(len(rows), cols)} != expected {want_shape}")


def _download(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "genetic-coil-arena"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _from_archive(blob: bytes) -> tuple[dict[str, bytes], bytes]:
    """Pull the data files and the readme out of the archive (by exact member name)."""
    if len(blob) != ARCHIVE_BYTES or sha256(blob) != ARCHIVE_SHA256:
        raise ValueError(
            f"Archive mismatch: {len(blob)} bytes, sha256 {sha256(blob)} "
            f"(expected {ARCHIVE_BYTES} bytes, {ARCHIVE_SHA256})"
        )
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in MANIFEST:
            files[name] = archive.read(f"TE_process/{name}")
        readme = archive.read("TE_process/readme.txt")
    return files, readme


def _from_mirror(timeout: float) -> tuple[dict[str, bytes], bytes]:
    files = {name: _download(MIRROR_URL + name, timeout) for name in MANIFEST}
    return files, b""


def is_complete(cache_dir: Path) -> bool:
    """True if every data file is present in ``cache_dir`` with the pinned checksum."""
    for name in MANIFEST:
        path = Path(cache_dir) / name
        if not path.exists():
            return False
        try:
            check_file(name, path.read_bytes())
        except (ValueError, UnicodeDecodeError):
            return False
    return True


def fetch(
    cache_dir: Optional[Path] = None, force: bool = False, timeout: float = 60.0
) -> Path:
    """Make sure the verified TEP files are in the cache and return its path.

    Args:
        cache_dir: Target directory (default ``default_cache_dir()``)
        force: Download again even if a complete, verified cache exists
        timeout: Per-request timeout in seconds

    Raises:
        ValueError: If a download fails verification (nothing is written)
        OSError: If neither the archive nor the mirror is reachable
    """
    target = default_cache_dir() if cache_dir is None else Path(cache_dir)
    if not force and is_complete(target):
        return target

    origin = SOURCE_URL
    try:
        files, readme = _from_archive(_download(SOURCE_URL, timeout))
    except (urllib.error.URLError, OSError, ValueError, zipfile.BadZipFile) as primary:
        try:
            files, readme = _from_mirror(timeout)
        except (urllib.error.URLError, OSError) as secondary:
            raise OSError(
                f"Could not fetch the TEP data from {SOURCE_URL} ({primary}) "
                f"or the mirror ({secondary})"
            ) from secondary
        origin = MIRROR_URL
    for name, data in files.items():
        check_file(name, data)
    if readme and sha256(readme) != README_SHA256:
        raise ValueError("readme.txt does not match the pinned checksum")

    target.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (target / name).write_bytes(data)
    if readme:
        (target / "LICENSE.txt").write_bytes(readme)
    record = {
        "dataset": "Tennessee Eastman Process, Braatz group benchmark (d00-d21)",
        "url": SOURCE_URL,
        "fetched_from": origin,
        "mirror": MIRROR_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "retrieved": datetime.date.today().isoformat(),
        "license": SOURCE_LICENSE,
        "files": {
            name: {"sha256": digest, "shape": list(shape)}
            for name, (digest, shape) in MANIFEST.items()
        },
    }
    (target / "SOURCE.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return target
