"""Tests for the Rieth et al. data layer: RData reader, conversion, loader, run-level splits."""

import bz2
import gc
import gzip
import itertools
import json
import lzma
import struct
from pathlib import Path

import numpy as np
import pytest

from genetic_coil_arena.envs.plant import fetch_rieth as fetch_mod
from genetic_coil_arena.envs.plant import rieth
from genetic_coil_arena.envs.plant.rdata import (
    RList,
    data_frame_columns,
    read_rdata,
)
from genetic_coil_arena.envs.plant.rieth import (
    ONSET_TEST,
    ONSET_TRAIN,
    RiethData,
    RunSplits,
    is_converted,
    rieth_cache_dir,
    synthetic_rieth,
    to_runs,
)
from genetic_coil_arena.envs.plant.tep import Normaliser, default_cache_dir
from genetic_coil_arena.envs.plant.windows import NO_ONSET

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------- a tiny R serialization writer
class RWriter:
    """Writes the XDR serialization subset that ``rdata`` reads (symbols are shared)."""

    def __init__(self):
        self.symbols: dict[str, int] = {}

    @staticmethod
    def i(v: int) -> bytes:
        return struct.pack(">i", v)

    def charsxp(self, text: str) -> bytes:
        raw = text.encode("utf-8")
        return self.i(9 | (64 << 12)) + self.i(len(raw)) + raw

    def sym(self, name: str) -> bytes:
        if name in self.symbols:
            return self.i((self.symbols[name] << 8) | 255)  # REFSXP
        self.symbols[name] = len(self.symbols) + 1
        return self.i(1) + self.charsxp(name)

    def strsxp(self, items: list[str]) -> bytes:
        return (
            self.i(16) + self.i(len(items)) + b"".join(self.charsxp(t) for t in items)
        )

    def real(self, arr) -> bytes:
        a = np.asarray(arr, dtype=">f8")
        return self.i(14) + self.i(a.size) + a.tobytes()

    def integer(self, arr) -> bytes:
        a = np.asarray(arr, dtype=">i4")
        return self.i(13) + self.i(a.size) + a.tobytes()

    def pairlist(self, items: list[tuple[str, bytes]]) -> bytes:
        out = b"".join(self.i(2 | 0x400) + self.sym(n) + v for n, v in items)
        return out + self.i(254)

    def frame(self, columns: dict[str, bytes], n_rows: int) -> bytes:
        attrs = self.pairlist(
            [
                ("names", self.strsxp(list(columns))),
                ("class", self.strsxp(["data.frame"])),
                ("row.names", self.integer([-(2**31), -n_rows])),
            ]
        )
        return (
            self.i(19 | 0x200)
            + self.i(len(columns))
            + b"".join(columns.values())
            + attrs
        )

    def file(self, objects: dict, version: int = 2) -> bytes:
        """``objects`` maps a name to a builder, called after the name's symbol is written
        so that symbol indices follow the order of the stream."""
        header = b"X\n" + self.i(version) + self.i(0x00030300) + self.i(0x00020300)
        if version == 3:
            header += self.i(5) + b"UTF-8"
        body = b"".join(
            self.i(2 | 0x400) + self.sym(name) + build()
            for name, build in objects.items()
        )
        return b"RDX%d\n" % version + header + body + self.i(254)


def write_frames(path: Path, frames: dict[str, dict[str, np.ndarray]], **kw) -> None:
    w = RWriter()
    objects = {}
    for name, cols in frames.items():
        n = len(next(iter(cols.values())))
        enc = {
            c: (w.integer(v) if np.asarray(v).dtype.kind == "i" else w.real(v))
            for c, v in cols.items()
        }
        objects[name] = lambda enc=enc, n=n: w.frame(enc, n)
    blob = w.file(objects, **kw)
    path.write_bytes(gzip.compress(blob))


class TestRData:
    def test_reads_data_frames_with_shared_symbols(self, tmp_path):
        a = {"x": np.array([1.5, 2.5, 3.5]), "n": np.array([1, 2, 3])}
        b = {"x": np.array([9.0, 8.0]), "n": np.array([7, 6])}
        write_frames(tmp_path / "f.RData", {"first": a, "second": b})
        out = read_rdata(
            tmp_path / "f.RData"
        )  # the second frame refers to symbols by index
        assert list(out) == ["first", "second"]
        cols = data_frame_columns(out["first"])
        assert list(cols) == ["x", "n"] and cols["x"].dtype == np.float64
        assert np.array_equal(cols["x"], a["x"]) and np.array_equal(cols["n"], a["n"])
        assert np.array_equal(data_frame_columns(out["second"])["x"], b["x"])

    def test_version_3_header_and_other_compressions(self, tmp_path):
        frames = {"d": {"x": np.arange(4.0)}}
        write_frames(tmp_path / "v3.RData", frames, version=3)
        assert np.array_equal(
            data_frame_columns(read_rdata(tmp_path / "v3.RData")["d"])["x"],
            np.arange(4.0),
        )
        blob = gzip.decompress((tmp_path / "v3.RData").read_bytes())
        for name, packed in (
            ("bz", bz2.compress(blob)),
            ("xz", lzma.compress(blob)),
            ("raw", blob),
        ):
            (tmp_path / f"{name}.RData").write_bytes(packed)
            got = data_frame_columns(read_rdata(tmp_path / f"{name}.RData")["d"])
            assert np.array_equal(got["x"], np.arange(4.0)), name

    def test_doubles_can_be_narrowed_on_the_way_in(self, tmp_path):
        write_frames(
            tmp_path / "f.RData",
            {"d": {"x": np.array([0.1, 2.0]), "n": np.array([4, 5])}},
        )
        cols = data_frame_columns(read_rdata(tmp_path / "f.RData", np.float32)["d"])
        assert cols["x"].dtype == np.float32 and cols["n"].dtype == np.int32
        assert cols["x"][0] == np.float32(0.1)

    def test_unsupported_content_is_refused_not_guessed(self, tmp_path):
        w = RWriter()
        (tmp_path / "env.RData").write_bytes(
            gzip.compress(w.file({"e": lambda: w.i(4)}))
        )
        with pytest.raises(ValueError, match="Unsupported R object type"):
            read_rdata(tmp_path / "env.RData")
        (tmp_path / "junk.RData").write_bytes(gzip.compress(b"not an R file"))
        with pytest.raises(ValueError, match="not an XDR RData"):
            read_rdata(tmp_path / "junk.RData")
        (tmp_path / "trunc.RData").write_bytes(
            gzip.compress(w.file({"d": lambda: w.real([1.0, 2.0])})[:-9])
        )
        with pytest.raises(ValueError, match="end of the R serialization"):
            read_rdata(tmp_path / "trunc.RData")

    def test_only_data_frames_of_equal_columns(self):
        with pytest.raises(ValueError, match="data frame"):
            data_frame_columns(RList([np.zeros(2)], {"names": ["x"]}))
        frame = RList(
            [np.zeros(2), np.zeros(3)], {"names": ["x", "y"], "class": ["data.frame"]}
        )
        with pytest.raises(ValueError, match="different lengths"):
            data_frame_columns(frame)


# --------------------------------------------------------------------------- conversion
def tiny_frame(order: str, n_faults=2, n_runs=3, n_samples=4, faulty=True):
    """A Rieth-shaped frame whose value encodes (fault, run, sample, column)."""
    faults = list(range(1, n_faults + 1)) if faulty else [0]
    grid = {
        "fault_run_sample": itertools.product(
            faults, range(1, n_runs + 1), range(1, n_samples + 1)
        ),
        # The published files are ordered run, fault, sample
        "run_fault_sample": (
            (f, r, s)
            for r in range(1, n_runs + 1)
            for f in faults
            for s in range(1, n_samples + 1)
        ),
    }[order]
    rows = list(grid)
    cols: dict[str, np.ndarray] = {
        "faultNumber": np.array([r[0] for r in rows], dtype=float),
        "simulationRun": np.array([r[1] for r in rows], dtype=float),
        "sample": np.array([r[2] for r in rows]),
    }
    for j, name in enumerate(fetch_mod.COLUMNS[3:]):
        cols[name] = np.array([r[0] * 1000 + r[1] * 10 + r[2] + j / 100 for r in rows])
    return cols


@pytest.fixture
def tiny_spec(monkeypatch):
    monkeypatch.setattr(fetch_mod, "N_RUNS", 3)
    monkeypatch.setattr(fetch_mod, "RIETH_FAULTS", (1, 2))
    monkeypatch.setattr(
        fetch_mod,
        "CACHE_FILES",
        {
            "faulty_training": ("a.npy", (2, 3, 4, 52)),
            "fault_free_training": ("b.npy", (3, 4, 52)),
        },
    )


class TestConversion:
    @pytest.mark.parametrize("order", ["run_fault_sample", "fault_run_sample"])
    def test_rows_land_in_fault_run_sample_order(self, tmp_path, tiny_spec, order):
        raw = fetch_mod.RawFile("faulty_training", "f.RData", 1, "", 0, "", "", True, 4)
        write_frames(tmp_path / "f.RData", {"faulty_training": tiny_frame(order)})
        dense = fetch_mod._dense(raw, tmp_path / "f.RData")
        assert dense.shape == (2, 3, 4, 52) and dense.dtype == np.float32
        for f, r, s, j in [(1, 1, 1, 0), (2, 3, 4, 51), (1, 3, 2, 7), (2, 1, 3, 20)]:
            want = f * 1000 + r * 10 + s + j / 100
            assert dense[f - 1, r - 1, s - 1, j] == pytest.approx(want, rel=1e-6)

    def test_fault_free_file_has_no_fault_axis(self, tmp_path, tiny_spec):
        raw = fetch_mod.RawFile(
            "fault_free_training", "f.RData", 1, "", 0, "", "", False, 4
        )
        write_frames(
            tmp_path / "f.RData",
            {"fault_free_training": tiny_frame("run_fault_sample", faulty=False)},
        )
        assert fetch_mod._dense(raw, tmp_path / "f.RData").shape == (3, 4, 52)

    def test_incomplete_or_misnamed_files_are_refused(self, tmp_path, tiny_spec):
        raw = fetch_mod.RawFile("faulty_training", "f.RData", 1, "", 0, "", "", True, 4)
        cols = tiny_frame("run_fault_sample")
        short = {k: v[:-1] for k, v in cols.items()}
        write_frames(tmp_path / "short.RData", {"faulty_training": short})
        with pytest.raises(ValueError, match="complete run grid"):
            fetch_mod._dense(raw, tmp_path / "short.RData")
        write_frames(tmp_path / "name.RData", {"other": cols})
        with pytest.raises(ValueError, match="expected one object"):
            fetch_mod._dense(raw, tmp_path / "name.RData")
        renamed = {("zzz" if k == "xmv_11" else k): v for k, v in cols.items()}
        write_frames(tmp_path / "col.RData", {"faulty_training": renamed})
        with pytest.raises(ValueError, match="unexpected columns"):
            fetch_mod._dense(raw, tmp_path / "col.RData")

    def test_pinned_downloads_are_complete(self):
        assert set(fetch_mod.RAW_FILES) == set(rieth.CACHE_FILES)
        for raw in fetch_mod.RAW_FILES.values():
            assert len(raw.md5) == 32 and len(raw.sha256) == 64 and raw.size > 1e6
            assert raw.name.endswith(".RData") and raw.doi.startswith(
                "doi:10.7910/DVN/6C3JR1/"
            )
        assert "public domain" in fetch_mod.LICENSE.lower()
        assert "6C3JR1" in fetch_mod.CITATION

    def test_check_raw_rejects_altered_files(self, tmp_path):
        raw = fetch_mod.RawFile("x", "x.RData", 1, "", 4, "0" * 32, "0" * 64, False, 4)
        path = tmp_path / "x.RData"
        path.write_bytes(b"abcd")
        with pytest.raises(ValueError, match="MD5"):
            fetch_mod.check_raw(raw, path)
        path.write_bytes(b"abc")
        with pytest.raises(ValueError, match="bytes"):
            fetch_mod.check_raw(raw, path)

    def test_onsets_are_measured_from_the_data(self):
        data = synthetic_rieth(n_runs=40, n_faults=7, seed=1, strength=6.0)
        arrays = {name: getattr(data, name) for name in rieth.CACHE_FILES}
        assert fetch_mod.verify_onsets(arrays) == {
            "training": ONSET_TRAIN,
            "testing": ONSET_TEST,
        }
        # A fault that starts one sample late disagrees with the others
        late = np.array(data.faulty_testing)
        late[3, :, ONSET_TEST : ONSET_TEST + 1, :] = data.fault_free_testing[
            :, :1, :
        ].mean(axis=0)
        arrays["faulty_testing"] = late
        with pytest.raises(ValueError, match="disagree|clean onset"):
            fetch_mod.verify_onsets(arrays)

    def test_fetch_script_is_a_thin_cli(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "fetch_rieth_script", REPO / "scripts" / "fetch_rieth.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)


# ------------------------------------------------------------------------------ loader
class TestLoader:
    def test_cache_dir_is_outside_the_repo_and_overridable(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GCA_TEP_DIR", raising=False)
        default = rieth_cache_dir().resolve()
        assert REPO not in default.parents and default.name == "rieth2017"
        monkeypatch.setenv("GCA_TEP_DIR", str(tmp_path / "x"))
        assert rieth_cache_dir() == tmp_path / "x" / "rieth2017"
        assert rieth_cache_dir(tmp_path / "y") == tmp_path / "y" / "rieth2017"

    def test_missing_cache_points_to_the_fetch_script(self, tmp_path):
        assert not is_converted(tmp_path)
        with pytest.raises(FileNotFoundError, match="fetch_rieth"):
            RiethData.from_directory(tmp_path)

    def test_converted_cache_round_trips_through_npy(self, tmp_path, monkeypatch):
        tiny = synthetic_rieth(n_runs=3, n_faults=2, seed=2)
        shapes = {
            k: (name, tuple(getattr(tiny, k).shape))
            for k, (name, _) in rieth.CACHE_FILES.items()
        }
        monkeypatch.setattr(rieth, "CACHE_FILES", shapes)
        assert not is_converted(tmp_path)
        for key, (name, _) in shapes.items():
            np.save(tmp_path / name, getattr(tiny, key))
        (tmp_path / rieth.SOURCE_FILE).write_text(json.dumps({"doi": rieth.DOI}))
        assert is_converted(tmp_path)
        loaded = RiethData.from_directory(tmp_path)  # memory-mapped
        assert isinstance(loaded.faulty_testing, np.memmap)
        assert loaded.source["doi"] == rieth.DOI and loaded.faults == (1, 2)
        assert np.array_equal(
            loaded.fault_runs("testing", 2, np.array([2, 0])),
            tiny.faulty_testing[1, [2, 0]],
        )
        assert np.array_equal(
            loaded.normal_runs("training", np.array([1])), tiny.fault_free_training[[1]]
        )
        del loaded
        gc.collect()  # release the memory maps before touching the files (Windows)
        bad = tmp_path / "bad"
        bad.mkdir()
        for path in tmp_path.iterdir():
            if path.is_file():
                (bad / path.name).write_bytes(path.read_bytes())
        (bad / shapes["fault_free_testing"][0]).write_bytes(b"truncated")
        assert not is_converted(bad)

    def test_bad_shapes_kinds_and_faults_are_rejected(self):
        good = synthetic_rieth(n_runs=2, n_faults=2)
        with pytest.raises(ValueError, match="shaped like"):
            RiethData(
                good.fault_free_training[:, :100],
                good.fault_free_testing,
                good.faulty_training,
                good.faulty_testing,
            )
        with pytest.raises(ValueError, match="kind"):
            good.normal_runs("validation", np.array([0]))
        with pytest.raises(KeyError, match="1..20"):
            good.fault_runs("testing", 5, np.array([0]))
        assert good.n_runs == 2

    def test_synthetic_data_has_the_real_shapes_and_onsets(self):
        data = synthetic_rieth(n_runs=4, n_faults=3, seed=0, strength=8.0)
        assert data.fault_free_training.shape == (4, 500, 52)
        assert data.faulty_testing.shape == (3, 4, 960, 52)
        free = data.fault_free_testing.reshape(-1, 52)
        z = (
            np.abs((data.faulty_testing[0] - free.mean(0)) / free.std(0))
            .max(axis=2)
            .mean(0)
        )
        assert z[:ONSET_TEST].max() < 4.5 < z[ONSET_TEST:].min()


# ------------------------------------------------------------------------------ splits
SPLIT_KW = dict(  # noqa: C408
    faults=(1, 2, 4),
    n_replicates=3,
    train_runs=4,
    val_runs=3,
    test_runs=5,
    train_normal_runs=6,
    val_normal_runs=4,
    test_normal_runs=5,
    n_available=40,
)


class TestRunSplits:
    @pytest.fixture(scope="class")
    def splits(self):
        return RunSplits(**SPLIT_KW)

    def test_sizes_per_fault_and_per_part(self, splits):
        sel = splits.selection(1)
        assert sel.train_normal.shape == (6,) and sel.val_normal.shape == (4,)
        assert sel.test_normal.shape == (5,)
        for f in (1, 2, 4):
            assert sel.train_fault[f].shape == (4,)
            assert sel.val_fault[f].shape == (3,) and sel.test_fault[f].shape == (5,)
            assert sel.train_fault[f].max() < 40 and sel.train_fault[f].min() >= 0

    def test_train_val_and_test_runs_are_disjoint_simulations(self, splits):
        for rep in [None, 0, 1, 2]:
            sel = splits.selection(rep)
            train, val, test = (sel.keys(p) for p in ("train", "val", "test"))
            assert (
                train.isdisjoint(val)
                and train.isdisjoint(test)
                and val.isdisjoint(test)
            )
            # No duplicate run inside a part either
            assert (
                len(train) == 6 + 3 * 4
                and len(val) == 4 + 3 * 3
                and len(test) == 5 + 3 * 5
            )

    def test_replicates_and_the_tuning_stage_never_share_a_run(self, splits):
        keys = {
            rep: [splits.selection(rep).keys(p) for p in ("train", "val", "test")]
            for rep in [None, 0, 1, 2]
        }
        for a, b in itertools.combinations(keys, 2):
            for ka, kb in itertools.product(keys[a], keys[b]):
                assert ka.isdisjoint(kb), (a, b)

    def test_same_fault_runs_are_the_same_across_fault_sets(self):
        a = RunSplits(**{**SPLIT_KW, "faults": (1, 2)}).selection(0)
        b = RunSplits(**SPLIT_KW).selection(0)
        assert np.array_equal(a.test_fault[2], b.test_fault[2])
        assert np.array_equal(a.test_normal, b.test_normal)

    def test_deterministic_and_seed_dependent(self, splits):
        again = RunSplits(**SPLIT_KW).selection(2)
        assert np.array_equal(again.test_fault[4], splits.selection(2).test_fault[4])
        other = RunSplits(**{**SPLIT_KW, "split_seed": 1}).selection(2)
        assert not np.array_equal(other.test_fault[4], again.test_fault[4])

    def test_too_many_blocks_or_bad_arguments_raise(self):
        with pytest.raises(ValueError, match="blocks need"):
            RunSplits(**{**SPLIT_KW, "n_replicates": 9})
        with pytest.raises(ValueError, match="faults"):
            RunSplits(**{**SPLIT_KW, "faults": (21,)})
        with pytest.raises(ValueError, match="at least one run"):
            RunSplits(**{**SPLIT_KW, "val_runs": 0})
        with pytest.raises(ValueError, match="replicate"):
            RunSplits(**SPLIT_KW).selection(3)

    def test_to_runs_normalises_with_the_given_statistics_and_sets_onsets(self):
        data = synthetic_rieth(n_runs=12, n_faults=3, seed=3)
        splits = RunSplits(
            faults=(1, 2, 3),
            n_replicates=1,
            train_runs=3,
            val_runs=2,
            test_runs=2,
            train_normal_runs=5,
            val_normal_runs=2,
            test_normal_runs=2,
            n_available=12,
        )
        sel = splits.selection(0)
        norm = Normaliser.fit(
            data.normal_runs("training", np.sort(sel.train_normal)).reshape(-1, 52)
        )
        train = to_runs(data, norm, sel, "train", (1, 3))
        test = to_runs(data, norm, sel, "test", (1, 3))
        assert [r.fault for r in train] == [0] * 5 + [1] * 3 + [3] * 3
        assert [r.onset for r in train][:6] == [NO_ONSET] * 5 + [ONSET_TRAIN]
        assert {r.onset for r in test if r.fault} == {ONSET_TEST}
        assert train[0].series.shape == (500, 52) and test[0].series.shape == (960, 52)
        assert train[0].series.dtype == np.float32
        pooled = np.concatenate([r.series for r in train[:5]])
        assert np.allclose(pooled.mean(0), 0.0, atol=1e-5)
        with pytest.raises(ValueError, match="part"):
            to_runs(data, norm, sel, "dev", (1,))


# ----------------------------------------------------------------- real data (optional)
needs_rieth = pytest.mark.skipif(
    not is_converted(rieth_cache_dir()),
    reason="Rieth et al. data not fetched (python scripts/fetch_rieth.py)",
)


@needs_rieth
class TestRealRiethData:
    @pytest.fixture(scope="class")
    def real(self):
        return RiethData.from_cache()

    def test_shapes_provenance_and_licence(self, real):
        assert real.fault_free_training.shape == (500, 500, 52)
        assert real.fault_free_testing.shape == (500, 960, 52)
        assert real.faulty_training.shape == (20, 500, 500, 52)
        assert real.faulty_testing.shape == (20, 500, 960, 52)
        assert real.faults == tuple(range(1, 21)) and real.n_runs == 500
        s = real.source
        assert (
            s["doi"] == "10.7910/DVN/6C3JR1" and "public domain" in s["license"].lower()
        )
        assert s["subset"].startswith("none")
        for key, raw in s["raw_files"].items():
            assert (
                len(raw["md5"]) == 32
                and len(raw["sha256"]) == 64
                and raw["bytes"] > 1e6
            ), key
        assert set(s["arrays"]) == set(rieth.CACHE_FILES)

    def test_faults_begin_at_the_documented_samples(self, real):
        arrays = {name: getattr(real, name) for name in rieth.CACHE_FILES}
        assert fetch_mod.verify_onsets(arrays) == {"training": 20, "testing": 160}
        assert real.source["first_faulty_sample_0based"] == {
            "training": 20,
            "testing": 160,
        }

    def test_step_fault_6_cuts_the_a_feed_after_the_onset_only(self, real):
        free = np.asarray(real.fault_free_testing[:100, :, 0], dtype=np.float64)
        runs = np.asarray(real.faulty_testing[5, :100, :, 0], dtype=np.float64)
        mean, std = free.mean(), free.std()
        z = (runs - mean) / std
        assert np.abs(z[:, :ONSET_TEST].mean()) < 0.1
        assert z[:, ONSET_TEST : ONSET_TEST + 20].mean() < -5.0

    def test_runs_are_independent_and_files_do_not_share_seeds(self, real):
        def first(a):
            return {tuple(np.asarray(a[i, 0, :5])) for i in range(a.shape[0])}

        train, test = first(real.fault_free_training), first(real.fault_free_testing)
        assert len(train) == 500 and len(test) == 500
        assert train.isdisjoint(test)

    def test_values_are_finite_and_in_one_unit_system(self, real):
        assert np.isfinite(real.fault_free_training).all()
        assert np.isfinite(real.faulty_training[:, :5]).all()
        assert np.isfinite(real.faulty_testing[:, :5]).all()
        free = np.asarray(real.fault_free_training).reshape(-1, 52)
        braatz_like = default_cache_dir() / "d00.dat"
        if (
            braatz_like.exists()
        ):  # the same variables as the Braatz set, same engineering units
            old = np.loadtxt(braatz_like).T
            assert np.allclose(free.mean(0), old.mean(0), rtol=0.05, atol=0.5)

    def test_a_replicate_of_real_runs_is_disjoint_and_correctly_labelled(self, real):
        splits = RunSplits(
            faults=(1, 6),
            n_replicates=2,
            train_runs=3,
            val_runs=2,
            test_runs=2,
            train_normal_runs=4,
            val_normal_runs=2,
            test_normal_runs=2,
        )
        sel = splits.selection(1)
        assert sel.keys("train").isdisjoint(sel.keys("test"))
        norm = Normaliser.fit(
            real.normal_runs("training", np.sort(sel.train_normal)).reshape(-1, 52)
        )
        test = to_runs(real, norm, sel, "test", (1, 6))
        assert len(test) == 2 + 2 * 2 and test[0].series.shape == (960, 52)
        a_feed = [r for r in test if r.fault == 6]
        assert all(
            r.series[:ONSET_TEST, 0].mean() > -2 > r.series[ONSET_TEST:, 0].mean()
            for r in a_feed
        )
