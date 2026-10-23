"""Tests for the Tennessee Eastman data layer: variables, node mapping, loading, fetching."""

import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from genetic_coil_arena.envs.plant import fetch as fetch_mod
from genetic_coil_arena.envs.plant.tep import (
    FAULT_DESCRIPTIONS,
    FAULT_IDS,
    N_VARIABLES,
    NODE_VARIABLES,
    NODES,
    ONSET_TEST,
    SOURCE_LICENSE,
    SOURCE_URL,
    VARIABLES,
    TepData,
    default_cache_dir,
    describe_mapping,
    synthetic_tep,
)

REPO = Path(__file__).resolve().parents[1]


class TestVariablesAndNodes:
    def test_52_variables_in_published_order(self):
        assert len(VARIABLES) == N_VARIABLES == 52
        assert [v.index for v in VARIABLES] == list(range(52))
        tags = [v.tag for v in VARIABLES]
        assert tags[0] == "XMEAS(1)" and tags[40] == "XMEAS(41)"
        assert tags[41] == "XMV(1)" and tags[51] == "XMV(11)"
        assert len(set(tags)) == 52

    def test_published_descriptions_of_key_variables(self):
        by_tag = {v.tag: v for v in VARIABLES}
        assert by_tag["XMEAS(1)"].name.startswith("A feed")
        assert by_tag["XMEAS(7)"].name == "Reactor pressure"
        assert by_tag["XMEAS(9)"].name == "Reactor temperature"
        assert by_tag["XMEAS(12)"].name == "Product separator level"
        assert by_tag["XMEAS(18)"].name == "Stripper temperature"
        assert by_tag["XMEAS(20)"].name == "Compressor work"
        assert by_tag["XMV(5)"].name == "Compressor recycle valve"
        assert by_tag["XMV(9)"].name == "Stripper steam valve"

    def test_four_nodes_partition_all_variables_exactly_once(self):
        assert NODES == ("reactor", "separator", "stripper", "compressor_utilities")
        cols = [c for node in NODES for c in NODE_VARIABLES[node]]
        assert sorted(cols) == list(range(52))
        assert [len(NODE_VARIABLES[n]) for n in NODES] == [18, 15, 10, 9]

    def test_node_assignment_matches_the_process_units(self):
        by_tag = {v.tag: v.node for v in VARIABLES}
        assert (
            by_tag["XMEAS(7)"] == by_tag["XMEAS(8)"] == by_tag["XMEAS(9)"] == "reactor"
        )
        assert by_tag["XMEAS(12)"] == by_tag["XMEAS(13)"] == "separator"
        assert by_tag["XMEAS(15)"] == by_tag["XMEAS(18)"] == "stripper"
        assert by_tag["XMEAS(20)"] == by_tag["XMV(5)"] == "compressor_utilities"
        assert by_tag["XMEAS(21)"] == by_tag["XMV(10)"] == "compressor_utilities"
        assert by_tag["XMEAS(23)"] == "reactor"  # reactor feed analyser
        assert by_tag["XMEAS(29)"] == "separator"  # purge gas analyser
        assert by_tag["XMEAS(37)"] == "stripper"  # product analyser

    def test_mapping_can_be_documented(self):
        mapping = describe_mapping()
        assert set(mapping) == set(NODES)
        assert sum(len(v) for v in mapping.values()) == 52
        assert all("XMEAS" in s or "XMV" in s for s in mapping["reactor"])

    def test_21_faults_with_descriptions(self):
        assert FAULT_IDS == tuple(range(1, 22))
        assert set(FAULT_DESCRIPTIONS) == set(FAULT_IDS)
        assert ONSET_TEST == 160


class TestSyntheticAndLoading:
    def test_synthetic_has_the_real_shapes_and_is_deterministic(self):
        a, b = synthetic_tep(seed=1), synthetic_tep(seed=1)
        assert a.normal_train.shape == (500, 52) and a.normal_test.shape == (960, 52)
        assert all(x.shape == (480, 52) for x in a.fault_train.values())
        assert all(x.shape == (960, 52) for x in a.fault_test.values())
        assert a.faults == FAULT_IDS
        assert np.array_equal(a.fault_test[7], b.fault_test[7])
        assert not np.array_equal(a.fault_test[7], synthetic_tep(seed=2).fault_test[7])

    def test_synthetic_faults_begin_at_the_documented_onsets(self):
        data = synthetic_tep(seed=0, faults=(1,), strength=8.0)
        norm = (data.fault_test[1] - data.normal_train.mean(0)) / data.normal_train.std(
            0
        )
        shifted = np.argmax(np.abs(norm).max(axis=1)[150:] > 6) + 150
        assert 160 <= shifted <= 166
        train = (
            data.fault_train[1] - data.normal_train.mean(0)
        ) / data.normal_train.std(0)
        assert np.abs(train[10:]).max() > 6

    def write_cache(self, directory, data):
        directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(directory / "d00.dat", data.normal_train.T)  # stored transposed
        np.savetxt(directory / "d00_te.dat", data.normal_test)
        for f in FAULT_IDS:
            np.savetxt(directory / f"d{f:02d}.dat", data.fault_train[f])
            np.savetxt(directory / f"d{f:02d}_te.dat", data.fault_test[f])

    def test_loader_reads_the_real_file_layout(self, tmp_path):
        data = synthetic_tep(seed=4)
        self.write_cache(tmp_path, data)
        loaded = TepData.from_directory(tmp_path)
        assert loaded.normal_train.shape == (500, 52)  # d00.dat transposed back
        assert np.allclose(loaded.normal_train, data.normal_train)
        assert np.allclose(loaded.fault_test[21], data.fault_test[21])

    def test_missing_cache_points_to_the_fetch_script(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fetch_tep"):
            TepData.from_directory(tmp_path)

    def test_bad_shapes_are_rejected(self):
        good = synthetic_tep(seed=0, faults=(1,))
        with pytest.raises(ValueError):
            TepData(good.normal_train[:, :10], good.normal_test, {}, {})
        with pytest.raises(ValueError):
            TepData(
                good.normal_train, good.normal_test, {1: good.normal_train[:, :3]}, {}
            )

    def test_cache_dir_is_outside_the_repo_and_overridable(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GCA_TEP_DIR", raising=False)
        default = default_cache_dir().resolve()
        assert REPO not in default.parents and default != REPO
        assert default.parts[-2:] == ("genetic_coil_arena", "tep")
        monkeypatch.setenv("GCA_TEP_DIR", str(tmp_path / "x"))
        assert default_cache_dir() == tmp_path / "x"


class TestFetchVerification:
    def test_manifest_covers_all_44_files_with_published_shapes(self):
        assert len(fetch_mod.MANIFEST) == 44
        assert fetch_mod.MANIFEST["d00.dat"][1] == (52, 500)
        assert fetch_mod.MANIFEST["d00_te.dat"][1] == (960, 52)
        assert fetch_mod.MANIFEST["d05.dat"][1] == (480, 52)
        assert fetch_mod.MANIFEST["d05_te.dat"][1] == (960, 52)
        assert all(len(h) == 64 for h, _ in fetch_mod.MANIFEST.values())

    def test_check_file_rejects_altered_or_unknown_files(self):
        with pytest.raises(ValueError, match="sha256"):
            fetch_mod.check_file("d01.dat", b"1.0 2.0\n")
        with pytest.raises(ValueError, match="Unexpected"):
            fetch_mod.check_file("evil.dat", b"")

    def test_check_file_accepts_a_matching_file(self, monkeypatch):
        body = b"1.0 2.0\n3.0 4.0\n"
        monkeypatch.setitem(
            fetch_mod.MANIFEST, "d01.dat", (fetch_mod.sha256(body), (2, 2))
        )
        fetch_mod.check_file("d01.dat", body)
        monkeypatch.setitem(
            fetch_mod.MANIFEST, "d01.dat", (fetch_mod.sha256(body), (3, 2))
        )
        with pytest.raises(ValueError, match="shape"):
            fetch_mod.check_file("d01.dat", body)

    def test_incomplete_cache_is_reported(self, tmp_path):
        assert not fetch_mod.is_complete(tmp_path)
        (tmp_path / "d00.dat").write_text("not the data")
        assert not fetch_mod.is_complete(tmp_path)

    def test_archive_must_match_the_pinned_checksum(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("TE_process/d00.dat", "1 2 3")
        with pytest.raises(ValueError, match="Archive mismatch"):
            fetch_mod._from_archive(buf.getvalue())

    def test_source_and_licence_are_recorded(self):
        assert SOURCE_URL.startswith("https://web.mit.edu/braatzgroup/")
        assert "Braatz" in SOURCE_LICENSE and "without restriction" in SOURCE_LICENSE
        assert len(fetch_mod.ARCHIVE_SHA256) == 64

    def test_fetch_script_is_a_thin_cli(self):
        path = REPO / "scripts" / "fetch_tep.py"
        spec = importlib.util.spec_from_file_location("fetch_tep_script", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)


# ----------------------------------------------------------------- real data (optional)
needs_data = pytest.mark.skipif(
    not fetch_mod.is_complete(default_cache_dir()),
    reason="TEP data not fetched (python scripts/fetch_tep.py)",
)


@needs_data
class TestRealData:
    @pytest.fixture(scope="class")
    def real(self):
        return TepData.from_cache()

    def test_shapes_and_provenance(self, real):
        assert real.normal_train.shape == (500, 52) and real.normal_test.shape == (
            960,
            52,
        )
        assert real.faults == FAULT_IDS
        assert all(x.shape == (480, 52) for x in real.fault_train.values())
        assert all(x.shape == (960, 52) for x in real.fault_test.values())
        assert real.source["url"] == SOURCE_URL

    def test_faults_begin_at_sample_160_of_the_test_runs(self, real):
        mean, std = real.normal_train.mean(0), real.normal_train.std(0)
        for fault in (6, 1):  # step faults with a clear signature
            z = np.abs((real.fault_test[fault] - mean) / std).max(axis=1)
            assert z[:160].max() < 4.5, fault
            assert z[160:180].max() > 5.0, fault

    def test_fault_training_runs_are_faulty_from_the_start(self, real):
        mean, std = real.normal_train.mean(0), real.normal_train.std(0)
        z = np.abs((real.fault_train[6] - mean) / std).max(axis=1)
        assert z[:5].min() > 5.0  # already deviating at the first samples
