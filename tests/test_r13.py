"""R13: RE1 parsing (metrics only), the ranker features and the stats. Real-data tests skip."""

import io

import numpy as np
import pytest

from genetic_coil_arena.envs.logtriage import rcaeval
from genetic_coil_arena.envs.logtriage.incident import NoLogsError
from genetic_coil_arena.experiments import r13_real_centre as r13


def _csv(n_pre: int = 60, n_post: int = 60, shift: float = 50.0) -> io.StringIO:
    rng = np.random.default_rng(0)
    rows = ["time,a_cpu,b_cpu,a_load,a_latency,a_mem"]
    for t in range(n_pre + n_post):
        bump = shift if t >= n_pre else 0.0
        rows.append(
            f"{1000 + t},{1 + rng.normal() * 0.1 + bump},{1 + rng.normal() * 0.1},"
            f"{2 + rng.normal() * 0.1},{3 + rng.normal() * 0.1},{4 + rng.normal() * 0.1}"
        )
    return io.StringIO("\n".join(rows) + "\n")


class TestRE1Parse:
    def test_metrics_only_incident(self):
        inc = rcaeval.convert_metrics_case(
            _csv(), 1060.0, "a", "cpu", 1, "RE1-SS", "a_cpu", rcaeval.RE1_KIND_ALIAS
        )
        assert inc.services == ["a", "b"]
        assert not inc.has_logs and inc.n_lines == 0
        assert set(inc.metrics["a"]) == {"cpu", "workload", "latency-50", "mem"}
        assert inc.metrics["a"]["cpu"][0] > 20 > inc.metrics["b"]["cpu"][0]

    def test_log_methods_refuse(self):
        inc = rcaeval.convert_metrics_case(
            _csv(), 1060.0, "a", "cpu", 1, "RE1-SS", "a_cpu"
        )
        with pytest.raises(NoLogsError):
            inc.stats()
        with pytest.raises(NoLogsError):
            inc.line_text(0)

    def test_unknown_root_is_an_error(self):
        with pytest.raises(rcaeval.NoRootLogsError):
            rcaeval.convert_metrics_case(_csv(), 1060.0, "zzz", "cpu", 1, "RE1-SS", "x")

    def test_dataset_and_archives(self):
        assert rcaeval.dataset_of("RE1-TT") == "RE1"
        assert set(rcaeval.RE1_SYSTEMS) <= set(rcaeval.ARCHIVES)

    def test_round_trip_keeps_has_logs(self, tmp_path):
        inc = rcaeval.convert_metrics_case(
            _csv(), 1060.0, "a", "cpu", 1, "RE1-SS", "a_cpu"
        )
        rcaeval.save_incident(inc, tmp_path / "x.npz")
        back = rcaeval.load_incident(tmp_path / "x.npz")
        assert not back.has_logs
        assert not r13.load_metrics_view(tmp_path / "x.npz").has_logs


class _Probs:
    def predict_proba(self, x):
        return np.full((len(x), 6), 1 / 6)


class TestRanker:
    def _data(self, n=60):
        incs = []
        for k in range(n):
            inc = rcaeval.convert_metrics_case(
                _csv(shift=30.0 + k), 1060.0, "a", "cpu", 1, "RE1-SS", f"c{k}"
            )
            inc.id = f"RE1-SS/c{k}/1"
            incs.append(inc)
        return r13.RankData(incs, _Probs())

    def test_features_shape_and_rank(self):
        data = self._data(3)
        assert data.x.shape[2] == len(r13.ALL_NAMES)
        top = data.x[0, :, r13.ALL_NAMES.index("is_top")]
        assert top[0] == 1.0 and top[1] == 0.0

    def test_fit_ranks_root_first(self):
        data = self._data()
        model = r13.ConditionalLogit("all", 0.1).fit(data)
        assert (model.top(data) == data.y).all()
        again = r13.ConditionalLogit.from_dict(model.to_dict())
        assert (again.top(data) == model.top(data)).all()

    def test_choose_config_tie_rule(self):
        table = [
            {"features": "all", "l2": 1.0, "mean_top1": 0.8},
            {"features": "compact", "l2": 0.01, "mean_top1": 0.8},
            {"features": "compact", "l2": 1.0, "mean_top1": 0.8},
        ]
        assert r13.choose_config(table) == 2
        assert len(r13.CONFIGS) == 6

    def test_loso_table_runs(self):
        data = self._data(9)
        data.systems = np.array(["SS", "OB", "TT"] * 3)
        table = r13.loso_table(data)
        assert len(table) == 6 and all("mean_top1" in r for r in table)


class TestStats:
    def test_paired_and_verdicts(self):
        a = np.r_[np.ones(80), np.zeros(20)]
        b = np.r_[np.ones(50), np.zeros(50)]
        r = r13.paired(a, b)
        assert r["diff"] == pytest.approx(0.3) and r["lo"] > 0
        ok = {"diff": 0.2, "lo": 0.1, "hi": 0.3}
        res = {"joint": {"pooled": ok, "SS": ok, "OB": ok, "TT": ok}}
        assert r13.g11a_verdict(res)["verdict"] == "passed"
        res["joint"]["TT"] = {"diff": 0.01, "lo": -0.05, "hi": 0.07}
        assert r13.g11a_verdict(res)["verdict"] == "partial"
        res["joint"]["pooled"] = {"diff": 0.01, "lo": -0.02, "hi": 0.04}
        assert r13.g11a_verdict(res)["verdict"] == "kill"
        top = {"diff": 0.04, "lo": 0.01, "hi": 0.07}
        ni = {"diff": 0.0, "lo": -0.03, "hi": 0.03}
        res = {"top1": {"pooled": top, "SS": ni, "OB": ni, "TT": ni}}
        assert r13.g11b_verdict(res)["verdict"] == "passed"

    def test_confusion(self):
        c = r13.confusion(["cpu", "mem"], ["cpu", "cpu"])
        assert c["cpu"]["cpu"] == 1 and c["mem"]["cpu"] == 1


@pytest.mark.skipif(
    not (rcaeval.incident_dir(rcaeval.cache_dir(), "RE1-SS") / "_done.json").exists(),
    reason="RCAEval RE1 not fetched",
)
class TestRealRE1:
    def test_counts_and_no_failures(self):
        stats = r13.re1_statistics()
        for system in rcaeval.RE1_SYSTEMS:
            assert stats[system]["n_failed"] == 0
            assert stats[system]["n_incidents"] == stats[system]["n_cases_found"] == 125
            assert stats[system]["logs"] is False

    def test_loads_without_logs(self):
        incs = r13.load_re1(limit=2)
        assert len(incs) == 6
        assert all(not i.has_logs and i.metrics[i.root_service] for i in incs)
        assert {i.fault_type for i in incs} <= {"cpu", "mem", "disk", "delay", "loss"}
