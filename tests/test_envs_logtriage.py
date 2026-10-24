"""Log-triage data layer: templating, incidents, views, heuristics, messages, RCAEval convert.

Synthetic and tiny inline data only; the real-data test skips when the RCAEval cache is
absent.
"""

import io
import json

import numpy as np
import pytest

from genetic_coil_arena.envs.logtriage import heuristics as hz
from genetic_coil_arena.envs.logtriage import messages as msgs
from genetic_coil_arena.envs.logtriage import rcaeval, views
from genetic_coil_arena.envs.logtriage.incident import Incident
from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
from genetic_coil_arena.envs.logtriage.templating import make_template, severity
from genetic_coil_arena.protocol.goalspec import SpecValidationError

COUNTER = views.ApproxCounter()


class TestTemplating:
    def test_variable_parts_are_masked(self):
        a = make_template(
            "2024-01-19 11:10:29.212  INFO [carts,a1af5f5ba2a21018ec1b79e6a4510f4e,"
            "1494a97596f12245,false] 1 --- [p-nio-80-exec-6] Opened connection 7 to db:27017"
        )
        b = make_template(
            "2024-01-19 11:11:02.900  INFO [carts,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,"
            "1111111111111111,false] 1 --- [p-nio-80-exec-9] Opened connection 9 to db:27017"
        )
        assert a == b
        assert "<ts>" in a and "<trace>" in a and "<thread>" in a

    def test_http_status_and_latency_buckets_survive(self):
        t = make_template("POST /orders 503 54.540 ms - -")
        assert "HTTP503" in t and "lat_tens_ms" in t
        fast = make_template("took=2.7ms")
        slow = make_template("took=2.7s")
        assert fast != slow
        blob = make_template(
            'Order response: {"statusCode":503,"body":"upstream connect error or reset",'
            '"headers":{"content-length":152,"content-type":"text/plain"}}'
        )
        assert "HTTP503" in blob and "upstream connect error" in blob

    def test_keyvalue_values_masked_but_method_kept(self):
        t = make_template("caller=x.go:40 method=Health user=abc123 result=1")
        assert "method=Health" in t and "user=<*>" in t

    def test_severity(self):
        assert severity("ERROR something broke") == 2
        assert severity("connection refused") == 1
        assert severity("request ok") == 0
        assert severity("request ok", level="ERROR") == 2

    def test_long_lines_are_cut(self):
        assert len(make_template("x" * 1000)) <= 200


class TestIncident:
    def test_stats_counts_pre_and_post(self):
        inc = synthetic_incident(1)
        st = inc.stats()
        assert int(st.pre.sum()) == int((inc.t < 0).sum())
        assert int(st.post.sum()) == int((inc.t >= 0).sum())
        pre, post = inc.service_counts()
        assert pre.sum() + post.sum() == inc.n_lines
        assert np.isfinite(st.first_post[st.post > 0]).all()

    def test_root_must_be_a_service(self):
        inc = synthetic_incident(2)
        with pytest.raises(ValueError, match="not in services"):
            Incident(
                id="x",
                system="s",
                services=["a", "b"],
                root_service="c",
                fault_type="cpu",
                rep=1,
                t=inc.t[:5],
                svc=inc.svc[:5],
                tpl=inc.tpl[:5],
                templates=inc.templates,
            )

    def test_unsorted_lines_rejected(self):
        with pytest.raises(ValueError, match="sorted"):
            Incident(
                id="x",
                system="s",
                services=["a"],
                root_service="a",
                fault_type="cpu",
                rep=1,
                t=np.array([2.0, 1.0], dtype=np.float32),
                svc=np.zeros(2, dtype=np.int16),
                tpl=np.zeros(2, dtype=np.int32),
                templates=["t"],
            )


class TestViews:
    def test_naive_head_and_tail_respect_budget_and_differ(self):
        inc = synthetic_incident(3)
        head = views.naive_view(inc, 300, COUNTER)
        tail = views.naive_view(inc, 300, COUNTER, tail=True)
        assert COUNTER.count(head) <= 300 and COUNTER.count(tail) <= 300
        assert head != tail
        assert head.splitlines()[0].startswith("-")  # earliest lines first
        assert (
            float(tail.splitlines()[-1].split("s ")[0]) == float(inc.t[-1]).__round__()
        )

    def test_selected_view_fits_and_shows_the_fault_line(self):
        inc = synthetic_incident(4)
        text = views.selected_view(inc, 700, COUNTER)
        assert COUNTER.count(text) <= 700
        assert inc.root_service in text
        assert inc.templates[len(inc.templates) - 2] in text  # the injected error line
        for name in inc.services:
            assert f"{name}: lines before/after" in text

    def test_selected_view_tiny_budget_drops_services_not_the_budget(self):
        inc = synthetic_incident(5)
        text = views.selected_view(inc, 120, COUNTER)
        assert COUNTER.count(text) <= 120

    def test_source_view_only_shows_its_own_service(self):
        inc = synthetic_incident(6)
        s = inc.services.index(inc.root_service)
        text = views.source_view(inc, s, 500, COUNTER)
        other = [n for n in inc.services if n != inc.root_service]
        assert all(f"[{n}]" not in text for n in other)
        assert f"[{inc.root_service}]" in text

    def test_metrics_can_be_left_out(self):
        inc = synthetic_incident(7)
        with_m = views.selected_view(inc, 2000, COUNTER, use_metrics=True)
        without = views.selected_view(inc, 2000, COUNTER, use_metrics=False)
        assert "metrics:" in with_m and "metrics:" not in without

    def test_overflow_is_large_for_a_big_incident(self):
        inc = synthetic_incident(8, lines_per_service=3000)
        assert COUNTER.count(views.full_text(inc)) > 100 * 4096 / 10


class TestHeuristics:
    def test_signal_is_found_in_synthetic_data(self):
        hits = {"baro_logs": 0, "surprise": 0, "baro_metrics": 0}
        n = 12
        for seed in range(n):
            inc = synthetic_incident(seed)
            for name, fn in (
                ("baro_logs", hz.baro_logs),
                ("surprise", hz.surprise_sum),
                ("baro_metrics", hz.baro_metrics),
            ):
                top = hz.ranking(fn(inc))[0]
                hits[name] += inc.services[top] == inc.root_service
        assert hits["baro_metrics"] == n
        assert hits["surprise"] >= n - 2
        assert hits["baro_logs"] >= n - 3

    def test_guess_fault_keywords(self):
        assert hz.guess_fault(["ERROR java.lang.OutOfMemoryError heap"]) == "mem"
        assert hz.guess_fault(["write failed: no space left on device"]) == "disk"
        assert hz.guess_fault(["connection reset by peer"]) == "loss"
        assert hz.guess_fault(["nothing special"], default="x") == "x"
        # fewer weights than texts (a placeholder text was added): no crash
        assert hz.guess_fault(["connection reset", "slow"], weights=[2.0]) == "loss"

    def test_quiet_service_message_is_valid(self):
        inc = synthetic_incident(3)
        # a service whose lines nearly vanish after the alert and has no rising template
        quiet = (inc.svc == 1) & (inc.t >= 0)
        keep = ~quiet | (np.arange(inc.n_lines) % 25 == 0)
        inc2 = Incident(
            id=inc.id,
            system=inc.system,
            services=inc.services,
            root_service=inc.root_service,
            fault_type=inc.fault_type,
            rep=inc.rep,
            t=inc.t[keep],
            svc=inc.svc[keep],
            tpl=inc.tpl[keep],
            templates=inc.templates,
            metrics={},
        )
        assert msgs.message_errors(msgs.heuristic_message(inc2, 1)) == []

    def test_ranking_is_stable(self):
        assert hz.ranking(np.array([1.0, 3.0, 3.0, 0.0])) == [1, 2, 0, 3]


class TestMessages:
    def test_heuristic_message_is_schema_valid_and_points_at_root(self):
        inc = synthetic_incident(9)
        for s in range(inc.n_services):
            m = msgs.heuristic_message(inc, s)
            assert msgs.message_errors(m) == []
        root = inc.services.index(inc.root_service)
        m = msgs.heuristic_message(inc, root)
        assert m["anomaly"] is True and m["top_errors"]
        assert m["suspected_fault"] == inc.fault_type

    def test_schema_rejects_bad_messages(self):
        good = msgs.empty_message("a")
        assert msgs.message_errors(good) == []
        assert msgs.message_errors({**good, "suspected_fault": "gremlins"})
        assert msgs.message_errors({**good, "extra": 1})
        bad = dict(good)
        del bad["anomaly"]
        assert msgs.message_errors(bad)
        too_many = {
            **good,
            "top_errors": [{"template": "t", "count": 1}] * 4,
        }
        assert msgs.message_errors(too_many)

    def test_fit_message_meets_budget_by_shrinking_then_dropping(self):
        m = {
            **msgs.empty_message("svc"),
            "anomaly": True,
            "top_errors": [{"template": "x" * 100, "count": 5} for _ in range(3)],
        }
        big = COUNTER.count(msgs.message_json(m))
        fitted = msgs.fit_message(m, COUNTER, 60)
        assert COUNTER.count(msgs.message_json(fitted)) <= 60 < big
        assert msgs.message_errors(fitted) == []
        assert len(m["top_errors"]) == 3  # the input is not modified

    def test_truncate_text(self):
        assert COUNTER.count(msgs.truncate_text("word " * 200, COUNTER, 20)) <= 20
        assert msgs.truncate_text("  short  ", COUNTER, 20) == "short"

    def test_parse_json_object_finds_the_first_valid_object(self):
        text = 'Sure! {"a": {"b": "}"}, "c": 1} and {"d": 2}'
        assert msgs.parse_json_object(text) == {"a": {"b": "}"}, "c": 1}
        assert msgs.parse_json_object("no json") is None
        assert msgs.parse_json_object("{broken") is None
        assert msgs.parse_json_object('[1,2] {"x":1}') == {"x": 1}

    def test_goal_spec_is_validated(self):
        registry = msgs.triage_registry()
        inc = synthetic_incident(10)
        goal = msgs.make_goal(registry, inc, 4096, 128)
        assert goal.params["candidates"] == inc.services
        assert (
            registry.result_errors(goal, {"ranking": ["svc0"], "fault_type": "cpu"})
            == []
        )
        assert registry.result_errors(goal, {"ranking": ["svc0"], "fault_type": "gc"})
        with pytest.raises(SpecValidationError):
            msgs.make_goal(registry, inc, 100)  # context below the schema minimum


class TestRCAEvalConvert:
    LOGS = (
        "time,timestamp,container_name,message,level,req_path,error\n"
        "08:04,1000000000000,carts,hello 1,info,,\n"
        '08:04,1005000000000,front-end,"POST /orders 503 12 ms - -",,,\n'
        "08:05,1020000000000,carts,hello 2,info,,\n"
        "08:05,1019000000000,carts,boom,error,,\n"
    )
    METRICS = "time,carts_cpu,carts_latency-50,front-end_cpu\n" + "".join(
        f"{990 + i},{1.0 + (50 if i >= 10 and k == 0 else 0)},{0.1},{2.0}\n"
        for i in range(20)
        for k in (0,)
    )

    def test_convert_case_builds_a_sorted_incident(self):
        inc = rcaeval.convert_case(
            io.StringIO(self.LOGS),
            1000.0,
            "carts",
            "cpu",
            2,
            "SS",
            "carts_cpu",
            io.StringIO(self.METRICS),
        )
        assert inc.id == "SS/carts_cpu/2"
        assert inc.services == ["carts", "front-end"]
        assert inc.n_lines == 4 and np.all(np.diff(inc.t) >= 0)
        assert inc.t[0] == pytest.approx(0.0)
        templates = " ".join(inc.templates)
        assert "[ERROR] boom" in templates and "HTTP503" in templates
        z, direction, first, rel = inc.metrics["carts"]["cpu"]
        assert z > 10 and direction == 1 and first == pytest.approx(0.0)
        assert inc.metrics["front-end"]["cpu"][0] < 3

    def test_root_without_any_data_is_reported(self):
        with pytest.raises(rcaeval.NoRootLogsError):
            rcaeval.convert_case(
                io.StringIO(self.LOGS), 1000.0, "nginx", "cpu", 1, "SS", "nginx_cpu"
            )

    def test_missing_columns(self):
        with pytest.raises(ValueError, match="column"):
            rcaeval.convert_case(
                io.StringIO("a,b\n1,2\n"), 0.0, "x", "cpu", 1, "SS", "x_cpu"
            )

    def test_save_load_roundtrip(self, tmp_path):
        inc = synthetic_incident(11)
        path = tmp_path / "x.npz"
        rcaeval.save_incident(inc, path)
        back = rcaeval.load_incident(path)
        assert back.id == inc.id and back.services == inc.services
        assert np.array_equal(back.t, inc.t) and np.array_equal(back.tpl, inc.tpl)
        assert back.metrics == json.loads(json.dumps(inc.metrics))

    def test_split_and_case_names(self):
        assert rcaeval.parse_case("payment_mem") == ("payment", "mem")
        assert rcaeval.parse_case("recommendation_service_cpu") == (
            "recommendation_service",
            "cpu",
        )
        assert rcaeval.split_of(synthetic_incident(1, rep=1)) == "dev"
        assert rcaeval.split_of(synthetic_incident(1, rep=2)) == "test"

    def test_cache_dir_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GCA_RCAEVAL_DIR", str(tmp_path))
        assert rcaeval.cache_dir() == tmp_path
        assert rcaeval.list_incidents() == []


@pytest.mark.skipif(
    not rcaeval.list_incidents(),
    reason="RCAEval cache absent (scripts/fetch_rcaeval.py)",
)
def test_real_data_has_labels_and_overflows():
    files = rcaeval.list_incidents()
    inc = rcaeval.load_incident(files[0])
    assert inc.root_service in inc.services
    assert inc.fault_type in rcaeval.FAULT_TYPES
    assert inc.n_lines > 10_000
    assert COUNTER.count(views.full_text(inc)) > 100 * 4096
    text = views.selected_view(inc, 3000, COUNTER)
    assert COUNTER.count(text) <= 3000
