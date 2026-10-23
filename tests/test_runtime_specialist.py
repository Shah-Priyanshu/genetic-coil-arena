"""Test capabilities, the Specialist protocol and the ClockAdapter (sync and async)."""

import asyncio

import pytest

from genetic_coil_arena.protocol import (
    GoalSpec,
    complete,
    goal_envelope,
    is_result,
    sign,
    verify,
)
from genetic_coil_arena.runtime import (
    Capabilities,
    ClockAdapter,
    ClockAware,
    FunctionSpecialist,
    Specialist,
)


def goal(spec="math.double", **params):
    return goal_envelope(GoalSpec(spec, params), "centre", "doubler", task_id="t1")


def doubler():
    return FunctionSpecialist("doubler", {"math.double": lambda p: p["x"] * 2})


class Counter:
    """Clock-aware specialist: reports how many envelopes it saw on every tick."""

    def __init__(self):
        self.seen = []

    def capabilities(self):
        return Capabilities(name="counter")

    def handle(self, envelope):
        self.seen.append(envelope.msg_id)
        return []

    def on_tick(self, clock):
        request = goal()
        return [complete(request, "counter", {"clock": clock, "seen": len(self.seen)})]


class TestCapabilities:
    def test_accepts(self):
        caps = Capabilities("s", accepts=("arith.*", "nav.move@2"))
        assert caps.accepts_spec("arith.add", 1) and caps.accepts_spec("arith.mul", 7)
        assert caps.accepts_spec("nav.move", 2) and not caps.accepts_spec("nav.move", 1)
        assert not caps.accepts_spec("nav.turn", 1)
        assert caps.accepts_goal(GoalSpec("arith.add", {}))

    def test_single_string_fields(self):
        caps = Capabilities("s", accepts="a.b", payload_encodings="codes", tags="x")
        assert caps.accepts == ("a.b",) and caps.payload_encodings == ("codes",)
        assert caps.tags == ("x",)

    @pytest.mark.parametrize("accepts", [("a@0",), ("a@x",), ("",)])
    def test_bad_accepts(self, accepts):
        with pytest.raises(ValueError):
            Capabilities("s", accepts=accepts)

    def test_empty_name(self):
        with pytest.raises(ValueError):
            Capabilities("")

    def test_agent_card_round_trip(self):
        caps = Capabilities(
            "s", "3", ("a.b@1", "c.*"), ("goalspec", "codes"), "does things", ("t",)
        )
        card = caps.to_agent_card()
        assert card["name"] == "s" and card["version"] == "3"
        assert [skill["id"] for skill in card["skills"]] == ["a.b@1", "c.*"]
        assert card["defaultInputModes"] == ["goalspec", "codes"]
        assert Capabilities.from_agent_card(card) == caps


class TestProtocols:
    def test_function_specialist_is_a_specialist(self):
        assert isinstance(doubler(), Specialist)
        assert not isinstance(doubler(), ClockAware)
        assert isinstance(Counter(), ClockAware)

    def test_adapter_requires_specialist(self):
        with pytest.raises(TypeError):
            ClockAdapter(object())


class TestFunctionSpecialist:
    def test_answers_goal(self):
        request = goal(x=21)
        [reply] = doubler().handle(request)
        assert is_result(reply)
        assert reply.payload == {"state": "completed", "result": 42}
        assert reply.parent_id == request.msg_id and reply.recipients == ("centre",)
        assert reply.task_id == "t1"

    def test_exception_becomes_failure(self):
        [reply] = doubler().handle(goal())
        assert reply.payload["state"] == "failed"
        assert "KeyError" in reply.payload["error"]

    def test_unknown_spec_fails(self):
        [reply] = doubler().handle(goal("math.triple", x=1))
        assert "does not handle" in reply.payload["error"]

    def test_ignores_non_goals(self):
        specialist = doubler()
        assert specialist.handle(complete(goal(), "x", 1)) == []
        assert specialist.handled == 0

    def test_capabilities(self):
        caps = doubler().capabilities()
        assert caps.name == "doubler" and caps.accepts == ("math.double",)


class TestClockAdapter:
    def test_queues_until_tick(self):
        adapter = ClockAdapter(doubler())
        adapter.receive(goal(x=1))
        adapter.receive(goal(x=2))
        assert adapter.pending == 2
        outputs = adapter.tick(5)
        assert adapter.clock == 5 and adapter.pending == 0
        assert [o.payload["result"] for o in outputs] == [2, 4]
        assert adapter.tick() == [] and adapter.clock == 6

    def test_step_is_observe_then_act(self):
        counter = Counter()
        adapter = ClockAdapter(counter)
        [report] = adapter.step([goal(), goal()], clock=3)
        assert report.payload["result"] == {"clock": 3, "seen": 2}
        [report] = adapter.step([])
        assert report.payload["result"] == {"clock": 4, "seen": 2}

    def test_same_specialist_sync_and_async(self):
        specialist = doubler()
        sync_out = ClockAdapter(specialist).step([goal(x=5)])

        async def run():
            inbox, sent = asyncio.Queue(), []
            adapter = ClockAdapter(specialist)
            for x in (5, 6):
                await inbox.put(goal(x=x))
            await inbox.put(None)
            processed = await adapter.serve(inbox, sent.append)
            return processed, sent

        processed, async_out = asyncio.run(run())
        assert processed == 2
        assert [o.payload for o in sync_out] == [async_out[0].payload]
        assert async_out[1].payload["result"] == 12

    def test_serve_awaits_async_send(self):
        async def run():
            inbox, sent = asyncio.Queue(), []

            async def send(envelope):
                await asyncio.sleep(0)
                sent.append(envelope)

            consumer = asyncio.ensure_future(ClockAdapter(doubler()).serve(inbox, send))
            await inbox.put(goal(x=1))
            await inbox.join()
            assert len(sent) == 1
            await inbox.put(None)
            return await consumer

        assert asyncio.run(run()) == 1

    def test_signs_outputs(self):
        adapter = ClockAdapter(doubler(), signing_key=b"k")
        [reply] = adapter.step([goal(x=1)])
        assert reply.manifest["key_id"] == "doubler"
        assert verify(reply, {"doubler": b"k"})

    def test_rejects_unverified_inputs(self):
        adapter = ClockAdapter(doubler(), verify_keys={"centre": b"c"})
        good = sign(goal(x=1), b"c", "centre")
        forged = sign(goal(x=1), b"wrong", "centre")
        outputs = adapter.step([good, goal(x=2), forged])
        assert len(outputs) == 1
        assert [r.reason for r in adapter.rejected] == ["unsigned", "bad signature"]

    def test_non_envelope_output_is_an_error(self):
        class Broken:
            def capabilities(self):
                return Capabilities("broken")

            def handle(self, envelope):
                return ["not an envelope"]

        with pytest.raises(TypeError, match="not Envelope"):
            ClockAdapter(Broken()).step([goal()])
