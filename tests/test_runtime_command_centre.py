"""Test the bus, specialist registry, router and command centre."""

import pytest

from genetic_coil_arena.protocol import (
    Envelope,
    GoalSpec,
    SpecDefinition,
    SpecRegistry,
    SpecValidationError,
    complete,
    fail,
    goal_envelope,
    goal_from_envelope,
    is_control,
    is_goal,
)
from genetic_coil_arena.runtime import (
    Capabilities,
    ClockAdapter,
    CommandCentre,
    FunctionSpecialist,
    InProcessBus,
    MessageBus,
    Router,
    RoutingRule,
    RuleBasedRouter,
    SpecialistRegistry,
    TraceLog,
)

NUM = {"type": "number"}
PAIR = {
    "type": "object",
    "required": ["a", "b"],
    "properties": {"a": NUM, "b": NUM},
    "additionalProperties": False,
}


@pytest.fixture
def specs():
    registry = SpecRegistry()
    registry.register(SpecDefinition("arith.add", 1, PAIR, result_schema=NUM))
    registry.register(SpecDefinition("arith.add", 2, PAIR, result_schema=NUM))
    registry.register(SpecDefinition("arith.mul", 1, PAIR, result_schema=NUM))
    registry.register(SpecDefinition("misc.wait", 1, {"type": "object"}))
    registry.register(SpecDefinition("misc.pair", 1, PAIR))
    return registry


def adder(name="adder"):
    return FunctionSpecialist(name, {"arith.add": lambda p: p["a"] + p["b"]})


def envelope(to="x", sender="s"):
    return Envelope(sender=sender, recipients=to, payload_type="json", payload=1)


class Silent:
    """Accepts misc.wait and never answers; remembers control messages."""

    def __init__(self, name="silent"):
        self.name = name
        self.controls = []

    def capabilities(self):
        return Capabilities(self.name, accepts=("misc.wait",))

    def handle(self, env):
        if is_control(env):
            self.controls.append(env)
        return []


class Scripted:
    """Answers misc.pair goals with a fixed reply built by ``make_reply``."""

    def __init__(self, make_reply, name="scripted"):
        self.make_reply = make_reply
        self.name = name

    def capabilities(self):
        return Capabilities(self.name, accepts=("misc.pair",))

    def handle(self, env):
        return [self.make_reply(env)] if is_goal(env) else []


# ---------------------------------------------------------------------- bus


class TestBus:
    def test_rounds_and_taps(self):
        bus = InProcessBus()
        assert isinstance(bus, MessageBus)
        got, tapped = [], []
        bus.subscribe("x", got.append)
        bus.add_tap(tapped.append)
        first = envelope()
        bus.publish(first)
        assert bus.pending == 1 and tapped == [first] and got == []
        assert bus.deliver() == 1 and got == [first] and bus.pending == 0
        assert bus.rounds == 1

    def test_publish_during_delivery_waits(self):
        bus = InProcessBus()
        got = []

        def echo(env):
            got.append(env)
            if len(got) == 1:
                bus.publish(envelope())

        bus.subscribe("x", echo)
        bus.publish(envelope())
        assert bus.deliver() == 1 and bus.pending == 1
        assert bus.deliver() == 1 and len(got) == 2

    def test_multicast_and_dead_letters(self):
        bus = InProcessBus()
        got = []
        bus.subscribe("x", got.append)
        bus.subscribe("y", got.append)
        bus.publish(envelope(to=("x", "y", "z")))
        assert bus.deliver() == 2
        assert [(d.recipient, d.reason) for d in bus.dead_letters] == [
            ("z", "no subscriber")
        ]

    def test_subscription_management(self):
        bus = InProcessBus()
        bus.subscribe("x", print)
        with pytest.raises(ValueError):
            bus.subscribe("x", print)
        bus.unsubscribe("x")
        bus.unsubscribe("x")
        assert not bus.is_subscribed("x")
        with pytest.raises(TypeError):
            bus.publish("not an envelope")


# ---------------------------------------------------------------------- registry + router


class TestRegistry:
    def test_register_lookup_unregister(self):
        registry = SpecialistRegistry()
        record = registry.register(adder(), scores={"score": 3})
        assert record.name == "adder" and record.version == "1"
        assert isinstance(record.adapter, ClockAdapter)
        assert registry.lookup("adder") is record and "adder" in registry
        assert record.specialist.name == "adder"
        with pytest.raises(ValueError):
            registry.register(adder())
        registry.set_score("adder", "elo", 1200)
        assert record.scores == {"score": 3.0, "elo": 1200.0}
        assert registry.unregister("adder") is record
        assert len(registry) == 0 and registry.get("adder") is None
        with pytest.raises(KeyError):
            registry.lookup("adder")
        with pytest.raises(KeyError):
            registry.unregister("adder")

    def test_replace_and_adapter_passthrough(self):
        registry = SpecialistRegistry()
        registry.register(adder())
        adapter = ClockAdapter(adder(), signing_key=b"k")
        record = registry.register(adapter, replace=True)
        assert record.adapter is adapter and len(registry) == 1

    def test_find_by_spec_and_encoding(self):
        registry = SpecialistRegistry()
        registry.register(adder("a1"))
        registry.register(adder("a2"))
        registry.register(Silent())
        goal = GoalSpec("arith.add", {})
        assert [r.name for r in registry.find(goal)] == ["a1", "a2"]
        assert registry.find(goal, encoding="codes") == []
        assert registry.names() == ["a1", "a2", "silent"]
        cards = registry.agent_cards()
        assert cards[0]["name"] == "a1" and cards[0]["scores"] == {}


class TestRouter:
    def make_registry(self):
        registry = SpecialistRegistry()
        registry.register(adder("a1"), scores={"score": 1})
        registry.register(adder("a2"), scores={"score": 5})
        registry.register(adder("a3"))
        return registry

    def test_highest_score_then_registration_order(self):
        registry = self.make_registry()
        router = RuleBasedRouter()
        assert isinstance(router, Router)
        assert router.route(GoalSpec("arith.add", {}), registry) == "a2"
        assert (
            RuleBasedRouter(score_key="elo").route(GoalSpec("arith.add", {}), registry)
            == "a1"
        )
        assert router.route(GoalSpec("arith.mul", {}), registry) is None

    def test_rules_take_precedence(self):
        registry = self.make_registry()
        router = RuleBasedRouter([RoutingRule("arith.*", "a3")])
        assert router.route(GoalSpec("arith.add", {}), registry) == "a3"

    def test_rule_predicate_and_fallthrough(self):
        registry = self.make_registry()
        router = RuleBasedRouter(
            [
                RoutingRule("arith.add", "ghost"),  # not registered: skipped
                RoutingRule(
                    "arith.add", "a1", when=lambda g: g.params.get("a", 0) > 10
                ),
            ]
        )
        assert router.route(GoalSpec("arith.add", {"a": 1}), registry) == "a2"
        assert router.route(GoalSpec("arith.add", {"a": 11}), registry) == "a1"
        router.add_rule(RoutingRule("arith.add", "a3"))
        assert router.route(GoalSpec("arith.add", {"a": 1}), registry) == "a3"

    def test_rule_ignored_if_target_does_not_accept(self):
        registry = self.make_registry()
        registry.register(Silent())
        router = RuleBasedRouter([RoutingRule("arith.add", "silent")])
        assert router.route(GoalSpec("arith.add", {}), registry) == "a2"


# ---------------------------------------------------------------------- command centre


class TestCommandCentre:
    def test_simple_task_lifecycle(self, specs):
        trace = TraceLog()
        centre = CommandCentre(specs, trace=trace)
        centre.register(adder())
        task = centre.submit(
            specs.make("arith.add", {"a": 2, "b": 3}), situation={"why": "test"}
        )
        assert centre.status(task).state == "submitted"
        centre.step()
        status = centre.status(task)
        assert status.state == "working" and status.assignee == "adder"
        centre.step()
        assert centre.status(task).state == "working"
        centre.step()
        status = centre.status(task)
        assert status.state == "completed" and status.result == 5 and status.done
        assert centre.result(task) == 5 and centre.idle
        assert trace.situation(status.trace_id) == {"why": "test"}
        states = [e.data["state"] for e in trace.events("task_state")]
        assert states == ["submitted", "working", "completed"]
        triple = trace.triples()[0]
        assert triple.spec.spec == "arith.add" and len(triple.messages) == 2

    def test_submit_validates(self, specs):
        centre = CommandCentre(specs)
        with pytest.raises(SpecValidationError):
            centre.submit({"spec": "arith.add", "params": {"a": "x", "b": 1}})
        with pytest.raises(SpecValidationError):
            centre.submit(GoalSpec("unknown.spec", {}))
        with pytest.raises(
            SpecValidationError, match="goal_id"
        ):  # dicts must be complete
            centre.submit({"spec": "arith.add", "params": {"a": 1, "b": 1}})
        data = specs.make("arith.add", {"a": 1, "b": 1}).to_dict()
        task = centre.submit(data)
        assert centre.status(task).goal_id == data["goal_id"]

    def test_unknown_task(self, specs):
        centre = CommandCentre(specs)
        for call in (centre.status, centre.cancel, centre.result):
            with pytest.raises(KeyError):
                call("nope")

    def test_rejected_without_specialist(self, specs):
        centre = CommandCentre(specs)
        task = centre.submit(specs.make("arith.mul", {"a": 1, "b": 2}))
        centre.step()
        status = centre.status(task)
        assert status.state == "rejected" and "no specialist" in status.error
        with pytest.raises(RuntimeError, match="rejected"):
            centre.result(task)

    def test_router_choosing_unknown_name_rejects(self, specs):
        class Bad:
            def route(self, goal, registry):
                return "ghost"

        centre = CommandCentre(specs, router=Bad())
        task = centre.submit(specs.make("arith.add", {"a": 1, "b": 2}))
        centre.step()
        assert "ghost" in centre.status(task).error

    def test_version_routing(self, specs):
        centre = CommandCentre(specs)
        v1 = FunctionSpecialist("v1", {"arith.add": lambda p: "v1"})
        v1.capabilities = lambda: Capabilities("v1", accepts=("arith.add@1",))
        centre.register(v1)
        centre.register(adder("v2"))
        t1 = centre.submit(specs.make("arith.add", {"a": 1, "b": 1}, version=2))
        centre.step()
        assert centre.status(t1).assignee == "v2"

    def test_priority_order(self, specs):
        centre = CommandCentre(specs)
        centre.register(adder())
        low = centre.submit(specs.make("arith.add", {"a": 1, "b": 1}))
        high = centre.submit(specs.make("arith.add", {"a": 1, "b": 1}, priority=5))
        centre.step()
        assert [t.task_id for t in centre.tasks("working")] == [low, high]
        assert centre.status(high).updated_clock == centre.status(low).updated_clock
        sent = [e for e in centre.bus._queue if is_goal(e)]
        assert [e.task_id for e in sent] == [high, low]

    def test_failure_is_reported(self, specs):
        centre = CommandCentre(specs)
        centre.register(FunctionSpecialist("bad", {"arith.add": lambda p: 1 / 0}))
        task = centre.submit(specs.make("arith.add", {"a": 1, "b": 1}))
        centre.run_until_idle()
        status = centre.status(task)
        assert status.state == "failed" and "ZeroDivisionError" in status.error

    def test_invalid_result_fails(self, specs):
        centre = CommandCentre(specs)
        centre.register(FunctionSpecialist("liar", {"arith.add": lambda p: "five"}))
        task = centre.submit(specs.make("arith.add", {"a": 2, "b": 3}))
        centre.run_until_idle()
        assert "invalid result" in centre.status(task).error

    def test_cancel(self, specs):
        centre = CommandCentre(specs)
        silent = Silent()
        centre.register(silent)
        task = centre.submit(specs.make("misc.wait", {}))
        centre.step()
        assert centre.cancel(task)
        assert centre.status(task).state == "canceled"
        assert not centre.cancel(task)
        centre.run_until_idle()
        assert len(silent.controls) == 1 and silent.controls[0].task_id == task

    def test_cancel_before_dispatch(self, specs):
        centre = CommandCentre(specs)
        centre.register(adder())
        task = centre.submit(specs.make("arith.add", {"a": 1, "b": 1}))
        assert centre.cancel(task)
        centre.run_until_idle()
        assert (
            centre.status(task).state == "canceled"
            and centre.status(task).assignee is None
        )

    def test_deadline(self, specs):
        centre = CommandCentre(specs)
        centre.register(Silent())
        task = centre.submit(specs.make("misc.wait", {}, deadline=3))
        for _ in range(3):
            centre.step()
        assert centre.status(task).state == "working"
        centre.step()
        status = centre.status(task)
        assert status.state == "failed" and "deadline" in status.error

    def test_late_result_is_stale(self, specs):
        trace = TraceLog()
        centre = CommandCentre(specs, trace=trace)
        centre.register(adder())
        task = centre.submit(specs.make("arith.add", {"a": 1, "b": 1}))
        centre.step()
        centre.step()  # the adder answers now
        centre.cancel(task)
        centre.run_until_idle()
        assert centre.status(task).state == "canceled"
        assert len(trace.events("stale_result")) == 1

    def test_result_from_wrong_sender_ignored(self, specs):
        trace = TraceLog()
        centre = CommandCentre(specs, trace=trace)
        centre.register(Scripted(lambda env: complete(env, "impostor", 1)))
        task = centre.submit(specs.make("misc.pair", {"a": 1, "b": 1}, deadline=5))
        centre.run_until_idle()
        assert centre.status(task).state == "failed"  # deadline, not the impostor
        assert trace.events("unexpected_sender")

    def test_unhandled_message_logged(self, specs):
        trace = TraceLog()
        centre = CommandCentre(specs, trace=trace)
        centre.bus.publish(envelope(to=centre.name))
        centre.step()
        assert len(trace.events("unhandled_message")) == 1

    def test_unregistered_assignee_fails_task(self, specs):
        centre = CommandCentre(specs)
        centre.register(Silent())
        task = centre.submit(specs.make("misc.wait", {}))
        centre.step()
        centre.registry.unregister("silent")
        centre.step()
        status = centre.status(task)
        assert status.state == "failed" and "unregistered" in status.error
        assert not centre.bus.is_subscribed("silent")

    def test_cannot_register_centre_name(self, specs):
        centre = CommandCentre(specs)
        with pytest.raises(ValueError):
            centre.register(adder(centre.name))

    def test_run_until_idle_respects_max_steps(self, specs):
        centre = CommandCentre(specs)
        centre.register(Silent())
        centre.submit(specs.make("misc.wait", {}))
        assert centre.run_until_idle(max_steps=4) == 4 and not centre.idle


class Delegator:
    """Answers misc.pair by asking the centre for arith.add, then doubling it."""

    def __init__(self, specs, spec="arith.add", params=None):
        self.specs = specs
        self.spec = spec
        self.params = params
        self.waiting = {}

    def capabilities(self):
        return Capabilities("delegator", accepts=("misc.pair",))

    def handle(self, env):
        if is_goal(env):
            params = goal_from_envelope(env).params
            sub_goal = GoalSpec(self.spec, self.params or params)
            sub = goal_envelope(
                sub_goal,
                "delegator",
                "command-centre",
                env.task_id,
                env.trace_id,
                env.msg_id,
            )
            self.waiting[sub.msg_id] = env
            return [sub]
        if env.parent_id in self.waiting:
            original = self.waiting.pop(env.parent_id)
            if env.payload["state"] != "completed":
                return [fail(original, "delegator", env.payload["error"])]
            return [complete(original, "delegator", env.payload["result"] * 2)]
        return []


class TestSubgoals:
    def test_delegation(self, specs):
        trace = TraceLog()
        centre = CommandCentre(specs, trace=trace)
        centre.register(adder())
        centre.register(Delegator(specs))
        task = centre.submit(specs.make("misc.pair", {"a": 2, "b": 5}))
        centre.run_until_idle()
        status = centre.status(task)
        assert status.state == "completed" and status.result == 14
        [sub_id] = status.subtasks
        sub = centre.status(sub_id)
        assert sub.parent_task_id == task and sub.requester == "delegator"
        assert sub.result == 7 and sub.trace_id == status.trace_id
        assert len(trace.trace_ids()) == 1

    def test_invalid_subgoal_is_refused(self, specs):
        centre = CommandCentre(specs)
        centre.register(Delegator(specs, spec="arith.add", params={"a": "x"}))
        task = centre.submit(specs.make("misc.pair", {"a": 2, "b": 5}))
        centre.run_until_idle()
        status = centre.status(task)
        assert status.state == "failed" and "invalid sub-goal" in status.error
        assert status.subtasks == ()

    def test_rejected_subgoal_reported_to_requester(self, specs):
        centre = CommandCentre(specs)
        centre.register(Delegator(specs, spec="arith.mul"))  # nobody multiplies
        task = centre.submit(specs.make("misc.pair", {"a": 2, "b": 5}))
        centre.run_until_idle()
        status = centre.status(task)
        assert status.state == "failed" and "no specialist" in status.error
        assert centre.status(status.subtasks[0]).state == "rejected"

    def test_cancel_cascades_to_subtasks(self, specs):
        centre = CommandCentre(specs)
        centre.register(Silent())
        centre.register(Delegator(specs, spec="misc.wait", params={}))
        task = centre.submit(specs.make("misc.pair", {"a": 1, "b": 1}))
        for _ in range(4):
            centre.step()
        [sub_id] = centre.status(task).subtasks
        assert centre.status(sub_id).state == "working"
        centre.cancel(task)
        assert centre.status(sub_id).state == "canceled"
        centre.run_until_idle()

    def test_depth_limit(self, specs):
        class Recursive(Delegator):
            def __init__(self, specs):
                super().__init__(specs, spec="misc.pair")

        centre = CommandCentre(specs, max_depth=2)
        centre.register(Recursive(specs))
        task = centre.submit(specs.make("misc.pair", {"a": 1, "b": 1}))
        centre.run_until_idle(max_steps=50)
        status = centre.status(task)
        assert status.state == "failed" and "too deep" in status.error


class TestSigning:
    KEYS = {"command-centre": b"c", "adder": b"a", "scripted": b"s"}

    def make_centre(self, specs, trace=None):
        centre = CommandCentre(
            specs,
            trace=trace,
            signing_key=self.KEYS["command-centre"],
            verify_keys=self.KEYS,
        )
        centre.register(
            ClockAdapter(
                adder(), signing_key=b"a", verify_keys={"command-centre": b"c"}
            )
        )
        return centre

    def test_signed_round_trip(self, specs):
        trace = TraceLog()
        centre = self.make_centre(specs, trace)
        task = centre.submit(specs.make("arith.add", {"a": 1, "b": 2}))
        centre.run_until_idle()
        assert centre.result(task) == 3
        assert all(e.is_signed for e in trace.envelopes())

    def test_unsigned_result_fails_task(self, specs):
        trace = TraceLog()
        centre = self.make_centre(specs, trace)
        centre.register(Scripted(lambda env: complete(env, "scripted", 1)))
        task = centre.submit(specs.make("misc.pair", {"a": 1, "b": 1}))
        centre.run_until_idle()
        status = centre.status(task)
        assert status.state == "failed" and "unsigned" in status.error
        assert trace.events("signature_rejected")[0].data["reason"] == "unsigned"

    def test_wrong_key_fails_task(self, specs):
        centre = self.make_centre(specs)
        centre.register(
            ClockAdapter(Scripted(lambda e: complete(e, "scripted", 1)), b"wrong")
        )
        task = centre.submit(specs.make("misc.pair", {"a": 1, "b": 1}))
        centre.run_until_idle()
        assert "bad signature" in centre.status(task).error
