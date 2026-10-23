"""Test GoalSpecs, the spec registry and both schema backends."""

import pytest

from genetic_coil_arena.protocol import (
    GOALSPEC_META_VERSION,
    HAS_JSONSCHEMA,
    GoalSpec,
    MinimalValidator,
    SpecDefinition,
    SpecRegistry,
    SpecValidationError,
    compile_schema,
)
from genetic_coil_arena.protocol.goalspec import resolve_backend

BACKENDS = ["minimal"] + (["jsonschema"] if HAS_JSONSCHEMA else [])

MOVE = SpecDefinition(
    name="nav.move",
    version=1,
    params_schema={
        "type": "object",
        "required": ["dx", "dy"],
        "properties": {
            "dx": {"type": "integer", "minimum": -1, "maximum": 1},
            "dy": {"type": "integer", "minimum": -1, "maximum": 1},
            "mode": {"enum": ["walk", "run"]},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        },
        "additionalProperties": False,
    },
    description="Move one cell",
    result_schema={"type": "object", "required": ["ok"]},
)


@pytest.fixture(params=BACKENDS)
def registry(request):
    specs = SpecRegistry(backend=request.param)
    specs.register(MOVE)
    return specs


class TestRegistry:
    def test_get_latest_and_versions(self, registry):
        v2 = SpecDefinition("nav.move", 2, {"type": "object"})
        registry.register(v2)
        assert registry.versions("nav.move") == [1, 2]
        assert registry.get("nav.move") is v2
        assert registry.get("nav.move", 1) is MOVE
        assert ("nav.move", 1) in registry and len(registry) == 2
        assert registry.names() == ["nav.move"]
        assert [d.key for d in registry.definitions()] == [
            ("nav.move", 1),
            ("nav.move", 2),
        ]

    def test_unknown(self, registry):
        with pytest.raises(KeyError):
            registry.get("nope")
        with pytest.raises(KeyError):
            registry.get("nav.move", 9)

    def test_reregister_identical_is_noop_conflict_is_error(self, registry):
        assert registry.register(SpecDefinition.from_dict(MOVE.to_dict())) is MOVE
        with pytest.raises(SpecValidationError, match="bump the version"):
            registry.register(SpecDefinition("nav.move", 1, {"type": "object"}))

    def test_unregister(self, registry):
        registry.unregister("nav.move", 1)
        assert len(registry) == 0
        with pytest.raises(KeyError):
            registry.unregister("nav.move", 1)

    @pytest.mark.parametrize(
        "definition",
        [
            SpecDefinition("Bad-Name", 1, {"type": "object"}),
            SpecDefinition("ok", 0, {"type": "object"}),
            SpecDefinition("ok", 1, {"type": "array"}),
            SpecDefinition("ok", 1, {"properties": {}}),
            SpecDefinition("ok", True, {"type": "object"}),
        ],
    )
    def test_invalid_definitions(self, registry, definition):
        with pytest.raises(SpecValidationError):
            registry.register(definition)

    def test_malformed_params_schema(self, registry):
        with pytest.raises(SpecValidationError):
            registry.register(
                SpecDefinition("ok", 1, {"type": "object", "minItems": "x"})
            )

    def test_unknown_backend(self):
        with pytest.raises(ValueError):
            SpecRegistry(backend="xml")
        assert resolve_backend("auto") in ("minimal", "jsonschema")


class TestGoalValidation:
    def test_make_returns_valid_spec(self, registry):
        goal = registry.make("nav.move", {"dx": 1, "dy": 0}, priority=2, deadline=5)
        assert isinstance(goal, GoalSpec)
        assert goal.version == 1 and goal.priority == 2 and goal.deadline == 5
        assert goal.meta_version == GOALSPEC_META_VERSION
        assert registry.is_valid(goal) and registry.errors(goal.to_dict()) == []

    def test_make_copies_params(self, registry):
        params = {"dx": 1, "dy": 0, "tags": ["a"]}
        goal = registry.make("nav.move", params)
        params["tags"].append("b")
        assert goal.params["tags"] == ["a"]

    @pytest.mark.parametrize(
        "params, fragment",
        [
            ({"dx": 1}, "dy"),
            ({"dx": 2, "dy": 0}, "/params/dx"),
            ({"dx": 1.5, "dy": 0}, "/params/dx"),
            ({"dx": True, "dy": 0}, "/params/dx"),
            ({"dx": 0, "dy": 0, "mode": "fly"}, "/params/mode"),
            ({"dx": 0, "dy": 0, "tags": ["a", "b", "c"]}, "/params/tags"),
            ({"dx": 0, "dy": 0, "tags": [1]}, "/params/tags/0"),
            ({"dx": 0, "dy": 0, "speed": 3}, "/params"),
        ],
    )
    def test_make_rejects_invalid_params(self, registry, params, fragment):
        with pytest.raises(SpecValidationError) as info:
            registry.make("nav.move", params)
        assert any(fragment in e for e in info.value.errors)

    def test_integral_float_is_an_integer(self, registry):
        assert registry.is_valid(registry.make("nav.move", {"dx": 1.0, "dy": 0}))

    @pytest.mark.parametrize(
        "changes",
        [
            {"meta_version": 2},
            {"version": 0},
            {"goal_id": ""},
            {"priority": 1.5},
            {"deadline": 0},
            {"constraints": []},
            {"surprise": 1},
            {"spec": "Nav"},
        ],
    )
    def test_meta_schema(self, registry, changes):
        data = {**registry.make("nav.move", {"dx": 0, "dy": 0}).to_dict(), **changes}
        assert registry.errors(data)
        with pytest.raises(SpecValidationError):
            registry.validate(data)

    def test_unknown_spec_or_version(self, registry):
        assert "unknown spec" in registry.errors(GoalSpec("nav.fly", {}))[0]
        errors = registry.errors(GoalSpec("nav.move", {"dx": 0, "dy": 0}, version=3))
        assert "known versions: [1]" in errors[0]

    def test_non_mapping(self, registry):
        assert registry.errors(42)

    def test_validate_dict_returns_goalspec(self, registry):
        data = {
            "spec": "nav.move",
            "version": 1,
            "goal_id": "g1",
            "meta_version": 1,
            "params": {"dx": 0, "dy": 1},
        }
        goal = registry.validate(data)
        assert isinstance(goal, GoalSpec) and goal.goal_id == "g1"

    def test_goalspec_dict_round_trip(self, registry):
        goal = registry.make("nav.move", {"dx": 0, "dy": 1}, constraints={"budget": 3})
        assert GoalSpec.from_dict(goal.to_dict()) == goal

    def test_from_dict_requires_spec_and_params(self):
        with pytest.raises(SpecValidationError):
            GoalSpec.from_dict({"spec": "x"})

    def test_result_errors(self, registry):
        goal = registry.make("nav.move", {"dx": 0, "dy": 0})
        assert registry.result_errors(goal, {"ok": True}) == []
        assert registry.result_errors(goal, {"nope": 1})
        registry.register(SpecDefinition("free", 1, {"type": "object"}))
        assert registry.result_errors(registry.make("free", {}), "anything") == []
        assert registry.result_errors(GoalSpec("ghost", {}), 1)


class TestMinimalValidator:
    def test_refuses_unsupported_keywords(self):
        with pytest.raises(SpecValidationError, match="does not support"):
            MinimalValidator({"type": "object", "$ref": "#/x"})
        with pytest.raises(SpecValidationError):
            MinimalValidator({"properties": {"a": {"format": "email"}}})

    def test_annotations_allowed(self):
        MinimalValidator(
            {"title": "t", "description": "d", "default": 1, "examples": []}
        )

    @pytest.mark.parametrize(
        "schema, good, bad",
        [
            ({"type": ["integer", "null"]}, [None, 3], ["x", 1.5]),
            ({"const": 1}, [1, 1.0], [True, "1"]),
            ({"enum": [False, "a"]}, [False, "a"], [0, "b"]),
            (
                {"type": "string", "minLength": 2, "maxLength": 3, "pattern": "^a"},
                ["ab", "abc"],
                ["a", "abcd", "ba"],
            ),
            (
                {"exclusiveMinimum": 0, "exclusiveMaximum": 1},
                [0.5, "not a number"],
                [0, 1],
            ),
            (
                {"type": "array", "uniqueItems": True, "minItems": 1},
                [[1, 2]],
                [[], [1, 1]],
            ),
            (
                {"type": "object", "minProperties": 1, "maxProperties": 1},
                [{"a": 1}],
                [{}, {"a": 1, "b": 2}],
            ),
            ({"additionalProperties": {"type": "integer"}}, [{"a": 1}], [{"a": "x"}]),
            ({"anyOf": [{"type": "string"}, {"minimum": 5}]}, ["x", 7], [3]),
            ({"oneOf": [{"type": "integer"}, {"minimum": 0}]}, [-1, 0.5], [3]),
            ({"allOf": [{"minimum": 0}, {"maximum": 2}]}, [1], [3]),
            ({"not": {"type": "null"}}, [0], [None]),
            ({"items": False}, [[]], [[1]]),
        ],
    )
    def test_keywords(self, schema, good, bad):
        for backend in BACKENDS:
            check = compile_schema(schema, backend)
            for value in good:
                assert check(value) == [], (backend, schema, value)
            for value in bad:
                assert check(value), (backend, schema, value)


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_backends_agree_on_meta_schema_decisions():
    minimal, full = SpecRegistry("minimal"), SpecRegistry("jsonschema")
    for specs in (minimal, full):
        specs.register(MOVE)
    candidates = [
        {
            "spec": "nav.move",
            "version": 1,
            "goal_id": "g",
            "meta_version": 1,
            "params": p,
        }
        for p in (
            {"dx": 0, "dy": 0},
            {"dx": 3, "dy": 0},
            {"dy": 0},
            {"dx": 0, "dy": 0, "z": 1},
        )
    ]
    for data in candidates:
        assert bool(minimal.errors(data)) == bool(full.errors(data))
