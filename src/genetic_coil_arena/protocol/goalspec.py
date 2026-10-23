"""GoalSpecs: versioned, schema-validated descriptions of what a task should achieve.

Two layers of schema:

- a **spec definition** (:class:`SpecDefinition`) names a kind of goal, e.g.
  ``arith.add`` version 1, and carries a JSON Schema for its ``params`` (and,
  optionally, for the result a specialist must return). Definitions are validated
  against :data:`SPEC_DEFINITION_SCHEMA`.
- a **goal spec** (:class:`GoalSpec`) is one instance: which definition, which
  parameters, priority and deadline. It is validated against the versioned
  meta-schema :data:`GOALSPEC_SCHEMA` and then against its definition's params schema.

:class:`SpecRegistry` keys definitions by (name, version). It is the gate every
producer of goals (the command centre, the future natural-language translator) goes
through: :meth:`SpecRegistry.make` and :meth:`SpecRegistry.validate` only ever return
specs that validate.

Validation uses the ``jsonschema`` package (Draft 2020-12) when it is installed.
Otherwise a minimal built-in validator handles the subset of keywords listed in
:data:`MINIMAL_KEYWORDS`; schemas using other keywords are refused at registration
rather than silently half-checked.
"""

import copy
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .envelope import new_id

try:  # Optional dependency
    import jsonschema as _jsonschema
except ImportError:  # pragma: no cover - exercised only without jsonschema
    _jsonschema = None

GOALSPEC_META_VERSION = 1
GOALSPEC_CODEC = "goalspec"
HAS_JSONSCHEMA = _jsonschema is not None

# Dotted lower-case names such as "arith.add" or "nav.reach_cell"
SPEC_NAME_PATTERN = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$"

SPEC_DEFINITION_SCHEMA: dict[str, Any] = {
    "$id": f"urn:genetic-coil-arena:spec-definition:{GOALSPEC_META_VERSION}",
    "type": "object",
    "required": ["name", "version", "params_schema"],
    "properties": {
        "name": {"type": "string", "pattern": SPEC_NAME_PATTERN},
        "version": {"type": "integer", "minimum": 1},
        "description": {"type": "string"},
        "params_schema": {
            "type": "object",
            "required": ["type"],
            "properties": {"type": {"const": "object"}},
        },
        "result_schema": {"type": "object"},
    },
    "additionalProperties": False,
}

GOALSPEC_SCHEMA: dict[str, Any] = {
    "$id": f"urn:genetic-coil-arena:goalspec:{GOALSPEC_META_VERSION}",
    "type": "object",
    "required": ["meta_version", "spec", "version", "goal_id", "params"],
    "properties": {
        "meta_version": {"const": GOALSPEC_META_VERSION},
        "spec": {"type": "string", "pattern": SPEC_NAME_PATTERN},
        "version": {"type": "integer", "minimum": 1},
        "goal_id": {"type": "string", "minLength": 1},
        "params": {"type": "object"},
        "priority": {"type": "integer"},
        "deadline": {"type": ["integer", "null"], "minimum": 1},
        "constraints": {"type": "object"},
    },
    "additionalProperties": False,
}

# Keywords understood by the built-in validator (plus annotations, which are ignored)
MINIMAL_KEYWORDS = frozenset(
    {
        "type",
        "enum",
        "const",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minProperties",
        "maxProperties",
        "anyOf",
        "allOf",
        "oneOf",
        "not",
    }
)
ANNOTATION_KEYWORDS = frozenset(
    {"$schema", "$id", "$comment", "title", "description", "default", "examples"}
)
_JSON_TYPES = ("null", "boolean", "integer", "number", "string", "array", "object")


class SpecValidationError(ValueError):
    """A spec definition or goal spec failed validation.

    Attributes:
        errors: One readable message per problem, prefixed with the JSON path
    """

    def __init__(self, message: str, errors: Optional[list[str]] = None):
        self.errors = list(errors or [])
        detail = "; ".join(self.errors)
        super().__init__(f"{message}: {detail}" if detail else message)


# ---------------------------------------------------------------------- validators


def _is_type(value: Any, json_type: str) -> bool:
    if json_type == "null":
        return value is None
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (
            isinstance(value, float) and math.isfinite(value) and value.is_integer()
        )
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "array":
        return isinstance(value, (list, tuple))
    if json_type == "object":
        return isinstance(value, dict)
    return False


def _json_equal(a: Any, b: Any) -> bool:
    """JSON equality: booleans never equal numbers, lists and tuples are alike."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if _is_type(a, "number") and _is_type(b, "number"):
        return bool(a == b)
    return type(a) is type(b) and bool(a == b)


def _path(parts: tuple[Any, ...]) -> str:
    return "/" + "/".join(str(p) for p in parts) if parts else "<root>"


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _schema_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


# Keyword -> (what is expected, check) for MinimalValidator.check_schema
_KEYWORD_VALUE_CHECKS: dict[str, tuple[str, Callable[[Any], bool]]] = {
    **dict.fromkeys(
        ("minItems", "maxItems", "minLength", "maxLength")
        + ("minProperties", "maxProperties"),
        ("must be a non-negative integer", _count),
    ),
    **{
        k: ("must be a number", lambda v: _is_type(v, "number"))
        for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
    },
    **dict.fromkeys(
        ("anyOf", "allOf", "oneOf"), ("must be a non-empty list", _schema_list)
    ),
    "required": (
        "must be a list of strings",
        lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
    ),
    "enum": ("must be a list", lambda v: isinstance(v, list)),
    "properties": ("must be an object", lambda v: isinstance(v, dict)),
    "uniqueItems": ("must be a boolean", lambda v: isinstance(v, bool)),
    "pattern": ("must be a string", lambda v: isinstance(v, str)),
}


class MinimalValidator:
    """Small JSON Schema validator for the keywords in :data:`MINIMAL_KEYWORDS`.

    Used when ``jsonschema`` is not installed. Error messages differ from
    ``jsonschema``'s, but accept/reject decisions agree on the supported subset.
    """

    def __init__(self, schema: Mapping[str, Any]):
        self.check_schema(schema)
        self.schema = schema

    @classmethod
    def check_schema(cls, schema: Any, where: tuple[Any, ...] = ()) -> None:
        """Raise SpecValidationError if ``schema`` uses unsupported or malformed keywords."""
        if isinstance(schema, bool):
            return
        if not isinstance(schema, Mapping):
            raise SpecValidationError(
                "Invalid schema", [f"{_path(where)}: not an object"]
            )
        unknown = sorted(set(schema) - MINIMAL_KEYWORDS - ANNOTATION_KEYWORDS)
        if unknown:
            raise SpecValidationError(
                "Schema uses keywords the minimal validator does not support "
                "(install jsonschema)",
                [f"{_path(where)}: {name}" for name in unknown],
            )
        types = schema.get("type")
        if types is not None:
            listed = [types] if isinstance(types, str) else types
            if not isinstance(listed, list) or any(
                t not in _JSON_TYPES for t in listed
            ):
                raise SpecValidationError(
                    "Invalid schema", [f"{_path(where)}: bad type"]
                )
        problems = [
            f"{_path((*where, key))}: {expected}"
            for key, (expected, ok) in _KEYWORD_VALUE_CHECKS.items()
            if key in schema and not ok(schema[key])
        ]
        if problems:
            raise SpecValidationError("Invalid schema", problems)
        for name, sub in schema.get("properties", {}).items():
            cls.check_schema(sub, (*where, "properties", name))
        for key in ("items", "additionalProperties", "not"):
            if key in schema:
                cls.check_schema(schema[key], (*where, key))
        for key in ("anyOf", "allOf", "oneOf"):
            for i, sub in enumerate(schema.get(key, [])):
                cls.check_schema(sub, (*where, key, i))
        if "pattern" in schema:
            try:
                re.compile(schema["pattern"])
            except re.error as exc:
                raise SpecValidationError(
                    "Invalid schema", [f"{_path(where)}: bad pattern ({exc})"]
                ) from exc

    def iter_errors(self, instance: Any) -> Iterator[str]:
        """Yield one message per validation problem."""
        yield from self._errors(self.schema, instance, ())

    def _errors(self, schema: Any, value: Any, at: tuple[Any, ...]) -> Iterator[str]:
        where = _path(at)
        if schema is True:
            return
        if schema is False:
            yield f"{where}: no value is allowed here"
            return

        types = schema.get("type")
        if types is not None:
            listed = [types] if isinstance(types, str) else types
            if not any(_is_type(value, t) for t in listed):
                yield f"{where}: {value!r} is not of type {' or '.join(listed)}"
                return
        if "enum" in schema and not any(_json_equal(value, e) for e in schema["enum"]):
            yield f"{where}: {value!r} is not one of {schema['enum']!r}"
        if "const" in schema and not _json_equal(value, schema["const"]):
            yield f"{where}: {value!r} should be {schema['const']!r}"

        if _is_type(value, "number"):
            if "minimum" in schema and value < schema["minimum"]:
                yield f"{where}: {value!r} is less than the minimum {schema['minimum']!r}"
            if "maximum" in schema and value > schema["maximum"]:
                yield f"{where}: {value!r} is more than the maximum {schema['maximum']!r}"
            if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
                yield f"{where}: {value!r} must be above {schema['exclusiveMinimum']!r}"
            if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
                yield f"{where}: {value!r} must be below {schema['exclusiveMaximum']!r}"

        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                yield f"{where}: {value!r} is shorter than {schema['minLength']}"
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                yield f"{where}: {value!r} is longer than {schema['maxLength']}"
            if "pattern" in schema and not re.search(schema["pattern"], value):
                yield f"{where}: {value!r} does not match {schema['pattern']!r}"

        if _is_type(value, "array"):
            if "minItems" in schema and len(value) < schema["minItems"]:
                yield f"{where}: expected at least {schema['minItems']} items"
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                yield f"{where}: expected at most {schema['maxItems']} items"
            if schema.get("uniqueItems"):
                for i, item in enumerate(value):
                    if any(_json_equal(item, other) for other in value[:i]):
                        yield f"{where}: items are not unique"
                        break
            if "items" in schema:
                for i, item in enumerate(value):
                    yield from self._errors(schema["items"], item, (*at, i))

        if isinstance(value, dict):
            if "minProperties" in schema and len(value) < schema["minProperties"]:
                yield f"{where}: expected at least {schema['minProperties']} properties"
            if "maxProperties" in schema and len(value) > schema["maxProperties"]:
                yield f"{where}: expected at most {schema['maxProperties']} properties"
            for name in schema.get("required", []):
                if name not in value:
                    yield f"{where}: {name!r} is a required property"
            properties = schema.get("properties", {})
            for name, sub in properties.items():
                if name in value:
                    yield from self._errors(sub, value[name], (*at, name))
            extra = schema.get("additionalProperties", True)
            for name in value:
                if name not in properties:
                    if extra is False:
                        yield f"{where}: additional property {name!r} is not allowed"
                    elif extra is not True:
                        yield from self._errors(extra, value[name], (*at, name))

        for sub in schema.get("allOf", []):
            yield from self._errors(sub, value, at)
        if "anyOf" in schema and not any(
            not list(self._errors(sub, value, at)) for sub in schema["anyOf"]
        ):
            yield f"{where}: {value!r} matches none of anyOf"
        if "oneOf" in schema:
            matches = sum(
                not list(self._errors(sub, value, at)) for sub in schema["oneOf"]
            )
            if matches != 1:
                yield f"{where}: {value!r} matches {matches} of oneOf, expected exactly 1"
        if "not" in schema and not list(self._errors(schema["not"], value, at)):
            yield f"{where}: {value!r} must not match the 'not' schema"


def _under(prefix: str, error: str) -> str:
    """Re-root an error message produced for a sub-document."""
    if error.startswith("<root>"):
        return prefix + error[len("<root>") :]
    return prefix + error


SchemaCheck = Callable[[Any], list[str]]


def compile_schema(schema: Mapping[str, Any], backend: str = "auto") -> SchemaCheck:
    """Check ``schema`` itself and return a function listing an instance's errors.

    Args:
        schema: JSON Schema (Draft 2020-12 subset for the minimal backend)
        backend: ``"jsonschema"``, ``"minimal"`` or ``"auto"`` (jsonschema if installed)

    Raises:
        SpecValidationError: The schema is malformed (or unsupported by the backend)
    """
    backend = resolve_backend(backend)
    if backend == "jsonschema":
        validator_cls = _jsonschema.Draft202012Validator
        try:
            validator_cls.check_schema(schema)
        except _jsonschema.exceptions.SchemaError as exc:
            raise SpecValidationError("Invalid schema", [exc.message]) from exc
        validator = validator_cls(schema)

        def check(instance: Any) -> list[str]:
            errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
            return [f"{_path(tuple(e.path))}: {e.message}" for e in errors]

        return check

    minimal = MinimalValidator(schema)
    return lambda instance: list(minimal.iter_errors(instance))


def resolve_backend(backend: str) -> str:
    """Map ``"auto"`` to the best available backend and validate the name."""
    if backend == "auto":
        return "jsonschema" if HAS_JSONSCHEMA else "minimal"
    if backend == "jsonschema" and not HAS_JSONSCHEMA:
        raise ImportError("jsonschema is not installed; use backend='minimal'")
    if backend not in ("jsonschema", "minimal"):
        raise ValueError(f"Unknown schema backend {backend!r}")
    return backend


# ---------------------------------------------------------------------- data types


@dataclass(frozen=True)
class SpecDefinition:
    """A kind of goal, identified by (name, version).

    Attributes:
        name: Dotted lower-case name, e.g. ``arith.add``
        version: Positive integer; bump it whenever params or result change shape
        params_schema: JSON Schema (``type: object``) for GoalSpec.params
        description: Human-readable purpose (also feeds agent cards and the translator)
        result_schema: Optional JSON Schema the task result must satisfy
    """

    name: str
    version: int
    params_schema: dict[str, Any]
    description: str = ""
    result_schema: Optional[dict[str, Any]] = None

    @property
    def key(self) -> tuple[str, int]:
        """Registry key (name, version)."""
        return (self.name, self.version)

    def __hash__(self) -> int:
        return hash(self.key)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict (validated by SPEC_DEFINITION_SCHEMA)."""
        data: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "params_schema": copy.deepcopy(self.params_schema),
        }
        if self.result_schema is not None:
            data["result_schema"] = copy.deepcopy(self.result_schema)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpecDefinition":
        """Inverse of :meth:`to_dict` (not validated; register it to validate)."""
        return cls(
            name=data["name"],
            version=data["version"],
            params_schema=copy.deepcopy(dict(data["params_schema"])),
            description=data.get("description", ""),
            result_schema=copy.deepcopy(data.get("result_schema")),
        )


@dataclass(frozen=True)
class GoalSpec:
    """One goal: a spec definition plus parameters.

    Build these through :meth:`SpecRegistry.make` (or check them with
    :meth:`SpecRegistry.validate`); the constructor alone does not validate.

    Attributes:
        spec: Name of the spec definition
        params: Parameters, validated by the definition's params schema
        version: Version of the spec definition
        goal_id: Unique id of this goal
        priority: Higher runs first
        deadline: Clock ticks allowed after submission, or None for no limit
        constraints: Free-form, schema-checked by nothing beyond being an object
        meta_version: GoalSpec meta-schema version
    """

    spec: str
    params: dict[str, Any]
    version: int = 1
    goal_id: str = field(default_factory=new_id)
    priority: int = 0
    deadline: Optional[int] = None
    constraints: dict[str, Any] = field(default_factory=dict)
    meta_version: int = GOALSPEC_META_VERSION

    @property
    def key(self) -> tuple[str, int]:
        """Registry key (spec, version) of the definition this goal instantiates."""
        return (self.spec, self.version)

    def __hash__(self) -> int:
        return hash(self.goal_id)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict (validated by GOALSPEC_SCHEMA)."""
        return {
            "meta_version": self.meta_version,
            "spec": self.spec,
            "version": self.version,
            "goal_id": self.goal_id,
            "params": copy.deepcopy(self.params),
            "priority": self.priority,
            "deadline": self.deadline,
            "constraints": copy.deepcopy(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalSpec":
        """Build from a dict (missing optional fields take defaults; not validated)."""
        missing = [k for k in ("spec", "params") if k not in data]
        if missing:
            raise SpecValidationError(
                "GoalSpec dict is incomplete", [f"missing {missing}"]
            )
        return cls(
            spec=data["spec"],
            params=copy.deepcopy(data["params"]),
            version=data.get("version", 1),
            goal_id=data.get("goal_id") or new_id(),
            priority=data.get("priority", 0),
            deadline=data.get("deadline"),
            constraints=copy.deepcopy(data.get("constraints") or {}),
            meta_version=data.get("meta_version", GOALSPEC_META_VERSION),
        )


# ---------------------------------------------------------------------- registry


class SpecRegistry:
    """Spec definitions keyed by (name, version), with validation.

    Args:
        backend: Schema backend, ``"auto"``, ``"jsonschema"`` or ``"minimal"``
    """

    def __init__(self, backend: str = "auto"):
        self.backend = resolve_backend(backend)
        self._definition_check = compile_schema(SPEC_DEFINITION_SCHEMA, self.backend)
        self._goal_check = compile_schema(GOALSPEC_SCHEMA, self.backend)
        self._definitions: dict[tuple[str, int], SpecDefinition] = {}
        self._params_checks: dict[tuple[str, int], SchemaCheck] = {}
        self._result_checks: dict[tuple[str, int], Optional[SchemaCheck]] = {}

    # -- definitions

    def register(self, definition: SpecDefinition) -> SpecDefinition:
        """Validate and add a definition.

        Re-registering an identical definition is a no-op; a different definition
        under an existing (name, version) is an error (bump the version instead).

        Raises:
            SpecValidationError: Invalid definition or conflicting (name, version)
        """
        errors = self._definition_check(definition.to_dict())
        if errors:
            raise SpecValidationError(
                f"Invalid spec definition {definition.key}", errors
            )
        existing = self._definitions.get(definition.key)
        if existing is not None:
            if existing.to_dict() == definition.to_dict():
                return existing
            raise SpecValidationError(
                f"Spec {definition.name!r} version {definition.version} is already "
                "registered with a different definition; bump the version"
            )
        params_check = compile_schema(definition.params_schema, self.backend)
        result_check = (
            compile_schema(definition.result_schema, self.backend)
            if definition.result_schema is not None
            else None
        )
        self._definitions[definition.key] = definition
        self._params_checks[definition.key] = params_check
        self._result_checks[definition.key] = result_check
        return definition

    def unregister(self, name: str, version: int) -> None:
        """Remove one definition (KeyError if absent)."""
        key = (name, version)
        del self._definitions[key]
        del self._params_checks[key]
        del self._result_checks[key]

    def get(self, name: str, version: Optional[int] = None) -> SpecDefinition:
        """Definition by name and version (latest version when ``version`` is None).

        Raises:
            KeyError: No such definition
        """
        if version is None:
            versions = self.versions(name)
            if not versions:
                raise KeyError(f"No spec named {name!r}")
            version = versions[-1]
        try:
            return self._definitions[(name, version)]
        except KeyError:
            raise KeyError(f"No spec {name!r} version {version}") from None

    def versions(self, name: str) -> list[int]:
        """Registered versions of ``name``, ascending."""
        return sorted(v for n, v in self._definitions if n == name)

    def names(self) -> list[str]:
        """Registered spec names, sorted."""
        return sorted({n for n, _ in self._definitions})

    def definitions(self) -> list[SpecDefinition]:
        """All definitions, sorted by (name, version)."""
        return [self._definitions[k] for k in sorted(self._definitions)]

    def __contains__(self, key: object) -> bool:
        return key in self._definitions

    def __len__(self) -> int:
        return len(self._definitions)

    # -- goals

    def errors(self, spec: Any) -> list[str]:
        """All validation problems of a GoalSpec (or its dict form); empty if valid."""
        if isinstance(spec, GoalSpec):
            data = spec.to_dict()
        elif isinstance(spec, Mapping):
            data = dict(spec)
        else:
            return [f"<root>: expected a GoalSpec or dict, got {type(spec).__name__}"]
        errors = self._goal_check(data)
        if errors:
            return errors
        key = (data["spec"], int(data["version"]))
        if key not in self._definitions:
            known = self.versions(data["spec"])
            hint = f" (known versions: {known})" if known else ""
            return [f"/spec: unknown spec {key[0]!r} version {key[1]}{hint}"]
        return [_under("/params", e) for e in self._params_checks[key](data["params"])]

    def is_valid(self, spec: Any) -> bool:
        """True if :meth:`errors` finds nothing."""
        return not self.errors(spec)

    def validate(self, spec: Any) -> GoalSpec:
        """Return ``spec`` as a GoalSpec if it validates.

        Raises:
            SpecValidationError: With every problem found
        """
        errors = self.errors(spec)
        if errors:
            name = spec.spec if isinstance(spec, GoalSpec) else None
            if isinstance(spec, Mapping):
                name = spec.get("spec")
            raise SpecValidationError(f"Invalid GoalSpec {name!r}", errors)
        return spec if isinstance(spec, GoalSpec) else GoalSpec.from_dict(spec)

    def make(
        self,
        name: str,
        params: Mapping[str, Any],
        version: Optional[int] = None,
        priority: int = 0,
        deadline: Optional[int] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        goal_id: Optional[str] = None,
    ) -> GoalSpec:
        """Build a validated GoalSpec (latest definition version by default).

        Raises:
            KeyError: Unknown spec name/version
            SpecValidationError: Parameters or options do not validate
        """
        definition = self.get(name, version)
        spec = GoalSpec(
            spec=name,
            params=copy.deepcopy(dict(params)),
            version=definition.version,
            goal_id=goal_id or new_id(),
            priority=priority,
            deadline=deadline,
            constraints=copy.deepcopy(dict(constraints or {})),
        )
        return self.validate(spec)

    def result_errors(self, spec: GoalSpec, result: Any) -> list[str]:
        """Problems of ``result`` against the spec's result schema (empty if none)."""
        if spec.key not in self._definitions:
            return [f"/spec: unknown spec {spec.spec!r} version {spec.version}"]
        check = self._result_checks[spec.key]
        return check(result) if check is not None else []
