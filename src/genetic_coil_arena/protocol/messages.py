"""Standard JSON messages of the task lifecycle: goals, results and control.

Each kind is a ``json`` envelope marked by ``payload_encoding["codec"]``:

- ``goalspec``: payload is :meth:`GoalSpec.to_dict`. Sent by the command centre to a
  specialist, or by a specialist to the centre to request a sub-goal.
- ``task_result``: payload ``{"state": "completed", "result": ...}`` or
  ``{"state": "failed", "error": "..."}``, answering a goal (``parent_id`` is the
  goal envelope's ``msg_id``).
- ``control``: payload ``{"action": "cancel"}``, telling a specialist to stop a task.

States follow the A2A task model (submitted, working, completed, failed, canceled,
rejected).
"""

from collections.abc import Sequence
from typing import Any, Optional, Union

from .envelope import Envelope, as_recipients
from .goalspec import GOALSPEC_CODEC, GOALSPEC_META_VERSION, GoalSpec

MESSAGES_VERSION = 1
RESULT_CODEC = "task_result"
CONTROL_CODEC = "control"

# A2A-style task states
SUBMITTED = "submitted"
WORKING = "working"
COMPLETED = "completed"
FAILED = "failed"
CANCELED = "canceled"
REJECTED = "rejected"
TASK_STATES = (SUBMITTED, WORKING, COMPLETED, FAILED, CANCELED, REJECTED)
TERMINAL_STATES = frozenset({COMPLETED, FAILED, CANCELED, REJECTED})


def codec_of(envelope: Envelope) -> Optional[str]:
    """``payload_encoding["codec"]`` of an envelope, or None."""
    codec = envelope.payload_encoding.get("codec")
    return codec if isinstance(codec, str) else None


def goal_envelope(
    spec: GoalSpec,
    sender: str,
    recipients: Union[str, Sequence[str]],
    task_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    clock: int = 0,
) -> Envelope:
    """Wrap a GoalSpec in a ``goalspec`` envelope."""
    extra: dict[str, Any] = {} if trace_id is None else {"trace_id": trace_id}
    return Envelope(
        sender=sender,
        recipients=as_recipients(recipients),
        payload_type="json",
        payload=spec.to_dict(),
        payload_encoding={
            "codec": GOALSPEC_CODEC,
            "meta_version": GOALSPEC_META_VERSION,
        },
        task_id=task_id,
        parent_id=parent_id,
        clock=clock,
        **extra,
    )


def is_goal(envelope: Envelope) -> bool:
    """Whether the envelope carries a GoalSpec."""
    return envelope.payload_type == "json" and codec_of(envelope) == GOALSPEC_CODEC


def goal_from_envelope(envelope: Envelope) -> GoalSpec:
    """Extract the GoalSpec (structure only; validate it with a SpecRegistry)."""
    if not is_goal(envelope):
        raise ValueError(f"Envelope {envelope.msg_id} does not carry a GoalSpec")
    if not isinstance(envelope.payload, dict):
        raise ValueError(f"Envelope {envelope.msg_id} has a malformed GoalSpec payload")
    return GoalSpec.from_dict(envelope.payload)


def _result(
    request: Envelope, sender: str, payload: dict[str, Any], clock: Optional[int]
) -> Envelope:
    return request.reply(
        sender,
        payload=payload,
        payload_encoding={"codec": RESULT_CODEC, "codec_version": MESSAGES_VERSION},
        clock=clock,
    )


def complete(
    request: Envelope, sender: str, result: Any, clock: Optional[int] = None
) -> Envelope:
    """Reply to a goal envelope reporting success with ``result`` (any JSON value)."""
    return _result(request, sender, {"state": COMPLETED, "result": result}, clock)


def fail(
    request: Envelope, sender: str, error: str, clock: Optional[int] = None
) -> Envelope:
    """Reply to a goal envelope reporting failure."""
    return _result(request, sender, {"state": FAILED, "error": str(error)}, clock)


def is_result(envelope: Envelope) -> bool:
    """Whether the envelope is a task result."""
    return (
        envelope.payload_type == "json"
        and codec_of(envelope) == RESULT_CODEC
        and isinstance(envelope.payload, dict)
        and envelope.payload.get("state") in (COMPLETED, FAILED)
    )


def control_envelope(
    action: str,
    sender: str,
    recipients: Union[str, Sequence[str]],
    task_id: Optional[str],
    trace_id: str,
    parent_id: Optional[str] = None,
    clock: int = 0,
) -> Envelope:
    """A control message, e.g. ``action="cancel"`` for a task."""
    return Envelope(
        sender=sender,
        recipients=as_recipients(recipients),
        payload_type="json",
        payload={"action": action},
        payload_encoding={"codec": CONTROL_CODEC, "codec_version": MESSAGES_VERSION},
        task_id=task_id,
        trace_id=trace_id,
        parent_id=parent_id,
        clock=clock,
    )


def is_control(envelope: Envelope, action: Optional[str] = None) -> bool:
    """Whether the envelope is a control message (with ``action``, if given)."""
    if envelope.payload_type != "json" or codec_of(envelope) != CONTROL_CODEC:
        return False
    if not isinstance(envelope.payload, dict):
        return False
    return action is None or envelope.payload.get("action") == action
