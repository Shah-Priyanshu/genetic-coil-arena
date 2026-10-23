"""The command centre: accepts GoalSpecs, routes them to specialists, tracks tasks.

Lifecycle of a task (A2A states)::

    submitted --route--> working --result--> completed | failed
        |                   |
        +--> rejected       +--> canceled / failed (deadline, bad signature, ...)
       (no specialist)

The centre is driven by :meth:`CommandCentre.step`, one logical tick:

1. deliver every envelope published during the previous tick;
2. process the centre's inbox: task results, and sub-goals that specialists send
   to the centre (these become subtasks whose result is sent back to the requester);
3. fail tasks whose deadline has passed;
4. route and dispatch submitted tasks, highest priority first;
5. tick every specialist's ClockAdapter (in name order); their outputs are published
   and delivered next tick.

So a task with no sub-goals completes in three ticks. Every published envelope is
recorded in the trace, along with task state changes and rejected signatures.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union

from ..protocol.envelope import Envelope, new_id
from ..protocol.goalspec import GoalSpec, SpecRegistry, SpecValidationError
from ..protocol.messages import (
    CANCELED,
    COMPLETED,
    FAILED,
    REJECTED,
    SUBMITTED,
    TERMINAL_STATES,
    WORKING,
    complete,
    control_envelope,
    fail,
    goal_envelope,
    goal_from_envelope,
    is_goal,
    is_result,
)
from ..protocol.signing import Keys, check_signature, sign
from ..protocol.tensor_store import TensorStore
from .bus import InProcessBus, MessageBus
from .registry import SpecialistRecord, SpecialistRegistry
from .router import Router, RuleBasedRouter
from .specialist import ClockAdapter, Specialist
from .trace import TraceLog


@dataclass(frozen=True)
class TaskStatus:
    """Snapshot of a task.

    Attributes:
        task_id: Task id
        spec: GoalSpec name
        version: GoalSpec version
        goal_id: GoalSpec id
        state: One of the A2A task states
        assignee: Specialist working on it (None until routed)
        result: Result value once completed
        error: Error message once failed, rejected or canceled
        trace_id: Trace correlating all messages of the request
        parent_task_id: Task that requested this one as a sub-goal
        requester: Specialist that requested it (None for external submissions)
        subtasks: Ids of sub-goal tasks
        created_clock: Tick of submission
        updated_clock: Tick of the last state change
    """

    task_id: str
    spec: str
    version: int
    goal_id: str
    state: str
    assignee: Optional[str]
    result: Any
    error: Optional[str]
    trace_id: str
    parent_task_id: Optional[str]
    requester: Optional[str]
    subtasks: tuple[str, ...]
    created_clock: int
    updated_clock: int

    @property
    def done(self) -> bool:
        """Whether the task reached a terminal state."""
        return self.state in TERMINAL_STATES


@dataclass
class _Task:
    task_id: str
    goal: GoalSpec
    trace_id: str
    created_clock: int
    seq: int
    state: str = SUBMITTED
    assignee: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    updated_clock: int = 0
    goal_msg_id: Optional[str] = None
    requester: Optional[str] = None
    request: Optional[Envelope] = None
    parent_task_id: Optional[str] = None
    depth: int = 0
    subtasks: list[str] = field(default_factory=list)

    def snapshot(self) -> TaskStatus:
        return TaskStatus(
            task_id=self.task_id,
            spec=self.goal.spec,
            version=self.goal.version,
            goal_id=self.goal.goal_id,
            state=self.state,
            assignee=self.assignee,
            result=self.result,
            error=self.error,
            trace_id=self.trace_id,
            parent_task_id=self.parent_task_id,
            requester=self.requester,
            subtasks=tuple(self.subtasks),
            created_clock=self.created_clock,
            updated_clock=self.updated_clock,
        )


class CommandCentre:
    """Routes validated GoalSpecs to specialists over a bus and tracks the tasks.

    Args:
        specs: Registry every goal is validated against (submissions and sub-goals)
        registry: Specialist registry (a new one by default)
        router: Router (default: RuleBasedRouter with no rules)
        bus: Message bus (default: InProcessBus)
        trace: Trace log; every published envelope and task event is recorded
        name: The centre's bus address
        signing_key: If set, every envelope the centre sends is HMAC-signed
        verify_keys: If set, inbound envelopes must verify under these keys; a bad
            result fails its task
        store: Tensor store for re-hashing ``latent_ref`` payloads during verification
        max_depth: Maximum nesting of sub-goals (guards against delegation loops)
    """

    def __init__(
        self,
        specs: SpecRegistry,
        registry: Optional[SpecialistRegistry] = None,
        router: Optional[Router] = None,
        bus: Optional[MessageBus] = None,
        trace: Optional[TraceLog] = None,
        name: str = "command-centre",
        signing_key: Optional[bytes] = None,
        verify_keys: Optional[Keys] = None,
        store: Optional[TensorStore] = None,
        max_depth: int = 8,
    ):
        self.specs = specs
        self.registry = registry if registry is not None else SpecialistRegistry()
        self.router: Router = router if router is not None else RuleBasedRouter()
        self.bus: MessageBus = bus if bus is not None else InProcessBus()
        self.trace = trace
        self.name = name
        self.signing_key = signing_key
        self.verify_keys = verify_keys
        self.store = store
        self.max_depth = max_depth
        self.clock = 0
        self._tasks: dict[str, _Task] = {}
        self._by_goal_msg: dict[str, str] = {}
        self._inbox: list[Envelope] = []
        self._subscribed: dict[str, ClockAdapter] = {}
        self._seq = 0
        self.bus.subscribe(self.name, self._inbox.append)
        if trace is not None and hasattr(self.bus, "add_tap"):
            self.bus.add_tap(trace.record)

    # ------------------------------------------------------------------ public API

    def register(
        self,
        specialist: Union[Specialist, ClockAdapter],
        scores: Optional[Mapping[str, float]] = None,
        replace: bool = False,
    ) -> SpecialistRecord:
        """Shortcut for ``registry.register``; the bus subscription follows next step."""
        if isinstance(specialist, ClockAdapter):
            name = specialist.name
        else:
            name = specialist.capabilities().name
        if name == self.name:
            raise ValueError(f"{name!r} is the command centre's own address")
        return self.registry.register(specialist, scores=scores, replace=replace)

    def submit(
        self,
        spec: Union[GoalSpec, Mapping[str, Any]],
        situation: Any = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """Validate a goal and queue it as a new task.

        Args:
            spec: GoalSpec (or its dict form)
            situation: Optional JSON snapshot of the world, logged under the trace id
            trace_id: Trace to join (default: a new trace)

        Returns:
            The task id

        Raises:
            SpecValidationError: The spec does not validate
        """
        goal = self.specs.validate(spec)
        task = self._new_task(goal, trace_id or new_id())
        if situation is not None and self.trace is not None:
            self.trace.snapshot(task.trace_id, situation, clock=self.clock)
        return task.task_id

    def step(self) -> int:
        """Advance one tick (see module docstring).

        Returns:
            Number of envelope deliveries made at the start of the tick
        """
        self.clock += 1
        self._sync_subscriptions()
        delivered = self.bus.deliver()

        inbox = list(self._inbox)
        self._inbox.clear()  # In place: the bus handler is bound to this list
        for envelope in inbox:
            self._process(envelope)

        self._check_assignees()
        self._check_deadlines()
        self._dispatch()

        for name in sorted(self._subscribed):
            for envelope in self._subscribed[name].tick(self.clock):
                self.bus.publish(envelope)
        return delivered

    def run_until_idle(self, max_steps: int = 1000) -> int:
        """Step until :attr:`idle` (or ``max_steps``); returns the number of steps."""
        for steps in range(1, max_steps + 1):
            self.step()
            if self.idle:
                return steps
        return max_steps

    @property
    def idle(self) -> bool:
        """No open tasks and no messages in flight."""
        if any(task.state not in TERMINAL_STATES for task in self._tasks.values()):
            return False
        if self.bus.pending or self._inbox:
            return False
        return not any(adapter.pending for adapter in self._subscribed.values())

    def status(self, task_id: str) -> TaskStatus:
        """Snapshot of a task (KeyError if unknown)."""
        try:
            return self._tasks[task_id].snapshot()
        except KeyError:
            raise KeyError(f"Unknown task {task_id!r}") from None

    def result(self, task_id: str) -> Any:
        """Result of a completed task.

        Raises:
            KeyError: Unknown task
            RuntimeError: The task is not completed
        """
        status = self.status(task_id)
        if status.state != COMPLETED:
            raise RuntimeError(
                f"Task {task_id} is {status.state}: {status.error or 'not done'}"
            )
        return status.result

    def tasks(self, state: Optional[str] = None) -> list[TaskStatus]:
        """All task snapshots in submission order, optionally filtered by state."""
        ordered = sorted(self._tasks.values(), key=lambda t: t.seq)
        return [t.snapshot() for t in ordered if state is None or t.state == state]

    def cancel(self, task_id: str) -> bool:
        """Cancel a task and its open subtasks.

        Returns:
            False if the task had already finished, True otherwise

        Raises:
            KeyError: Unknown task
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task {task_id!r}")
        if task.state in TERMINAL_STATES:
            return False
        self._finish(task, CANCELED, error="canceled")
        return True

    # ------------------------------------------------------------------ internals

    def _new_task(
        self,
        goal: GoalSpec,
        trace_id: str,
        requester: Optional[str] = None,
        request: Optional[Envelope] = None,
        parent: Optional[_Task] = None,
    ) -> _Task:
        task = _Task(
            task_id=new_id(),
            goal=goal,
            trace_id=trace_id,
            created_clock=self.clock,
            updated_clock=self.clock,
            seq=self._seq,
            requester=requester,
            request=request,
            parent_task_id=parent.task_id if parent else None,
            depth=parent.depth + 1 if parent else 0,
        )
        self._seq += 1
        self._tasks[task.task_id] = task
        if parent is not None:
            parent.subtasks.append(task.task_id)
        self._event("task_state", task, state=SUBMITTED)
        return task

    def _send(self, envelope: Envelope) -> None:
        if self.signing_key is not None:
            envelope = sign(envelope, self.signing_key, self.name)
        self.bus.publish(envelope)

    def _event(self, name: str, task: Optional[_Task] = None, **details: Any) -> None:
        if self.trace is None:
            return
        if task is not None:
            details = {"task_id": task.task_id, "spec": task.goal.spec, **details}
        self.trace.event(name, task.trace_id if task else None, self.clock, **details)

    def _sync_subscriptions(self) -> None:
        current = {record.name: record.adapter for record in self.registry.records()}
        for name in list(self._subscribed):
            if current.get(name) is not self._subscribed[name]:
                self.bus.unsubscribe(name)
                del self._subscribed[name]
        for name, adapter in current.items():
            if name not in self._subscribed:
                self.bus.subscribe(name, adapter.receive)
                self._subscribed[name] = adapter

    def _process(self, envelope: Envelope) -> None:
        if self.verify_keys is not None:
            reason = check_signature(envelope, self.verify_keys, self.store)
            if reason is not None:
                task = self._task_for_reply(envelope)
                self._event(
                    "signature_rejected",
                    task,
                    sender=envelope.sender,
                    msg_id=envelope.msg_id,
                    reason=reason,
                )
                if task is not None and task.state == WORKING:
                    self._finish(
                        task, FAILED, error=f"signature check failed: {reason}"
                    )
                return

        if is_result(envelope):
            self._on_result(envelope)
        elif is_goal(envelope):
            self._on_subgoal(envelope)
        else:
            self._event(
                "unhandled_message",
                None,
                sender=envelope.sender,
                msg_id=envelope.msg_id,
            )

    def _task_for_reply(self, envelope: Envelope) -> Optional[_Task]:
        task_id = self._by_goal_msg.get(envelope.parent_id or "")
        return self._tasks.get(task_id) if task_id else None

    def _on_result(self, envelope: Envelope) -> None:
        task = self._task_for_reply(envelope)
        if task is None or task.state != WORKING:
            self._event(
                "stale_result", task, sender=envelope.sender, msg_id=envelope.msg_id
            )
            return
        if envelope.sender != task.assignee:
            self._event(
                "unexpected_sender",
                task,
                sender=envelope.sender,
                assignee=task.assignee,
            )
            return
        payload = envelope.payload
        if payload["state"] == FAILED:
            self._finish(task, FAILED, error=str(payload.get("error", "failed")))
            return
        result = payload.get("result")
        errors = self.specs.result_errors(task.goal, result)
        if errors:
            self._finish(task, FAILED, error="invalid result: " + "; ".join(errors))
        else:
            self._finish(task, COMPLETED, result=result)

    def _on_subgoal(self, envelope: Envelope) -> None:
        parent = self._tasks.get(envelope.task_id or "")
        try:
            goal = self.specs.validate(goal_from_envelope(envelope))
        except (SpecValidationError, ValueError) as exc:
            self._send(
                fail(envelope, self.name, f"invalid sub-goal: {exc}", clock=self.clock)
            )
            return
        if parent is not None and parent.depth + 1 > self.max_depth:
            self._send(
                fail(envelope, self.name, "sub-goal nesting too deep", clock=self.clock)
            )
            return
        if parent is not None and parent.state in TERMINAL_STATES:
            self._send(
                fail(
                    envelope, self.name, f"parent task {parent.state}", clock=self.clock
                )
            )
            return
        self._new_task(goal, envelope.trace_id, envelope.sender, envelope, parent)

    def _check_assignees(self) -> None:
        for task in self._open_tasks(WORKING):
            if task.assignee not in self.registry:
                self._finish(
                    task, FAILED, error=f"assignee {task.assignee!r} unregistered"
                )

    def _check_deadlines(self) -> None:
        for task in self._open_tasks():
            deadline = task.goal.deadline
            if deadline is not None and self.clock - task.created_clock > deadline:
                self._finish(
                    task, FAILED, error=f"deadline of {deadline} ticks exceeded"
                )

    def _dispatch(self) -> None:
        submitted = sorted(
            self._open_tasks(SUBMITTED), key=lambda t: (-t.goal.priority, t.seq)
        )
        for task in submitted:
            name = self.router.route(task.goal, self.registry)
            record = self.registry.get(name) if name is not None else None
            if record is None:
                why = (
                    "no specialist accepts"
                    if name is None
                    else f"router chose unknown {name!r} for"
                )
                self._finish(
                    task, REJECTED, error=f"{why} {task.goal.spec}@{task.goal.version}"
                )
                continue
            envelope = goal_envelope(
                task.goal,
                sender=self.name,
                recipients=record.name,
                task_id=task.task_id,
                trace_id=task.trace_id,
                parent_id=task.request.msg_id if task.request is not None else None,
                clock=self.clock,
            )
            task.assignee = record.name
            task.goal_msg_id = envelope.msg_id
            self._by_goal_msg[envelope.msg_id] = task.task_id
            self._set_state(task, WORKING, assignee=record.name)
            self._send(envelope)

    def _open_tasks(self, state: Optional[str] = None) -> list[_Task]:
        return [
            t
            for t in sorted(self._tasks.values(), key=lambda t: t.seq)
            if t.state not in TERMINAL_STATES and (state is None or t.state == state)
        ]

    def _set_state(self, task: _Task, state: str, **details: Any) -> None:
        task.state = state
        task.updated_clock = self.clock
        self._event("task_state", task, state=state, **details)

    def _finish(
        self, task: _Task, state: str, result: Any = None, error: Optional[str] = None
    ) -> None:
        was_working = task.state == WORKING
        task.result = result
        task.error = error
        self._set_state(task, state, **({"error": error} if error else {}))

        if was_working and state in (CANCELED, FAILED) and task.assignee is not None:
            # Tell the assignee to stop; its late answer will be ignored as stale
            self._send(
                control_envelope(
                    "cancel",
                    self.name,
                    task.assignee,
                    task.task_id,
                    task.trace_id,
                    parent_id=task.goal_msg_id,
                    clock=self.clock,
                )
            )
        for child_id in task.subtasks:
            child = self._tasks[child_id]
            if child.state not in TERMINAL_STATES:
                self._finish(child, CANCELED, error=f"parent task {state}")
        if task.request is not None and task.requester is not None:
            if state == COMPLETED:
                reply = complete(task.request, self.name, result, clock=self.clock)
            else:
                reply = fail(task.request, self.name, error or state, clock=self.clock)
            # The answer belongs to the subtask; the requester correlates by parent_id
            self._send(_with_task(reply, task.task_id))


def _with_task(envelope: Envelope, task_id: str) -> Envelope:
    return replace(envelope, task_id=task_id)
