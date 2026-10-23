"""The trained squad protocol on the object layer: envelopes, command centre, trace.

The R6 policies (``learned_protocol``) are trained and scored on the batched numpy
fast path, where a centre message is one int in a (B, N, 1) array. This module runs
the same policies as ``runtime`` specialists, so every centre-to-specialist message is
a signed ``codes`` envelope (made by ``protocol.codecs``, V = 8, L = 1) that crosses
the ``InProcessBus`` and is verified on receipt, and one squad episode is one task:
a validated ``squad.forage`` GoalSpec submitted to the ``CommandCentre``.

Parties (bus addresses):

- ``command-centre``: the runtime ``CommandCentre``; routes ``squad.forage`` goals.
- ``squad-lead``: the squad's (learned, hand-coded or silent) centre policy. On a goal
  it starts the episode; every tick it reads the full-view observation, computes one
  code per live snake and sends it as an envelope; at the end it answers the goal
  with the team's score.
- ``squad-snake-<k>``: one specialist per snake, all sharing the trained specialist
  network. A snake decodes the verified envelope it received (``envelopes_to_codes``)
  and acts on its egocentric window, its health and that code.
- :class:`SquadWorld`: the simulator (``SquadEnv`` with B = 1). It is the snakes'
  sensors and actuators, not a party: it never carries messages. It steps once every
  live snake has submitted an action.

Timing matches the fast path: codes sent at tick t are delivered at tick t + 1 (one
bus round), which is when the environment would show them (``message`` is read one
step later). Episode steps advance one per ``CommandCentre.step``.

Determinism: the fast path draws the specialists' Gumbel action noise for all N
snakes from one generator per episode (``LearnedSpecialist.reset``). To compare step
for step, the world draws exactly that noise each step (common random numbers) and
hands snake k its row; with the same seed the object layer and the B = 1 fast path
therefore produce the same codes, actions and score (see ``compare_episode``).

Signature failures: a snake's ``ClockAdapter`` drops an envelope whose manifest does
not verify. The snake then behaves as if that message had not been sent: it keeps
the last verified code of the current task (0 before any), which is the
environment's own convention for the ``message`` observation ("the last payload
received"). :func:`fast_path_episode` can replay the same fallback with a message
hook, so a tampered object-layer episode is still reproducible on the fast path.

Trace (``runtime.TraceLog``): every envelope (goal, codes, result), the task state
events, one ``situation`` snapshot per episode step (compact: heads, headings, food,
health, alive, score; ``clock`` equals the clock of that step's code envelopes) and a
``signature_rejected`` event per dropped envelope. These are the (situation, spec,
message) triples the reverse translator will train on.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from ..envs.base import Observations
from ..envs.squad import SquadConfig, SquadEnv, run_episodes
from ..envs.squad.policies import CentrePolicy
from ..learning import MLP
from ..protocol.codecs import CODES_CODEC, codes_to_envelopes, envelopes_to_codes
from ..protocol.envelope import Envelope
from ..protocol.goalspec import GoalSpec, SpecDefinition, SpecRegistry
from ..protocol.messages import (
    codec_of,
    complete,
    fail,
    goal_from_envelope,
    is_control,
    is_goal,
)
from ..runtime import (
    Capabilities,
    ClockAdapter,
    CommandCentre,
    InProcessBus,
    TraceLog,
)
from .learned_protocol import LearnedSpecialist, specialist_inputs

SQUAD_FORAGE = "squad.forage"
CENTRE = "command-centre"
LEAD = "squad-lead"
SNAKE_PREFIX = "squad-snake-"
MAX_EPISODE_STEPS = 1000
FALLBACK = "hold-last-verified-code"

# Demo-only secrets; real deployments load keys from a secret store
DEFAULT_KEYS = {CENTRE: b"squad-centre-key", LEAD: b"squad-lead-key"}


def snake_names(n_snakes: int) -> list[str]:
    """Default bus addresses of the snake specialists."""
    return [f"{SNAKE_PREFIX}{k}" for k in range(n_snakes)]


# ------------------------------------------------------------------ GoalSpec
def squad_spec_definition(n_snakes: int = 3) -> SpecDefinition:
    """``squad.forage`` v1: play one cooperative-foraging episode.

    Params:
        episode_seed: Seed of the episode (board layout, food respawns, action noise)
        max_steps: Episode length cap (default: the squad configuration's)
        assignment: Specialist name per colour, i.e. which specialist plays snake c
            (default: ``squad-snake-0`` .. ``squad-snake-<N-1>`` in colour order)

    A small placeholder that the natural-language translator will target later.
    """
    params = {
        "type": "object",
        "required": ["episode_seed"],
        "properties": {
            "episode_seed": {"type": "integer", "minimum": 0},
            "max_steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_EPISODE_STEPS,
            },
            "assignment": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": n_snakes,
                "maxItems": n_snakes,
                "uniqueItems": True,
            },
        },
        "additionalProperties": False,
    }
    count = {"type": "integer", "minimum": 0}
    result = {
        "type": "object",
        "required": ["team_food", "food", "steps", "survivors", "return"],
        "properties": {
            "team_food": count,
            "food": {
                "type": "array",
                "items": count,
                "minItems": n_snakes,
                "maxItems": n_snakes,
            },
            "steps": count,
            "survivors": count,
            "return": {"type": "number"},
        },
    }
    return SpecDefinition(
        SQUAD_FORAGE,
        1,
        params,
        "Play one cooperative-foraging squad episode and report the team's food",
        result,
    )


def squad_specs(n_snakes: int = 3, backend: str = "auto") -> SpecRegistry:
    """A SpecRegistry holding ``squad.forage``."""
    specs = SpecRegistry(backend=backend)
    specs.register(squad_spec_definition(n_snakes))
    return specs


# ------------------------------------------------------------------ the world
class SquadWorld:
    """One squad episode at a time (``SquadEnv`` with B = 1): sensors and actuators.

    Args:
        config: Squad configuration (``max_steps`` is the default episode length)
        noise_salt: Salt of the action-noise generator, as ``LearnedSpecialist.salt``
    """

    def __init__(self, config: SquadConfig, noise_salt: int = LearnedSpecialist.salt):
        self.config = config
        self.noise_salt = noise_salt
        self.n = config.n_snakes
        self.env: Optional[SquadEnv] = None
        self.task_id: Optional[str] = None
        self.trace_id: Optional[str] = None
        self.seed: Optional[int] = None
        self.slots: dict[str, int] = {}
        self.obs: Optional[Observations] = None
        self.step_index = 0
        self.total = 0.0
        self._gen: Optional[np.random.Generator] = None
        self._noise = np.zeros((self.n, 3))
        self._actions = np.zeros(self.n, dtype=np.int64)
        self._submitted = np.zeros(self.n, dtype=bool)
        self.history: dict[str, list[np.ndarray]] = {}

    # -- lifecycle
    def start(
        self,
        seed: int,
        assignment: Sequence[str],
        task_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """Reset to a new episode."""
        if len(assignment) != self.n:
            raise ValueError(f"Need {self.n} specialists, got {len(assignment)}")
        config = self.config
        if max_steps is not None and max_steps != config.max_steps:
            config = replace(config, max_steps=int(max_steps))
        self.env = SquadEnv(config)
        self.seed = int(seed)
        self.obs = self.env.reset([self.seed])
        self.task_id, self.trace_id = task_id, trace_id
        self.slots = {name: c for c, name in enumerate(assignment)}
        self.step_index = 0
        self.total = 0.0
        self._gen = np.random.default_rng([self.seed, self.noise_salt])
        self.history = {"codes": [], "heard": [], "actions": [], "alive": []}
        self._new_step()

    def stop(self) -> None:
        """Abandon the episode (cancel)."""
        self.env = None
        self.task_id = None

    @property
    def active(self) -> bool:
        """An episode is running (possibly finished but not yet reported)."""
        return self.env is not None

    @property
    def done(self) -> bool:
        return self.env is not None and bool(self.env.done[0])

    def alive(self, slot: int) -> bool:
        assert self.env is not None
        return bool(self.env.alive[0, slot])

    def slot_of(self, name: str) -> Optional[int]:
        return self.slots.get(name) if self.active else None

    # -- per step
    def _new_step(self) -> None:
        assert self._gen is not None
        # The fast path's LearnedSpecialist draws (B, N, 3) Gumbel noise per step
        self._noise = self._gen.gumbel(size=(1, self.n, 3))[0]
        self._actions[:] = 0
        self._submitted[:] = False
        self._heard = np.zeros(self.n, dtype=np.int64)
        self._sent = np.full(self.n, -1, dtype=np.int64)

    def action_noise(self, slot: int) -> np.ndarray:
        """This step's Gumbel noise for snake ``slot`` (common random numbers)."""
        return np.asarray(self._noise[slot])

    def record_sent(self, codes: np.ndarray) -> None:
        """Bookkeeping only (for comparisons): the codes the lead sent this step."""
        self._sent[:] = np.asarray(codes).reshape(self.n)

    def submit(self, slot: int, action: int, heard: int) -> None:
        """Snake ``slot``'s action for this step and the code it acted on."""
        if self._submitted[slot]:
            raise RuntimeError(f"Snake {slot} already acted at step {self.step_index}")
        self._actions[slot] = int(action)
        self._heard[slot] = int(heard)
        self._submitted[slot] = True

    def has_acted(self, slot: int) -> bool:
        return bool(self._submitted[slot])

    @property
    def ready(self) -> bool:
        """Every live snake has acted this step."""
        assert self.env is not None
        return bool(np.all(self._submitted | ~self.env.alive[0]))

    def advance(self) -> None:
        """Step the environment with the submitted actions (no messages: they travel
        as envelopes, never through the simulator)."""
        assert self.env is not None and not self.done
        if not self.ready:
            raise RuntimeError(f"Not every live snake acted at step {self.step_index}")
        self.history["codes"].append(self._sent.copy())
        self.history["heard"].append(self._heard.copy())
        self.history["actions"].append(self._actions.copy())
        self.history["alive"].append(self.env.alive[0].copy())
        self.obs, rewards, _, _ = self.env.step(self._actions[None])
        self.total += float(rewards[0, 0])
        self.step_index += 1
        self._new_step()

    # -- reporting
    def situation(self) -> dict[str, Any]:
        """Compact JSON snapshot: heads, headings, food, health, alive, score."""
        assert self.env is not None and self.obs is not None
        c = self.obs.centre
        return {
            "step": self.step_index,
            "seed": self.seed,
            "heads": c["heads"][0].tolist(),
            "heading": c["heading"][0].tolist(),
            "food": self.env.foods(0).tolist(),
            "health": [round(float(h), 4) for h in c["health"][0]],
            "alive": c["alive"][0].tolist(),
            "score": self.env.score[0].tolist(),
            "done": self.done,
        }

    def result(self) -> dict[str, Any]:
        assert self.env is not None
        food = self.env.score[0]
        return {
            "team_food": int(food.sum()),
            "food": food.tolist(),
            "steps": int(self.env.steps[0]),
            "survivors": int(self.env.alive[0].sum()),
            "return": float(self.total),
        }

    def stacked_history(self) -> dict[str, np.ndarray]:
        """(T, N) arrays of codes sent, codes heard, actions and alive masks."""
        return {
            k: np.stack(v) if v else np.zeros((0, self.n), dtype=np.int64)
            for k, v in self.history.items()
        }


# ------------------------------------------------------------------ specialists
class SquadLead:
    """The squad's centre policy as a runtime specialist (see module docstring)."""

    def __init__(
        self,
        world: SquadWorld,
        centre: CentrePolicy,
        members: Sequence[str],
        vocab_size: int = 8,
        name: str = LEAD,
    ):
        self.world = world
        self.centre = centre
        self.members = list(members)
        self.vocab_size = vocab_size
        self.name = name
        self._goal: Optional[Envelope] = None
        self._sent_step = -1

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self.name,
            version="1",
            accepts=(f"{SQUAD_FORAGE}@1",),
            description="Squad centre: sends one code per snake per step",
            tags=("squad", "centre"),
        )

    def handle(self, envelope: Envelope) -> list[Envelope]:
        if is_goal(envelope):
            return self._start(envelope)
        if (
            is_control(envelope, "cancel")
            and self._goal is not None
            and envelope.task_id == self._goal.task_id
        ):
            self.world.stop()
            self._goal = None
        return []

    def _start(self, envelope: Envelope) -> list[Envelope]:
        if self._goal is not None:
            return [fail(envelope, self.name, "busy with another episode")]
        params = goal_from_envelope(envelope).params
        assignment = list(params.get("assignment", self.members))
        unknown = sorted(set(assignment) - set(self.members))
        if unknown:
            return [fail(envelope, self.name, f"unknown squad members {unknown}")]
        self.world.start(
            params["episode_seed"],
            assignment,
            task_id=envelope.task_id,
            trace_id=envelope.trace_id,
            max_steps=params.get("max_steps"),
        )
        self.centre.reset([params["episode_seed"]])
        self._goal = envelope
        self._sent_step = -1
        return []

    def on_tick(self, clock: int) -> list[Envelope]:
        goal, world = self._goal, self.world
        if goal is None or not world.active:
            return []
        if world.done:
            self._goal = None
            return [complete(goal, self.name, world.result(), clock=clock)]
        if self._sent_step == world.step_index:
            return []
        assert world.obs is not None
        codes = np.asarray(self.centre.send(world.obs), dtype=np.int64)  # (1, N, 1)
        world.record_sent(codes[0, :, 0])
        self._sent_step = world.step_index
        names = [""] * world.n
        for name, slot in world.slots.items():
            names[slot] = name
        batch = codes_to_envelopes(
            codes,
            self.vocab_size,
            self.name,
            names,
            task_ids=[goal.task_id],
            trace_ids=[goal.trace_id],
            parent_ids=[goal.msg_id],
            clock=clock,
        )[0]
        return [e for slot, e in enumerate(batch) if world.alive(slot)]


class SnakeSpecialist:
    """One snake: acts on its window, health and the last verified code it heard.

    Args:
        name: Bus address
        world: The simulator (sensors and actuators)
        net, params, vocab: The trained specialist network (shared by every snake)
    """

    def __init__(
        self, name: str, world: SquadWorld, net: MLP, params: np.ndarray, vocab: int
    ):
        self.name = name
        self.world = world
        self.net = net
        self.params = np.asarray(params)
        self.vocab = vocab
        self.code = 0
        self.code_task: Optional[str] = None
        self.received = 0

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self.name,
            version="1",
            accepts=(),
            payload_encodings=(CODES_CODEC,),
            description="Colour specialist snake (learned policy)",
            tags=("squad", "snake"),
        )

    def handle(self, envelope: Envelope) -> list[Envelope]:
        if envelope.payload_type == "codes" and codec_of(envelope) == CODES_CODEC:
            codes = envelopes_to_codes([[envelope]], recipients=[self.name])
            self.code = int(codes[0, 0, 0])
            self.code_task = envelope.task_id
            self.received += 1
        return []

    def heard(self) -> int:
        """The code to act on: the last verified one of the current task, else 0."""
        return self.code if self.code_task == self.world.task_id else 0

    def on_tick(self, clock: int) -> list[Envelope]:
        world = self.world
        slot = world.slot_of(self.name)
        if slot is None or world.done or not world.alive(slot):
            return []
        if world.has_acted(slot):
            return []
        assert world.obs is not None
        obs = world.obs.specialists
        heard = self.heard()
        own = {
            "window": obs["window"][:, slot : slot + 1],
            "health": obs["health"][:, slot : slot + 1],
            "message": np.full((1, 1, 1), heard, dtype=np.int64),
        }
        x = specialist_inputs(own, self.vocab).reshape(1, -1)
        logits = self.net.forward(self.params, x)[0]
        action = int(np.argmax(logits + world.action_noise(slot)))
        world.submit(slot, action, heard)
        return []


class TamperingAdapter(ClockAdapter):
    """A compromised sidecar on the lead: after signing, it rewrites the code of the
    envelopes listed in ``targets`` (``(episode step, recipient)`` pairs)."""

    def __init__(
        self,
        *args: Any,
        world: SquadWorld,
        targets: Sequence[tuple[int, str]],
        **kw: Any,
    ) -> None:
        super().__init__(*args, **kw)
        self.world = world
        self.targets = set(targets)
        self.tampered: list[dict[str, Any]] = []

    def _outbound(self, envelopes: Optional[Sequence[Envelope]]) -> list[Envelope]:
        out = []
        for envelope in super()._outbound(envelopes):
            key = (self.world.step_index, envelope.recipients[0])
            if envelope.payload_type == "codes" and key in self.targets:
                vocab = envelope.payload_encoding["V"]
                forged = [(envelope.payload[0] + 1) % vocab]
                self.tampered.append(
                    {
                        "step": key[0],
                        "recipient": key[1],
                        "msg_id": envelope.msg_id,
                        "sent": int(envelope.payload[0]),
                        "forged": int(forged[0]),
                    }
                )
                envelope = Envelope.from_dict({**envelope.to_dict(), "payload": forged})
            out.append(envelope)
        return out


# ------------------------------------------------------------------ the runtime
@dataclass
class EpisodeOutcome:
    """One object-layer episode."""

    task_id: str
    trace_id: str
    state: str
    result: Optional[dict[str, Any]]
    error: Optional[str]
    ticks: int
    history: dict[str, np.ndarray]
    rejections: list[dict[str, Any]] = field(default_factory=list)
    tampered: list[dict[str, Any]] = field(default_factory=list)


class SquadRuntime:
    """Command centre, lead and snakes over an ``InProcessBus``, everyone signing.

    Args:
        specialist: The trained specialist policy (its network is shared by the snakes)
        centre: The squad centre policy (learned, hand-coded, silent, ...)
        config: Squad configuration
        trace: Trace log (default: in-memory)
        keys: Signing keys of ``command-centre`` and ``squad-lead``
        tamper: ``(episode step, snake name)`` envelopes the lead's sidecar corrupts
    """

    def __init__(
        self,
        specialist: LearnedSpecialist,
        centre: CentrePolicy,
        config: SquadConfig,
        trace: Optional[TraceLog] = None,
        keys: Optional[Mapping[str, bytes]] = None,
        tamper: Sequence[tuple[int, str]] = (),
    ):
        keys = dict(keys or DEFAULT_KEYS)
        self.config = config
        self.trace = trace if trace is not None else TraceLog()
        self.specs = squad_specs(config.n_snakes)
        self.world = SquadWorld(config, noise_salt=specialist.salt)
        self.names = snake_names(config.n_snakes)
        self.centre = CommandCentre(
            self.specs,
            bus=InProcessBus(),
            trace=self.trace,
            name=CENTRE,
            signing_key=keys[CENTRE],
            verify_keys={LEAD: keys[LEAD]},
        )
        lead = SquadLead(
            self.world, centre, self.names, config.message.vocab_size, name=LEAD
        )
        lead_kw: dict[str, Any] = {
            "signing_key": keys[LEAD],
            "verify_keys": {CENTRE: keys[CENTRE]},
            "key_id": LEAD,
        }
        self.lead_adapter: ClockAdapter = (
            TamperingAdapter(lead, world=self.world, targets=tamper, **lead_kw)
            if tamper
            else ClockAdapter(lead, **lead_kw)
        )
        self.centre.register(self.lead_adapter)
        self.snakes: dict[str, ClockAdapter] = {}
        for name in self.names:
            snake = SnakeSpecialist(
                name, self.world, specialist.net, specialist.params, specialist.vocab
            )
            adapter = ClockAdapter(snake, verify_keys={LEAD: keys[LEAD]})
            self.snakes[name] = adapter
            self.centre.register(adapter)

    def goal(
        self,
        seed: int,
        max_steps: Optional[int] = None,
        assignment: Optional[Sequence[str]] = None,
    ) -> GoalSpec:
        """A validated ``squad.forage`` GoalSpec."""
        params: dict[str, Any] = {"episode_seed": int(seed)}
        if max_steps is not None:
            params["max_steps"] = int(max_steps)
        if assignment is not None:
            params["assignment"] = list(assignment)
        return self.specs.make(SQUAD_FORAGE, params)

    def run_episode(
        self,
        seed: int,
        max_steps: Optional[int] = None,
        assignment: Optional[Sequence[str]] = None,
        max_ticks: int = 10 * MAX_EPISODE_STEPS,
    ) -> EpisodeOutcome:
        """Submit one ``squad.forage`` task and step the runtime until it finishes."""
        task = self.centre.submit(self.goal(seed, max_steps, assignment))
        trace_id = self.centre.status(task).trace_id
        seen = {name: len(a.rejected) for name, a in self.snakes.items()}
        rejections: list[dict[str, Any]] = []
        tamper_seen = len(getattr(self.lead_adapter, "tampered", []))
        ticks = 0
        while not self.centre.status(task).done:
            if ticks >= max_ticks:
                self.centre.cancel(task)
                break
            self.centre.step()
            ticks += 1
            rejections += self._log_rejections(seen, trace_id)
            world = self.world
            if world.active and world.task_id == task and not world.done:
                # Snapshot of the step whose codes were just sent (same clock)
                self.trace.snapshot(
                    trace_id, world.situation(), clock=self.centre.clock
                )
                world.advance()
                if _finished(world):
                    self.trace.snapshot(
                        trace_id, world.situation(), clock=self.centre.clock
                    )
        self.centre.step()  # Deliver the stragglers (last codes, cancel notices)
        rejections += self._log_rejections(seen, trace_id)
        status = self.centre.status(task)
        history = self.world.stacked_history()
        self.world.stop()
        return EpisodeOutcome(
            task_id=task,
            trace_id=trace_id,
            state=status.state,
            result=status.result,
            error=status.error,
            ticks=ticks,
            history=history,
            rejections=rejections,
            tampered=list(getattr(self.lead_adapter, "tampered", []))[tamper_seen:],
        )

    def _log_rejections(
        self, seen: dict[str, int], trace_id: str
    ) -> list[dict[str, Any]]:
        out = []
        for name, adapter in self.snakes.items():
            for rejection in adapter.rejected[seen[name] :]:
                row = {
                    "recipient": name,
                    "sender": rejection.envelope.sender,
                    "msg_id": rejection.envelope.msg_id,
                    "reason": rejection.reason,
                    "fallback": FALLBACK,
                }
                self.trace.event(
                    "signature_rejected", trace_id, self.centre.clock, **row
                )
                out.append(row)
            seen[name] = len(adapter.rejected)
        return out


def _finished(world: SquadWorld) -> bool:
    return world.done


# ------------------------------------------------------------------ fast path
class _RecordingSpecialist:
    """Wraps a specialist policy and keeps every action batch it returns."""

    def __init__(self, inner: LearnedSpecialist):
        self.inner = inner
        self.actions: list[np.ndarray] = []

    def reset(self, seeds: Sequence[int]) -> None:
        self.inner.reset(seeds)
        self.actions = []

    def act(self, obs: Observations) -> np.ndarray:
        actions = self.inner.act(obs)
        self.actions.append(actions[0].copy())
        return actions

    def probs(self, obs: Observations) -> np.ndarray:
        return self.inner.probs(obs)


def hold_last_hook(drops: Sequence[tuple[int, int]]) -> Any:
    """A message hook that replays the object layer's signature fallback: the
    messages to ``(step, receiver)`` in ``drops`` are replaced by the last message
    that receiver got (0 before any). B = 1 only."""
    dropped = set(drops)
    last: dict[int, int] = {}

    def hook(messages: np.ndarray, step: int, mask: Any = None) -> np.ndarray:
        out = np.array(messages, copy=True)
        for r in range(out.shape[1]):
            if (step, r) in dropped:
                out[0, r, 0] = last.get(r, 0)
            last[r] = int(out[0, r, 0])
        return out

    return hook


def fast_path_episode(
    specialist: LearnedSpecialist,
    centre: CentrePolicy,
    config: SquadConfig,
    seed: int,
    hook: Any = None,
) -> dict[str, Any]:
    """One B = 1 fast-path episode with per-step codes sent and heard, actions and
    alive masks ((T, N) arrays) and the ``run_episodes`` outputs."""
    rec = _RecordingSpecialist(specialist)
    codes: list[np.ndarray] = []
    heard: list[np.ndarray] = []
    alive: list[np.ndarray] = []

    def observer(step: int, obs: Observations, sent: Any, done: Any) -> None:
        codes.append(np.asarray(sent)[0, :, 0].copy())
        heard.append(obs.specialists["message"][0, :, 0].copy())
        alive.append(obs.centre["alive"][0].copy())

    out = run_episodes(SquadEnv(config), rec, centre, [seed], hook, observer)  # type: ignore[arg-type]
    return {
        "team_food": float(out["team_food"][0]),
        "steps": int(out["steps"][0]),
        "return": float(out["return"][0]),
        "codes": np.stack(codes),
        "heard": np.stack(heard),
        "actions": np.stack(rec.actions),
        "alive": np.stack(alive),
    }


def compare_episode(obj: EpisodeOutcome, fast: dict[str, Any]) -> dict[str, Any]:
    """Step-for-step comparison on live snakes: codes sent, codes heard, actions."""
    h = obj.history
    same_length = h["actions"].shape == fast["actions"].shape
    row: dict[str, Any] = {
        "steps_object": int(h["actions"].shape[0]),
        "steps_fast": int(fast["actions"].shape[0]),
        "team_food_object": (obj.result or {}).get("team_food"),
        "team_food_fast": fast["team_food"],
    }
    if same_length:
        live = h["alive"]
        row["alive_equal"] = bool(np.array_equal(live, fast["alive"]))
        for key in ("codes", "heard", "actions"):
            row[f"{key}_mismatches"] = int((h[key] != fast[key])[live].sum())
    row["identical"] = bool(
        same_length
        and row.get("alive_equal")
        and row["codes_mismatches"] == row["heard_mismatches"] == 0
        and row["actions_mismatches"] == 0
        and row["team_food_object"] == fast["team_food"]
    )
    return row
