"""Cooperative foraging squad: N colour-specialist snakes and a full-view command centre."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from ...messages import MessageSpec
from ..base import EnvSpec, MultiAgentEnv, Observations, StepResult
from . import engine

N_ACTIONS = 3  # 0 straight, 1 left, 2 right (relative to the heading)
WINDOW_CHANNELS = ("own_body", "other_bodies_and_walls", "own_food", "other_food")


@dataclass(frozen=True)
class SquadConfig:
    """Squad parameters.

    Args:
        board_size: Side of the square board; the edges are deadly walls
        n_snakes: Colour specialists (N); snake i eats only colour i
        food_per_colour: Food items per colour, each respawning when eaten
        max_health: Starting health; -1 per step, refilled only by own-colour food
        init_length: Starting body length, laid out straight behind the head
        max_steps: Episode length (T)
        obs_radius: Specialist window radius r; the window is (2r+1)²
        survival_reward: Team reward per living snake per step
        respawn_food: Respawn eaten food (tests switch it off for fixed layouts)
        message: Centre-to-specialist channel; the centre may send every
            ``message.interval`` steps
    """

    board_size: int = 21
    n_snakes: int = 3
    food_per_colour: int = 2
    max_health: int = 60
    init_length: int = 3
    max_steps: int = 300
    obs_radius: int = 5
    survival_reward: float = 0.01
    respawn_food: bool = True
    message: MessageSpec = field(default_factory=lambda: MessageSpec("discrete", 1, 8))

    def __post_init__(self) -> None:
        if self.board_size < 5:
            raise ValueError("board_size must be >= 5")
        if self.n_snakes < 1 or self.n_snakes > 64:
            raise ValueError("n_snakes must lie in [1, 64]")
        if self.food_per_colour < 1:
            raise ValueError("food_per_colour must be >= 1")
        if self.init_length < 1 or self.init_length > self.board_size // 2:
            raise ValueError("init_length must lie in [1, board_size // 2]")
        if self.obs_radius < 1:
            raise ValueError("obs_radius must be >= 1")
        if self.max_health < 1 or self.max_steps < 1:
            raise ValueError("max_health and max_steps must be >= 1")
        receivers = self.message.n_receivers
        if receivers is not None and receivers != self.n_snakes:
            raise ValueError("message.n_receivers must equal n_snakes")


class SquadEnv(MultiAgentEnv):
    """Batched squad environment on the numba engine.

    Specialist observations (B, N, ...):
        window: (4, 2r+1, 2r+1) uint8 0/1 planes, egocentric and rotated (heading up), channels
            ``WINDOW_CHANNELS``; cells off the board read as walls
        health: (1,) float32, health / max_health (0 once dead)
        message: the last payload received, per the shared message conventions
    Centre observations (B, ...):
        board: (3N, H, W) uint8 0/1 planes: N body channels, N head channels, N food
            colours
        health: (N,) float32, health / max_health
        heads: (N, 2) int64 head [x, y] (-1 once dead)
        food: (N * F, 2) int64 food [x, y]; row c * F + f has colour c (-1 if empty)
        heading: (N,) int64, absolute heading 0 N, 1 E, 2 S, 3 W
        alive: (N,) bool, which snakes are alive (the live-receiver mask for message
            hooks and metrics is ``alive & ~done``)

    The reward is shared: every specialist gets the team's food eaten this step plus
    ``survival_reward`` per living snake. An episode ends after ``max_steps`` steps or
    when every snake is dead.
    """

    def __init__(self, config: Optional[SquadConfig] = None):
        self.config = config or SquadConfig()
        c = self.config
        self.pad = max(1, c.obs_radius)
        self.window = 2 * c.obs_radius + 1
        msg = replace(c.message, n_receivers=c.n_snakes)
        self.spec = EnvSpec(
            n_specialists=c.n_snakes,
            n_actions=N_ACTIONS,
            specialist_obs={
                "window": ((4, self.window, self.window), np.uint8),
                "health": ((1,), np.float32),
                "message": (msg.payload_shape, msg.dtype),
            },
            centre_obs={
                "board": ((3 * c.n_snakes, c.board_size, c.board_size), np.uint8),
                "health": ((c.n_snakes,), np.float32),
                "heads": ((c.n_snakes, 2), np.int64),
                "food": ((c.n_snakes * c.food_per_colour, 2), np.int64),
                "heading": ((c.n_snakes,), np.int64),
                "alive": ((c.n_snakes,), np.bool_),
            },
            message=msg,
            max_steps=c.max_steps,
        )
        self._batch = 0

    @property
    def batch_size(self) -> int:
        """Number of parallel episodes since the last reset."""
        return self._batch

    def _allocate(self, batch: int) -> None:
        c = self.config
        n = c.n_snakes
        side = c.board_size + 2 * self.pad
        cap = c.board_size * c.board_size + 1
        self._batch = batch
        self.grid = np.zeros((batch, side, side), dtype=np.int16)
        self.food = np.zeros((batch, side, side), dtype=np.int8)
        self.body_x = np.zeros((batch, n, cap), dtype=np.int16)
        self.body_y = np.zeros((batch, n, cap), dtype=np.int16)
        self.head = np.zeros((batch, n), dtype=np.int64)
        self.length = np.zeros((batch, n), dtype=np.int64)
        self.heading = np.zeros((batch, n), dtype=np.int64)
        self.health = np.zeros((batch, n), dtype=np.int64)
        self.alive = np.zeros((batch, n), dtype=np.bool_)
        self.score = np.zeros((batch, n), dtype=np.int64)
        self.death = np.zeros((batch, n), dtype=np.int64)
        self.food_x = np.full((batch, n * c.food_per_colour), -1, dtype=np.int64)
        self.food_y = np.full((batch, n * c.food_per_colour), -1, dtype=np.int64)
        self.rng = np.zeros(batch, dtype=np.uint64)
        self.steps = np.zeros(batch, dtype=np.int64)
        self.done = np.zeros(batch, dtype=np.bool_)
        self.eaten = np.zeros((batch, n), dtype=np.int64)
        self.last_message = c.message.zeros(batch, n)

    def reset(self, seeds: Sequence[int]) -> Observations:
        """Start one episode per seed; the same seed always gives the same episode."""
        seeds = [int(s) for s in seeds]
        if not seeds:
            raise ValueError("reset needs at least one seed")
        c = self.config
        self._allocate(len(seeds))
        for b, seed in enumerate(seeds):
            self.rng[b] = np.uint64(seed % (1 << 64))
            engine.reset_episode(
                b, self.grid, self.food, self.body_x, self.body_y, self.head,
                self.length, self.heading, self.health, self.alive, self.score,
                self.death, self.food_x, self.food_y, self.rng, self.pad,
                c.init_length, c.food_per_colour, c.max_health,
            )  # fmt: skip
        return self.observe()

    def step(
        self, actions: np.ndarray, messages: Optional[np.ndarray] = None
    ) -> StepResult:
        """Advance every unfinished episode; see ``engine`` for the move order."""
        if self._batch == 0:
            raise RuntimeError("Call reset() before step()")
        c = self.config
        acts = self.validate_actions(actions)
        msg_spec = c.message
        if messages is not None and msg_spec.kind != "none":
            msgs = msg_spec.validate(messages, self._batch, c.n_snakes)
            # The centre may send on steps 0, k, 2k, ... of each unfinished episode
            send = ~self.done & (self.steps % msg_spec.interval == 0)
            self.last_message[send] = msgs[send]
        was_done = self.done.copy()
        engine.step_batch(
            acts, self.grid, self.food, self.body_x, self.body_y, self.head,
            self.length, self.heading, self.health, self.alive, self.score,
            self.death, self.food_x, self.food_y, self.rng, self.steps, self.done,
            self.eaten, self.pad, c.food_per_colour, c.max_health, c.max_steps,
            c.respawn_food,
        )  # fmt: skip
        team = self.eaten.sum(axis=1) + c.survival_reward * self.alive.sum(axis=1)
        team = np.where(was_done, 0.0, team).astype(np.float32)
        rewards = np.repeat(team[:, None], c.n_snakes, axis=1)
        info = {
            "food_eaten": self.eaten.copy(),
            "score": self.score.copy(),
            "alive": self.alive.copy(),
            "death_reason": self.death.copy(),
            "steps": self.steps.copy(),
        }
        return StepResult(self.observe(), rewards, self.done.copy(), info)

    def observe(self) -> Observations:
        """Current observations (fresh arrays, safe to keep)."""
        c = self.config
        batch, n = self._batch, c.n_snakes
        windows = np.empty((batch, n, 4, self.window, self.window), dtype=np.uint8)
        board = np.empty((batch, 3 * n, c.board_size, c.board_size), dtype=np.uint8)
        heads = np.empty((batch, n, 2), dtype=np.int64)
        engine.specialist_windows(
            windows, self.grid, self.food, self.body_x, self.body_y, self.head,
            self.heading, self.alive, c.obs_radius, self.pad,
        )  # fmt: skip
        engine.centre_board(
            board, heads, self.food_x, self.food_y, self.body_x, self.body_y,
            self.head, self.length, self.alive, c.food_per_colour,
        )  # fmt: skip
        health = (self.health * self.alive / c.max_health).astype(np.float32)
        return Observations(
            specialists={
                "window": windows,
                "health": health[:, :, None],
                "message": self.last_message.copy(),
            },
            centre={
                "board": board,
                "health": health,
                "heading": self.heading.copy(),
                "heads": heads,
                "food": np.stack([self.food_x, self.food_y], axis=2),
                "alive": self.alive.copy(),
            },
        )

    # ------------------------------------------------------------------ inspection
    def bodies(self, b: int) -> tuple[np.ndarray, ...]:
        """Body cells of each snake in episode b, head first, as (length, 2) [x, y]."""
        cap = self.body_x.shape[2]
        out = []
        for i in range(self.config.n_snakes):
            if not self.alive[b, i]:
                out.append(np.zeros((0, 2), dtype=np.int64))
                continue
            idx = (self.head[b, i] - np.arange(self.length[b, i])) % cap
            out.append(
                np.stack([self.body_x[b, i, idx], self.body_y[b, i, idx]], 1).astype(
                    np.int64
                )
            )
        return tuple(out)

    def foods(self, b: int) -> np.ndarray:
        """Food in episode b as rows [x, y, colour]; empty slots are left out."""
        f = self.config.food_per_colour
        rows = [
            (int(self.food_x[b, s]), int(self.food_y[b, s]), s // f)
            for s in range(self.food_x.shape[1])
            if self.food_x[b, s] >= 0
        ]
        return np.array(rows, dtype=np.int64).reshape(-1, 3)

    def set_state(
        self,
        b: int,
        bodies: Sequence[Sequence[tuple[int, int]]],
        headings: Sequence[int],
        foods: Sequence[tuple[int, int, int]] = (),
        health: Optional[Sequence[int]] = None,
        steps: int = 0,
    ) -> Observations:
        """Overwrite episode b with a hand-made layout (for tests and scenarios).

        Args:
            b: Episode index (``reset`` must have been called)
            bodies: Per snake, its cells head first; an empty list makes it dead
            headings: Per snake heading, 0 N, 1 E, 2 S, 3 W
            foods: (x, y, colour) items, at most ``food_per_colour`` per colour
            health: Per snake health (default: max)
            steps: Step counter to start from
        """
        c = self.config
        n, f, p = c.n_snakes, c.food_per_colour, self.pad
        if len(bodies) != n or len(headings) != n:
            raise ValueError(f"Need {n} bodies and headings")
        cap = self.body_x.shape[2]
        self.grid[b] = engine.WALL
        self.grid[b, p : p + c.board_size, p : p + c.board_size] = 0
        self.food[b] = 0
        self.food_x[b] = -1
        self.food_y[b] = -1
        for i, cells in enumerate(bodies):
            self.alive[b, i] = len(cells) > 0
            self.death[b, i] = engine.ALIVE
            self.score[b, i] = 0
            self.heading[b, i] = int(headings[i])
            self.health[b, i] = c.max_health if health is None else int(health[i])
            self.length[b, i] = len(cells)
            self.head[b, i] = max(len(cells) - 1, 0) % cap
            for k, (x, y) in enumerate(cells):
                if not (0 <= x < c.board_size and 0 <= y < c.board_size):
                    raise ValueError(f"Cell {(x, y)} is off the board")
                if self.grid[b, y + p, x + p] != 0:
                    raise ValueError(f"Cell {(x, y)} is occupied twice")
                idx = len(cells) - 1 - k
                self.body_x[b, i, idx] = x
                self.body_y[b, i, idx] = y
                self.grid[b, y + p, x + p] = i + 1
        used = [0] * n
        for x, y, colour in foods:
            if used[colour] >= f:
                raise ValueError(f"More than {f} food items of colour {colour}")
            slot = colour * f + used[colour]
            used[colour] += 1
            self.food_x[b, slot] = x
            self.food_y[b, slot] = y
            self.food[b, y + p, x + p] = colour + 1
        self.steps[b] = steps
        self.done[b] = not self.alive[b].any()
        return self.observe()
