"""Numba kernels for the cooperative foraging squad.

Design: an occupancy grid plus ring-buffer bodies, all plain numpy arrays.

- ``grid`` (B, H + 2P, W + 2P) int16: ``WALL`` (-1) on the P-cell border, 0 empty,
  ``i + 1`` for a cell holding a segment of snake i. The border is at least as wide as
  the observation radius, so windows and moves never index outside the array.
- ``food`` (same shape) int8: 0 none, ``c + 1`` for food of colour c. Food and bodies
  are separate layers because a snake may walk over food of another colour.
- ``body_x``/``body_y`` (B, N, cap) int16 ring buffers in board coordinates; ``head``
  (B, N) is the ring index of the head and the tail is ``head - length + 1``.
- ``food_x``/``food_y`` (B, N * F) int64; slot ``c * F + f`` holds colour c, -1 = empty.

Simultaneous moves resolve in this order (``step_batch``):

1. Every living snake turns (0 straight, 1 left, 2 right) and picks its new head cell.
   It *intends to eat* if that cell is on the board and holds food of its own colour.
2. Tails vacate: each living snake that does not intend to eat frees its tail cell. A
   snake that eats keeps its tail and grows by one.
3. Health drops by 1, and is refilled to the maximum by eating.
4. Eliminations, checked against the grid after step 2 (bodies of snakes that die this
   step still block): a wall kills, an occupied cell (any body, own included) kills,
   head-to-head on one cell kills the shorter snake (length after growth; equal lengths
   both die), and health 0 kills.
5. Survivors write their new head; dead snakes are removed from the grid. Only
   survivors eat: their food is consumed and respawns on a random free cell.

Directions are 0 north (y - 1), 1 east, 2 south, 3 west. Every helper takes plain
ints; no enum ever reaches numba.
"""

import numba  # type: ignore
import numpy as np

WALL = -1

# Death reasons in the ``death`` array
ALIVE = 0
DEATH_WALL = 1
DEATH_BODY = 2
DEATH_HEAD = 3
DEATH_STARVATION = 4
DEATH_REASONS = ("alive", "wall", "body", "head", "starvation")

DIR_DX = np.array([0, 1, 0, -1], dtype=np.int64)
DIR_DY = np.array([-1, 0, 1, 0], dtype=np.int64)


@numba.njit(cache=True)  # type: ignore
def next_u64(rng: np.ndarray, b: int) -> np.uint64:
    """SplitMix64 step on the per-episode state ``rng[b]``."""
    z = rng[b] + np.uint64(0x9E3779B97F4A7C15)
    rng[b] = z
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@numba.njit(cache=True)  # type: ignore
def randint(rng: np.ndarray, b: int, n: int) -> int:
    """Uniform int in [0, n) from episode b's stream."""
    return int(next_u64(rng, b) % np.uint64(n))


@numba.njit(cache=True)  # type: ignore
def uniform(rng: np.ndarray, b: int) -> float:
    """Uniform float in [0, 1) from episode b's stream."""
    return float(next_u64(rng, b) >> np.uint64(11)) / 9007199254740992.0


@numba.njit(cache=True)  # type: ignore
def turn(direction: int, action: int) -> int:
    """Heading after a relative action (0 straight, 1 left, 2 right)."""
    if action == 1:
        return (direction + 3) % 4
    if action == 2:
        return (direction + 1) % 4
    return direction


@numba.njit(cache=True)  # type: ignore
def _place_food_slot(
    b: int, slot: int, colour: int, grid, food, food_x, food_y, rng, pad: int
) -> None:
    """Put food slot ``slot`` on a random cell with no body and no food."""
    height = grid.shape[1] - 2 * pad
    width = grid.shape[2] - 2 * pad
    free = 0
    for y in range(height):
        for x in range(width):
            if grid[b, y + pad, x + pad] == 0 and food[b, y + pad, x + pad] == 0:
                free += 1
    if free == 0:
        food_x[b, slot] = -1
        food_y[b, slot] = -1
        return
    k = randint(rng, b, free)
    for y in range(height):
        for x in range(width):
            if grid[b, y + pad, x + pad] == 0 and food[b, y + pad, x + pad] == 0:
                if k == 0:
                    food[b, y + pad, x + pad] = colour + 1
                    food_x[b, slot] = x
                    food_y[b, slot] = y
                    return
                k -= 1


@numba.njit(cache=True)  # type: ignore
def reset_episode(
    b: int,
    grid,
    food,
    body_x,
    body_y,
    head,
    length,
    heading,
    health,
    alive,
    score,
    death,
    food_x,
    food_y,
    rng,
    pad: int,
    init_length: int,
    food_per_colour: int,
    max_health: int,
) -> None:
    """Fresh episode b: walls, N straight snakes at random free spots, then food."""
    n = head.shape[1]
    cap = body_x.shape[2]
    height = grid.shape[1] - 2 * pad
    width = grid.shape[2] - 2 * pad
    grid[b, :, :] = WALL
    grid[b, pad : pad + height, pad : pad + width] = 0
    food[b, :, :] = 0
    for i in range(n):
        placed = False
        while not placed:
            x = randint(rng, b, width)
            y = randint(rng, b, height)
            d = randint(rng, b, 4)
            # Free cell ahead of the head, and the body trails straight behind it
            ok = grid[b, y + DIR_DY[d] + pad, x + DIR_DX[d] + pad] == 0
            for k in range(init_length):
                cx = x - k * DIR_DX[d]
                cy = y - k * DIR_DY[d]
                if grid[b, cy + pad, cx + pad] != 0:
                    ok = False
            if not ok:
                continue
            for k in range(init_length):
                idx = init_length - 1 - k
                cx = x - k * DIR_DX[d]
                cy = y - k * DIR_DY[d]
                body_x[b, i, idx] = cx
                body_y[b, i, idx] = cy
                grid[b, cy + pad, cx + pad] = i + 1
            head[b, i] = (init_length - 1) % cap
            length[b, i] = init_length
            heading[b, i] = d
            health[b, i] = max_health
            alive[b, i] = True
            score[b, i] = 0
            death[b, i] = ALIVE
            placed = True
    for c in range(n):
        for f in range(food_per_colour):
            _place_food_slot(
                b, c * food_per_colour + f, c, grid, food, food_x, food_y, rng, pad
            )


@numba.njit(cache=True)  # type: ignore
def step_batch(
    actions,
    grid,
    food,
    body_x,
    body_y,
    head,
    length,
    heading,
    health,
    alive,
    score,
    death,
    food_x,
    food_y,
    rng,
    steps,
    done,
    eaten,
    pad: int,
    food_per_colour: int,
    max_health: int,
    max_steps: int,
    respawn_food: bool,
) -> None:
    """Advance every unfinished episode one step (see module docstring for the order).

    ``eaten`` (B, N) is overwritten with the food each snake ate this step.
    """
    n_batch, n = head.shape
    cap = body_x.shape[2]
    new_x = np.zeros(n, dtype=np.int64)
    new_y = np.zeros(n, dtype=np.int64)
    new_dir = np.zeros(n, dtype=np.int64)
    new_len = np.zeros(n, dtype=np.int64)
    eats = np.zeros(n, dtype=np.bool_)
    moving = np.zeros(n, dtype=np.bool_)
    reason = np.zeros(n, dtype=np.int64)
    for b in range(n_batch):
        for i in range(n):
            eaten[b, i] = 0
        if done[b]:
            continue
        # 1. new heads and eating intent
        for i in range(n):
            moving[i] = alive[b, i]
            eats[i] = False
            reason[i] = ALIVE
            if not moving[i]:
                continue
            d = turn(heading[b, i], actions[b, i])
            h = head[b, i]
            new_dir[i] = d
            new_x[i] = body_x[b, i, h] + DIR_DX[d]
            new_y[i] = body_y[b, i, h] + DIR_DY[d]
            eats[i] = food[b, new_y[i] + pad, new_x[i] + pad] == i + 1 and (
                grid[b, new_y[i] + pad, new_x[i] + pad] != WALL
            )
        # 2. tails vacate unless the snake eats
        for i in range(n):
            if not moving[i]:
                continue
            if eats[i]:
                new_len[i] = length[b, i] + 1
            else:
                tail = (head[b, i] - length[b, i] + 1) % cap
                grid[b, body_y[b, i, tail] + pad, body_x[b, i, tail] + pad] = 0
                length[b, i] -= 1
                new_len[i] = length[b, i] + 1
        # 3. health
        for i in range(n):
            if moving[i]:
                health[b, i] = max_health if eats[i] else health[b, i] - 1
        # 4. eliminations
        for i in range(n):
            if not moving[i]:
                continue
            cell = grid[b, new_y[i] + pad, new_x[i] + pad]
            if cell == WALL:
                reason[i] = DEATH_WALL
            elif cell > 0:
                reason[i] = DEATH_BODY
        for i in range(n):
            if not moving[i]:
                continue
            for j in range(n):
                if j == i or not moving[j]:
                    continue
                if new_x[i] == new_x[j] and new_y[i] == new_y[j]:
                    if new_len[i] <= new_len[j] and reason[i] == ALIVE:
                        reason[i] = DEATH_HEAD
        for i in range(n):
            if moving[i] and reason[i] == ALIVE and health[b, i] <= 0:
                reason[i] = DEATH_STARVATION
        # 5. apply
        for i in range(n):
            if not moving[i]:
                continue
            if reason[i] == ALIVE:
                h = (head[b, i] + 1) % cap
                head[b, i] = h
                body_x[b, i, h] = new_x[i]
                body_y[b, i, h] = new_y[i]
                length[b, i] = new_len[i]
                heading[b, i] = new_dir[i]
                grid[b, new_y[i] + pad, new_x[i] + pad] = i + 1
                if eats[i]:
                    food[b, new_y[i] + pad, new_x[i] + pad] = 0
                    score[b, i] += 1
                    eaten[b, i] = 1
            else:
                alive[b, i] = False
                death[b, i] = reason[i]
                health[b, i] = max(health[b, i], 0)
                for k in range(length[b, i]):
                    idx = (head[b, i] - k) % cap
                    gy = body_y[b, i, idx] + pad
                    gx = body_x[b, i, idx] + pad
                    if grid[b, gy, gx] == i + 1:
                        grid[b, gy, gx] = 0
        # Food respawn after all heads are written so it never lands on a body
        for i in range(n):
            if eaten[b, i] == 0:
                continue
            for f in range(food_per_colour):
                slot = i * food_per_colour + f
                if food_x[b, slot] == new_x[i] and food_y[b, slot] == new_y[i]:
                    food_x[b, slot] = -1
                    food_y[b, slot] = -1
                    if respawn_food:
                        _place_food_slot(
                            b, slot, i, grid, food, food_x, food_y, rng, pad
                        )
                    break
        steps[b] += 1
        any_alive = False
        for i in range(n):
            if alive[b, i]:
                any_alive = True
        if not any_alive or steps[b] >= max_steps:
            done[b] = True


@numba.njit(cache=True)  # type: ignore
def specialist_windows(
    out, grid, food, body_x, body_y, head, heading, alive, radius: int, pad: int
) -> None:
    """Egocentric windows (B, N, 4, 2r+1, 2r+1), rotated so the heading points up.

    Channels (0/1): own body, other bodies and walls, own-colour food, other food. Row r - 1,
    column r is the cell straight ahead; column r + 1 is to the right. Dead snakes get
    zeros.
    """
    n_batch, n = head.shape
    size = 2 * radius + 1
    for b in range(n_batch):
        for i in range(n):
            out[b, i, :, :, :] = 0
            if not alive[b, i]:
                continue
            h = head[b, i]
            hx = body_x[b, i, h]
            hy = body_y[b, i, h]
            d = heading[b, i]
            fx = DIR_DX[d]
            fy = DIR_DY[d]
            rx = DIR_DX[(d + 1) % 4]
            ry = DIR_DY[(d + 1) % 4]
            for oy in range(size):
                ahead = radius - oy
                for ox in range(size):
                    right = ox - radius
                    gx = hx + ahead * fx + right * rx + pad
                    gy = hy + ahead * fy + right * ry + pad
                    cell = grid[b, gy, gx]
                    if cell == i + 1:
                        out[b, i, 0, oy, ox] = 1
                    elif cell != 0:
                        out[b, i, 1, oy, ox] = 1
                    item = food[b, gy, gx]
                    if item == i + 1:
                        out[b, i, 2, oy, ox] = 1
                    elif item != 0:
                        out[b, i, 3, oy, ox] = 1


@numba.njit(cache=True)  # type: ignore
def centre_board(
    out,
    heads,
    food_x,
    food_y,
    body_x,
    body_y,
    head,
    length,
    alive,
    food_per_colour: int,
) -> None:
    """Full board (B, 3N, H, W) of 0/1: N body channels, N head channels, N food
    colours; also writes ``heads`` (B, N, 2) [x, y], -1 for dead snakes.
    """
    n_batch, n = head.shape
    cap = body_x.shape[2]
    for b in range(n_batch):
        out[b, :, :, :] = 0
        for i in range(n):
            heads[b, i, 0] = -1
            heads[b, i, 1] = -1
            if not alive[b, i]:
                continue
            h = head[b, i]
            for k in range(length[b, i]):
                idx = (h - k) % cap
                out[b, i, body_y[b, i, idx], body_x[b, i, idx]] = 1
            out[b, n + i, body_y[b, i, h], body_x[b, i, h]] = 1
            heads[b, i, 0] = body_x[b, i, h]
            heads[b, i, 1] = body_y[b, i, h]
        for slot in range(food_x.shape[1]):
            if food_x[b, slot] >= 0:
                out[
                    b, 2 * n + slot // food_per_colour, food_y[b, slot], food_x[b, slot]
                ] = 1
