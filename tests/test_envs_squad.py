"""Tests for the cooperative foraging squad (envs.squad): rules, shapes, policies."""

import numpy as np
import pytest

from genetic_coil_arena.envs.base import MessageSpec
from genetic_coil_arena.envs.squad import (
    GreedyPolicy,
    HandCodedCentre,
    NoCommPolicy,
    OraclePolicy,
    RandomCentre,
    RandomPolicy,
    SilentCentre,
    SquadConfig,
    SquadEnv,
    engine,
    run_episodes,
)
from genetic_coil_arena.snake_core.game import SnakeGame

N, E, S, W = 0, 1, 2, 3
STRAIGHT, LEFT, RIGHT = 0, 1, 2


def make_env(n_snakes=2, board_size=11, **kwargs):
    kwargs.setdefault("respawn_food", False)
    kwargs.setdefault("obs_radius", 3)
    env = SquadEnv(SquadConfig(board_size=board_size, n_snakes=n_snakes, **kwargs))
    env.reset([0])
    return env


def step(env, *actions, messages=None):
    return env.step(np.array([actions], dtype=np.int64), messages)


def death(env, i):
    return engine.DEATH_REASONS[env.death[0, i]]


class TestCollisions:
    def test_wall_kills(self):
        env = make_env(1)
        env.set_state(0, [[(0, 5), (1, 5)]], [W])
        _, _, done, _ = step(env, STRAIGHT)
        assert not env.alive[0, 0] and death(env, 0) == "wall" and done[0]

    def test_own_body_kills(self):
        env = make_env(1)
        env.set_state(0, [[(2, 2), (1, 2), (1, 3), (2, 3), (3, 3)]], [E])
        step(env, RIGHT)  # south into (2, 3), not the tail
        assert death(env, 0) == "body"

    def test_tail_vacates_first(self):
        env = make_env(1)
        env.set_state(0, [[(2, 2), (2, 3), (3, 3), (3, 2)]], [N])
        step(env, RIGHT)  # east into its own tail cell
        assert env.alive[0, 0]
        assert env.bodies(0)[0].tolist() == [[3, 2], [2, 2], [2, 3], [3, 3]]

    def test_other_body_kills(self):
        env = make_env(2)
        env.set_state(0, [[(5, 5), (5, 6), (5, 7)], [(4, 6), (3, 6)]], [N, E])
        step(env, STRAIGHT, STRAIGHT)  # B moves into (5, 6), A's body
        assert env.alive[0, 0] and death(env, 1) == "body"

    def test_tail_of_other_snake_vacates(self):
        env = make_env(2)
        env.set_state(0, [[(5, 5), (5, 6), (5, 7)], [(4, 7), (3, 7)]], [N, E])
        step(env, STRAIGHT, STRAIGHT)  # B into A's tail cell as A moves on
        assert env.alive[0].all()

    def test_eating_keeps_tail(self):
        env = make_env(2)
        env.set_state(
            0, [[(5, 5), (5, 6), (5, 7)], [(4, 7), (3, 7)]], [N, E], foods=[(5, 4, 0)]
        )
        _, rewards, _, info = step(env, STRAIGHT, STRAIGHT)
        assert env.alive[0, 0] and death(env, 1) == "body"
        assert env.length[0, 0] == 4 and env.score[0, 0] == 1
        assert info["food_eaten"][0].tolist() == [1, 0]
        assert rewards[0].tolist() == pytest.approx([1.01, 1.01])

    def test_head_to_head_longer_survives(self):
        env = make_env(2)
        env.set_state(0, [[(3, 5), (2, 5), (1, 5)], [(5, 5), (6, 5)]], [E, W])
        step(env, STRAIGHT, STRAIGHT)
        assert env.alive[0, 0] and death(env, 1) == "head"
        assert env.bodies(0)[0][0].tolist() == [4, 5]

    def test_head_to_head_equal_both_die(self):
        env = make_env(2)
        env.set_state(0, [[(3, 5), (2, 5)], [(5, 5), (6, 5)]], [E, W])
        _, _, done, _ = step(env, STRAIGHT, STRAIGHT)
        assert death(env, 0) == "head" and death(env, 1) == "head" and done[0]

    def test_head_to_head_uses_length_after_eating(self):
        # A (2) eats on the contested cell and ties B (3): both die
        env = make_env(2)
        env.set_state(
            0,
            [[(3, 5), (2, 5)], [(5, 5), (6, 5), (7, 5)]],
            [E, W],
            foods=[(4, 5, 0)],
        )
        step(env, STRAIGHT, STRAIGHT)
        assert not env.alive[0].any()
        assert env.score[0, 0] == 0  # only survivors eat
        assert env.foods(0).tolist() == [[4, 5, 0]]

    def test_swap_kills_both(self):
        env = make_env(2)
        env.set_state(0, [[(3, 5), (2, 5)], [(4, 5), (5, 5)]], [E, W])
        step(env, STRAIGHT, STRAIGHT)
        assert death(env, 0) == "body" and death(env, 1) == "body"

    def test_length_one_snakes_pass_through(self):
        env = make_env(2)
        env.set_state(0, [[(3, 5)], [(4, 5)]], [E, W])
        step(env, STRAIGHT, STRAIGHT)
        assert env.alive[0].all()
        assert [b[0].tolist() for b in env.bodies(0)] == [[4, 5], [3, 5]]

    def test_dead_bodies_are_removed(self):
        env = make_env(2)
        env.set_state(0, [[(0, 5), (1, 5), (2, 5)], [(1, 7), (1, 8)]], [W, N])
        step(env, STRAIGHT, STRAIGHT)  # A hits the wall
        assert death(env, 0) == "wall"
        assert (env.grid[0] == 1).sum() == 0
        step(env, STRAIGHT, STRAIGHT)  # B walks through where A was
        assert env.alive[0, 1] and env.bodies(0)[1][0].tolist() == [1, 5]


class TestFoodAndHealth:
    def test_other_colour_food_is_not_eaten(self):
        env = make_env(2)
        env.set_state(0, [[(5, 5), (5, 6)], [(1, 1)]], [N, E], foods=[(5, 4, 1)])
        step(env, STRAIGHT, STRAIGHT)
        step(env, STRAIGHT, STRAIGHT)  # walk off it again
        assert env.score[0].sum() == 0 and env.length[0, 0] == 2
        assert env.foods(0).tolist() == [[5, 4, 1]]

    def test_starvation(self):
        env = make_env(1)
        env.set_state(0, [[(5, 5)]], [N], health=[1])
        step(env, STRAIGHT)
        assert death(env, 0) == "starvation"

    def test_eating_refills_health_on_the_last_step(self):
        env = make_env(1, max_health=60)
        env.set_state(0, [[(5, 5)]], [N], foods=[(5, 4, 0)], health=[1])
        step(env, STRAIGHT)
        assert env.alive[0, 0] and env.health[0, 0] == 60

    def test_health_drops_one_per_step(self):
        env = make_env(1, max_health=60)
        env.set_state(0, [[(5, 9)]], [N])
        for _ in range(5):
            step(env, STRAIGHT)
        assert env.health[0, 0] == 55

    def test_food_respawns_on_a_free_cell(self):
        env = SquadEnv(SquadConfig(respawn_food=True))
        env.reset(range(16))
        policy = OraclePolicy()
        policy.reset(range(16))
        obs = env.observe()
        for _ in range(100):
            obs, _, _, _ = env.step(policy.act(obs))
            for b in range(16):
                foods = env.foods(b)
                assert len(foods) == 6
                assert len({(x, y) for x, y, _ in foods.tolist()}) == 6
        assert env.score.sum() > 0


class TestEpisode:
    def test_max_steps_and_zero_rewards_after_done(self):
        env = SquadEnv(SquadConfig(max_steps=3, survival_reward=0.5))
        env.reset([0, 1])
        for _t in range(3):
            _, rewards, done, _ = env.step(np.zeros((2, 3), dtype=np.int64))
        assert done.all() and (env.steps == 3).all()
        _, rewards, done, _ = env.step(np.zeros((2, 3), dtype=np.int64))
        assert (rewards == 0).all() and (env.steps == 3).all()

    def test_reset_layout(self):
        env = SquadEnv()
        env.reset(range(32))
        for b in range(32):
            bodies = env.bodies(b)
            assert all(len(body) == 3 for body in bodies)
            cells = [tuple(c) for body in bodies for c in body.tolist()]
            assert len(set(cells)) == 9
            foods = env.foods(b)
            assert sorted(foods[:, 2].tolist()) == [0, 0, 1, 1, 2, 2]
            assert not {(x, y) for x, y, _ in foods.tolist()} & set(cells)
        assert (env.health == 60).all() and env.alive.all()

    def test_determinism_and_batch_independence(self):
        def rollout(seeds):
            env = SquadEnv()
            r = run_episodes(env, GreedyPolicy(), HandCodedCentre(), seeds)
            return r["food"], r["steps"], env.grid.copy()

        a = rollout([3, 7, 11])
        b = rollout([3, 7, 11])
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        single = rollout([7])
        np.testing.assert_array_equal(single[0][0], a[0][1])
        np.testing.assert_array_equal(single[2][0], a[2][1])

    def test_different_seeds_differ(self):
        env = SquadEnv()
        env.reset([1, 2])
        assert not np.array_equal(env.grid[0], env.grid[1])

    def test_step_before_reset_raises(self):
        with pytest.raises(RuntimeError):
            SquadEnv().step(np.zeros((1, 3), dtype=np.int64))

    @pytest.mark.parametrize(
        "bad", [np.zeros((1, 2)), np.full((1, 3), 3), np.zeros((1, 3)) + 0.5]
    )
    def test_invalid_actions(self, bad):
        env = SquadEnv()
        env.reset([0])
        with pytest.raises(ValueError):
            env.step(bad)


class TestObservations:
    @pytest.mark.parametrize("radius", [3, 5, 7])
    def test_shapes_and_dtypes_match_spec(self, radius):
        env = SquadEnv(SquadConfig(obs_radius=radius))
        obs = env.reset(range(4))
        spec = env.spec
        assert spec.n_specialists == 3 and spec.n_actions == 3 and spec.max_steps == 300
        for name, (shape, dtype) in spec.specialist_obs.items():
            arr = obs.specialists[name]
            assert arr.shape == (4, 3) + shape and arr.dtype == dtype, name
        for name, (shape, dtype) in spec.centre_obs.items():
            arr = obs.centre[name]
            assert arr.shape == (4,) + shape and arr.dtype == dtype, name
        assert obs.specialists["window"].shape[-1] == 2 * radius + 1
        assert obs.specialists["message"].dtype == np.int64

    def test_continuous_message_obs(self):
        env = SquadEnv(SquadConfig(message=MessageSpec("continuous", dim=4)))
        obs = env.reset([0, 1])
        assert obs.specialists["message"].shape == (2, 3, 4)
        assert obs.specialists["message"].dtype == np.float32
        msgs = np.ones((2, 3, 4), dtype=np.float32)
        obs, _, _, _ = env.step(np.zeros((2, 3), dtype=np.int64), msgs)
        np.testing.assert_array_equal(obs.specialists["message"], msgs)

    def test_messages_arrive_next_observation_every_k_steps(self):
        env = SquadEnv(SquadConfig(message=MessageSpec("discrete", 2, 8, interval=2)))
        obs = env.reset([0])
        assert (obs.specialists["message"] == 0).all()
        actions = np.zeros((1, 3), dtype=np.int64)
        for t, expect in [(0, 5), (1, 5), (2, 7)]:
            msgs = np.full((1, 3, 2), 5 if t < 2 else 7, dtype=np.int64)
            if t == 1:
                msgs[:] = 6  # step 1 is not a send step: ignored
            obs, _, _, _ = env.step(actions, msgs)
            assert (obs.specialists["message"] == expect).all(), t
        obs, _, _, _ = env.step(actions, None)  # silence keeps the last payload
        assert (obs.specialists["message"] == 7).all()

    def test_invalid_messages(self):
        env = SquadEnv()
        env.reset([0])
        with pytest.raises(ValueError):
            env.step(np.zeros((1, 3), dtype=np.int64), np.full((1, 3, 1), 8))
        with pytest.raises(ValueError):
            env.step(
                np.zeros((1, 3), dtype=np.int64), np.zeros((1, 2, 1), dtype=np.int64)
            )

    @pytest.mark.parametrize("heading", [N, E, S, W])
    def test_window_is_egocentric(self, heading):
        env = make_env(2, obs_radius=3)
        dx, dy = engine.DIR_DX[heading], engine.DIR_DY[heading]
        rx, ry = engine.DIR_DX[(heading + 1) % 4], engine.DIR_DY[(heading + 1) % 4]
        hx, hy = 5, 5
        neck = (hx - dx, hy - dy)
        foods = [(hx + 2 * dx, hy + 2 * dy, 0), (hx + 2 * rx, hy + 2 * ry, 1)]
        obs = env.set_state(0, [[(hx, hy), neck], [(1, 1)]], [heading, N], foods=foods)
        win = obs.specialists["window"][0, 0]
        r = 3
        assert win[0, r, r] == 1 and win[0, r + 1, r] == 1  # head, neck behind
        assert win[2, r - 2, r] == 1  # own food two ahead
        assert win[3, r, r + 2] == 1  # other food two to the right
        assert win[2].sum() == 1 and win[3].sum() == 1

    def test_off_board_reads_as_wall(self):
        env = make_env(2, obs_radius=3)
        obs = env.set_state(0, [[(0, 0)], [(8, 8)]], [N, N])
        win = obs.specialists["window"][0, 0]
        assert win[1, :3, :].all() and win[1, :, :3].all()  # above and left
        assert win[1, 3 + 1 :, 3 + 1 :].sum() == 0
        assert win[1].sum() == 49 - 16  # only the 4x4 on-board corner is clear

    def test_centre_board(self):
        env = make_env(2)
        obs = env.set_state(
            0,
            [[(5, 5), (5, 6)], []],
            [N, N],
            foods=[(1, 1, 0), (2, 2, 1)],
            health=[30, 0],
        )
        board = obs.centre["board"][0]
        assert board[0].sum() == 2 and board[0, 6, 5] == 1 and board[1].sum() == 0
        assert board[2, 5, 5] == 1 and board[3].sum() == 0
        assert board[4, 1, 1] == 1 and board[5, 2, 2] == 1
        assert obs.centre["health"][0].tolist() == [0.5, 0.0]
        assert obs.centre["heads"][0].tolist() == [[5, 5], [-1, -1]]
        assert obs.specialists["window"][0, 1].sum() == 0  # dead snake sees nothing


class TestSingleSnakeEquivalence:
    """N=1, walls, one colour: the squad engine must match SnakeGame move for move.

    The single-snake engine treats the tail as solid, so the driving policy never
    steps onto any body cell (the only place the two rule sets differ).
    """

    @staticmethod
    def _policy(game, rng):
        head = game.snake_positions[0]
        body = {tuple(p) for p in game.snake_positions.tolist()}
        best, best_dist = None, None
        for action in rng.permutation(3):
            d = (int(game.snake_direction) + (0, -1, 1)[action]) % 4
            x, y = head[0] + engine.DIR_DX[d], head[1] + engine.DIR_DY[d]
            if not (0 <= x < game.board_size and 0 <= y < game.board_size):
                continue
            if (x, y) in body:
                continue
            dist = abs(game.food_position.x - x) + abs(game.food_position.y - y)
            if best is None or dist < best_dist:
                best, best_dist = int(action), dist
        return 0 if best is None else best

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_snake_game(self, seed):
        size = 21
        game = SnakeGame(
            board_size=size,
            walls=True,
            starvation_steps=60,
            max_steps=10_000,
            seed=seed,
        )
        env = make_env(1, board_size=size, max_health=60, init_length=1, max_steps=300)
        center = size // 2
        food = game.food_position
        env.set_state(0, [[(center, center)]], [N], foods=[(food.x, food.y, 0)])
        rng = np.random.default_rng(seed)
        for t in range(300):
            action = self._policy(game, rng) if t < 250 else STRAIGHT  # then crash
            state, _, done = game.step(action)
            _, _, squad_done, _ = step(env, action)
            assert env.alive[0, 0] == state.is_alive, t
            if not state.is_alive:
                assert death(env, 0) == {"wall": "wall", "collision": "body"}.get(
                    state.end_reason, state.end_reason
                )
                break
            assert env.score[0, 0] == state.score, t
            np.testing.assert_array_equal(env.bodies(0)[0], state.snake_positions)
            if env.foods(0).size == 0:  # eaten: copy the single engine's new food
                food = state.food_position
                env.set_state(
                    0,
                    [env.bodies(0)[0].tolist()],
                    [int(state.snake_direction)],
                    foods=[(food.x, food.y, 0)],
                    health=[int(env.health[0, 0])],
                    steps=int(env.steps[0]),
                )
                env.score[0, 0] = state.score
        assert not env.alive[0, 0] and game.score >= 5

    def test_starves_like_snake_game(self):
        game = SnakeGame(board_size=21, walls=True, starvation_steps=60, seed=0)
        env = make_env(1, board_size=21, max_health=60, init_length=1)
        env.set_state(0, [[(10, 10)]], [N])  # no food at all
        for _t in range(61):
            state, _, _ = game.step(RIGHT)  # spin on the spot
            step(env, RIGHT)
            if not state.is_alive:
                break
        assert state.end_reason == "starvation" and death(env, 0) == "starvation"
        assert env.steps[0] == state.steps == 60


class TestPolicies:
    def test_hand_coded_codes(self):
        env = make_env(2, obs_radius=3)
        obs = env.set_state(
            0,
            [[(5, 5), (5, 6)], [(1, 5), (1, 6)]],
            [N, N],
            foods=[(5, 0, 0), (3, 5, 1)],
        )
        msgs = HandCodedCentre(near_distance=3).send(obs)
        assert msgs.shape == (1, 2, 1) and msgs.dtype == np.int64
        assert msgs[0, 0, 0] == 2 * 0 + 1  # ahead, far (5 > 3)
        assert msgs[0, 1, 0] == 2 * 1 + 0  # right, near
        obs = env.set_state(0, [[(5, 5), (4, 5)], [(1, 5)]], [E, N], foods=[(5, 8, 0)])
        assert (
            HandCodedCentre(near_distance=3).send(obs)[0, 0, 0] == 2 * 1
        )  # right of E

    def test_centres_respect_message_conventions(self):
        env = SquadEnv()
        obs = env.reset(range(5))
        for centre in (HandCodedCentre(), RandomCentre()):
            centre.reset(range(5))
            msgs = centre.send(obs)
            assert msgs.shape == (5, 3, 1) and msgs.dtype == np.int64
            assert msgs.min() >= 0 and msgs.max() < 8
        silent = SilentCentre().send(obs)
        assert silent.shape == (5, 3, 1) and silent.dtype == np.int64
        assert not silent.any()

    def test_silent_centre_matches_never_sending(self):
        class Mute(SilentCentre):
            def send(self, obs):
                return None

        seeds = list(range(16))
        sent = run_episodes(SquadEnv(), GreedyPolicy(), SilentCentre(), seeds)
        mute = run_episodes(SquadEnv(), GreedyPolicy(), Mute(), seeds)
        assert np.array_equal(sent["team_food"], mute["team_food"])
        assert np.array_equal(sent["steps"], mute["steps"])

    @pytest.mark.parametrize(
        "policy",
        [GreedyPolicy(), NoCommPolicy(), RandomPolicy(), OraclePolicy()],
        ids=lambda p: p.name,
    )
    def test_probs_are_the_sampling_distribution(self, policy):
        env = SquadEnv(SquadConfig(obs_radius=3))
        obs = env.reset(range(6))
        gen = np.random.default_rng(0)
        for _ in range(15):  # Move on to a varied state with messages in flight
            obs = env.step(
                gen.integers(0, 3, size=(6, 3)), gen.integers(0, 8, size=(6, 3, 1))
            ).obs
        probs = policy.probs(obs)
        assert probs.shape == (6, 3, 3)
        assert np.allclose(probs.sum(axis=-1), 1.0) and probs.min() >= 0
        draws = 3000
        counts = np.zeros_like(probs)
        rows, cols = np.indices((6, 3))
        for k in range(draws // 6):
            policy.reset(range(k * 6, k * 6 + 6))
            for _ in range(6):
                counts[rows, cols, policy.act(obs)] += 1
        freq = counts / counts.sum(axis=-1, keepdims=True)
        assert np.abs(freq - probs).max() < 0.05
        # Never samples a zero-probability action
        assert np.all(counts[probs == 0] == 0)

    def test_greedy_probs_follow_the_message(self):
        env = make_env(2, obs_radius=3)
        obs = env.set_state(0, [[(5, 5), (5, 6)], [(1, 9)]], [N, N])
        policy = GreedyPolicy(explore=0.0)
        for code, action in ((0, STRAIGHT), (2, RIGHT), (6, LEFT)):
            msgs = np.full((1, 2, 1), code, dtype=np.int64)
            view = type(obs)({**obs.specialists, "message": msgs}, obs.centre)
            assert policy.probs(view)[0, 0, action] == 1.0
        deaf = NoCommPolicy(explore=0.3).probs(obs)[0, 0]
        assert deaf[STRAIGHT] == pytest.approx(0.7 + 0.3 / 3)

    def test_specialists_return_valid_actions(self):
        env = SquadEnv()
        obs = env.reset(range(5))
        for policy in (GreedyPolicy(), NoCommPolicy(), RandomPolicy(), OraclePolicy()):
            policy.reset(range(5))
            actions = policy.act(obs)
            assert actions.shape == (5, 3) and actions.dtype == np.int64
            assert actions.min() >= 0 and actions.max() < 3

    def test_greedy_takes_visible_food_and_avoids_walls(self):
        env = make_env(2, obs_radius=3)
        obs = env.set_state(
            0, [[(5, 5), (5, 6)], [(0, 3), (0, 4)]], [N, N], foods=[(6, 5, 0)]
        )
        policy = NoCommPolicy(explore=0.0)
        policy.reset([0])
        assert policy.act(obs)[0, 0] == RIGHT
        obs = env.set_state(0, [[(5, 0), (5, 1)], [(1, 1)]], [N, N])
        assert policy.act(obs)[0, 0] in (LEFT, RIGHT)

    def test_run_episodes_hook_gets_live_mask_and_observer_sees_sent(self):
        seeds = list(range(8))
        calls, seen = [], []

        def hook(messages, step, mask):
            calls.append((step, mask.copy()))
            return np.zeros_like(messages)

        out = run_episodes(
            SquadEnv(SquadConfig(max_steps=80)),
            GreedyPolicy(),
            HandCodedCentre(),
            seeds,
            hook=hook,
            observer=lambda step, obs, sent, done: seen.append(sent.copy()),
        )
        silent = run_episodes(
            SquadEnv(SquadConfig(max_steps=80)), GreedyPolicy(), SilentCentre(), seeds
        )
        assert np.array_equal(out["team_food"], silent["team_food"])
        assert [s for s, _ in calls] == list(range(len(calls)))
        assert calls[0][1].shape == (8, 3) and calls[0][1].all()
        assert len(seen) == len(calls) and any(m.any() for m in seen)

    def test_hand_coded_messages_help(self):
        seeds = list(range(32))
        results = {
            name: run_episodes(SquadEnv(), GreedyPolicy(), centre, seeds)[
                "team_food"
            ].mean()
            for name, centre in [
                ("hand", HandCodedCentre()),
                ("random", RandomCentre()),
            ]
        }
        oracle = run_episodes(SquadEnv(), OraclePolicy(), None, seeds)[
            "team_food"
        ].mean()
        assert oracle > results["hand"] > results["random"]

    @pytest.mark.slow
    def test_need_to_talk_ordering(self):
        seeds = list(range(2_000_000_000, 2_000_000_100))
        env = SquadEnv()
        food = {
            name: run_episodes(env, spec, centre, seeds)["team_food"].mean()
            for name, spec, centre in [
                ("hand", GreedyPolicy(), HandCodedCentre()),
                ("nocomm", NoCommPolicy(), None),
                ("random", GreedyPolicy(), RandomCentre()),
                ("oracle", OraclePolicy(), None),
                ("chance", RandomPolicy(), None),
            ]
        }
        assert (
            food["oracle"]
            > food["hand"]
            > food["nocomm"]
            > food["random"]
            > food["chance"]
        )
