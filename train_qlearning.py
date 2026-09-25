"""
Q-learning для Pac-Man на своей сетке.

Игра pygame и спрайты — репозиторий автора usawa:
https://github.com/usawa/pypacman (GPLv3).
Здесь только среда, награды и обучение. Этот файл не содержит кода usawa.
"""

import json
from pathlib import Path

import numpy as np

MAZE = [
    "###############",
    "#.............#",
    "#.###.###.###.#",
    "#o...........o#",
    "#.###.#.#.###.#",
    "#.....#.#.....#",
    "#####.#.#.#####",
    "#.....#.#.....#",
    "#.###.#.#.###.#",
    "#o...........o#",
    "#.###.###.###.#",
    "#.............#",
    "###############",
]
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # up down left right
HEIGHT = len(MAZE)
WIDTH = len(MAZE[0])


def walls():
    return {(x, y) for y, row in enumerate(MAZE) for x, cell in enumerate(row) if cell == "#"}


WALLS = walls()


def xy_state(x1, y1, x2, y2):
    dx = int(np.sign(x1 - x2))
    dy = int(np.sign(y1 - y2))
    n = 3 * dx + dy + 4
    if n > 4:
        n -= 1
    return n


def legal_moves(pos):
    x, y = pos
    moves = []
    for i, (dx, dy) in enumerate(DIRS):
        nxt = (x + dx, y + dy)
        if nxt not in WALLS and 0 <= nxt[0] < WIDTH and 0 <= nxt[1] < HEIGHT:
            moves.append(i)
    return moves


class Game:
    def __init__(self, n_ghosts=1, rng=None):
        self.n_ghosts = n_ghosts
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.pellets = {(x, y) for y, row in enumerate(MAZE) for x, cell in enumerate(row) if cell == "."}
        self.power = {(x, y) for y, row in enumerate(MAZE) for x, cell in enumerate(row) if cell == "o"}
        self.pacman = (1, 11)
        self.ghosts = [(7, 6)]
        spots = [(7, 5), (6, 6), (8, 6), (7, 7)]
        for i in range(self.n_ghosts - 1):
            self.ghosts.append(spots[i % len(spots)])
        self.frightened = 0
        self.jailed = []
        self.score = 0
        self.steps = 0
        self.done = False
        self.win = False
        return self.state()

    def nearest(self, points):
        if not points:
            return self.pacman
        px, py = self.pacman
        return min(points, key=lambda p: abs(p[0] - px) + abs(p[1] - py))

    def state(self):
        ghost = self.nearest(self.ghosts)
        pellet = self.nearest(self.pellets | self.power)
        g_dir = xy_state(self.pacman[0], self.pacman[1], ghost[0], ghost[1])
        p_dir = xy_state(self.pacman[0], self.pacman[1], pellet[0], pellet[1])
        dist = abs(ghost[0] - self.pacman[0]) + abs(ghost[1] - self.pacman[1])
        if dist <= 2:
            bucket = 0
        elif dist <= 5:
            bucket = 1
        else:
            bucket = 2
        return (g_dir * 8 + p_dir) * 3 + bucket

    def step(self, action):
        if self.done:
            return self.state(), 0.0, True
        self.steps += 1
        reward = -0.2
        moves = legal_moves(self.pacman)
        before = self.nearest(self.pellets | self.power)
        before_dist = abs(before[0] - self.pacman[0]) + abs(before[1] - self.pacman[1])
        if action in moves:
            dx, dy = DIRS[action]
            self.pacman = (self.pacman[0] + dx, self.pacman[1] + dy)
        else:
            reward -= 1.0
        after = self.nearest(self.pellets | self.power)
        after_dist = abs(after[0] - self.pacman[0]) + abs(after[1] - self.pacman[1])
        reward += 0.5 * (before_dist - after_dist)

        if self.pacman in self.pellets:
            self.pellets.remove(self.pacman)
            self.score += 10
            reward += 10
        if self.pacman in self.power:
            self.power.remove(self.pacman)
            self.frightened = 16
            self.score += 20
            reward += 20

        ghost_moves = (self.frightened and self.steps % 3 == 0) or (not self.frightened and self.steps % 2 == 0)
        if ghost_moves:
            self._move_ghosts()
        if self.frightened:
            self.frightened -= 1
            if self.frightened == 0 and self.jailed:
                self.ghosts.extend(self.jailed)
                self.jailed = []

        eaten = [g for g in self.ghosts if g == self.pacman]
        if eaten and self.frightened:
            reward += 40 * len(eaten)
            self.score += 40 * len(eaten)
            self.ghosts = [g for g in self.ghosts if g != self.pacman]
            self.jailed.extend([(7, 6) for _ in eaten])
        elif eaten:
            reward -= 80
            self.done = True
            return self.state(), reward, True

        if not self.pellets and not self.power:
            reward += 80
            self.score += 80
            self.win = True
            self.done = True
        elif self.steps >= 400:
            self.done = True
        return self.state(), reward, self.done

    def _move_ghosts(self):
        moved = []
        for ghost in self.ghosts:
            options = []
            for i in legal_moves(ghost):
                dx, dy = DIRS[i]
                nxt = (ghost[0] + dx, ghost[1] + dy)
                dist = abs(nxt[0] - self.pacman[0]) + abs(nxt[1] - self.pacman[1])
                options.append((dist, nxt))
            if not options:
                moved.append(ghost)
                continue
            if self.rng.random() < 0.25:
                moved.append(tuple(self.rng.choice([pos for _, pos in options])))
                continue
            options.sort(key=lambda item: item[0], reverse=bool(self.frightened))
            best = options[0][0]
            pool = [pos for dist, pos in options if dist == best]
            moved.append(tuple(self.rng.choice(pool)))
        self.ghosts = moved


N_STATES = 8 * 8 * 3
N_ACTIONS = 4


def train(episodes=700, n_ghosts=1, seed=0):
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, N_ACTIONS))
    alpha, gamma = 0.25, 0.95
    epsilon = 1.0
    history = []
    for ep in range(episodes):
        ghosts = n_ghosts
        game = Game(ghosts, np.random.default_rng(rng.integers(1_000_000_000)))
        state = game.reset()
        total = 0.0
        for _ in range(400):
            moves = legal_moves(game.pacman)
            if rng.random() < epsilon:
                action = int(rng.choice(moves))
            else:
                scores = q[state].copy()
                mask = np.full(N_ACTIONS, -1e9)
                mask[moves] = scores[moves]
                action = int(np.argmax(mask))
            nxt, reward, done = game.step(action)
            future = 0.0 if done else np.max(q[nxt])
            q[state, action] += alpha * (reward + gamma * future - q[state, action])
            state = nxt
            total += reward
            if done:
                break
        epsilon = max(0.05, epsilon * 0.995)
        history.append({"episode": ep + 1, "ghosts": ghosts, "score": game.score, "win": game.win, "return": round(total, 2)})
    return q, history


def evaluate(q, n_ghosts, episodes=40, seed=1):
    rng = np.random.default_rng(seed)
    wins = 0
    scores = []
    steps = []
    left = []
    for i in range(episodes):
        game = Game(n_ghosts, np.random.default_rng(rng.integers(1_000_000_000)))
        state = game.reset()
        done = False
        while not done:
            moves = legal_moves(game.pacman)
            scores_q = q[state].copy()
            mask = np.full(N_ACTIONS, -1e9)
            mask[moves] = scores_q[moves]
            state, _, done = game.step(int(np.argmax(mask)))
        wins += int(game.win)
        scores.append(game.score)
        steps.append(game.steps)
        left.append(len(game.pellets) + len(game.power))
    return {
        "ghosts": n_ghosts,
        "episodes": episodes,
        "win_rate": round(wins / episodes, 4),
        "mean_score": round(float(np.mean(scores)), 2),
        "mean_steps": round(float(np.mean(steps)), 2),
        "mean_pellets_left": round(float(np.mean(left)), 2),
    }


def main():
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    tables = {}
    evals = []
    last_wins = {}
    for ghosts in (1, 2, 3):
        q, history = train(episodes=700, n_ghosts=ghosts, seed=7 + ghosts)
        tables[ghosts] = q
        np.save(out / f"q_table_{ghosts}.npy", q)
        evals.append(evaluate(q, ghosts))
        last_wins[str(ghosts)] = round(float(np.mean([row["win"] for row in history[-100:]])), 4)
    summary = {
        "states": N_STATES,
        "actions": N_ACTIONS,
        "train_episodes_per_policy": 700,
        "train_win_rate_last_100": last_wins,
        "evaluation": evals,
        "upstream_game": "https://github.com/usawa/pypacman",
    }
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
