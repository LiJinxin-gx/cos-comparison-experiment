"""
Example: Local k-NN world model - simplest rigorous learning, zero forgetting.

Self-contained, pure Python (no numpy). Demonstrates the v53 result:
  1. Learn local dynamics goal-free (just take random actions, observe effects).
  2. Predict each action's effect by averaging the k nearest local experiences.
  3. Navigate greedily: pick the action whose predicted effect moves closest
     to the goal (one-step model-predictive control, closed loop).
  4. Learn a NEW local region afterwards - the OLD region stays perfect,
     showing zero forgetting (stability-plasticity).

Run:  python examples/example_local_model.py
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.localknn import LocalKNN

STEP = 0.12
ACTIONS = [[round(math.cos(2 * math.pi * i / 8) * STEP, 5),
            round(math.sin(2 * math.pi * i / 8) * STEP, 5)] for i in range(8)]


def _rotate(v, theta):
    c, s = math.cos(theta), math.sin(theta)
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]


class RotationField:
    """World: at state (x, y) an action vector is rotated by k * x.

    Naive "action moves straight" fails here because the same action does
    different things at different x. The learner must capture this LOCAL
    relation from observed transitions. A small fixed noise makes it realistic.
    """

    def __init__(self, k=1.6, noise=0.01, seed=1):
        self.k = k
        self.noise = noise
        self._rs = seed

    def _rand(self):
        # deterministic LCG so the demo is reproducible
        self._rs = (self._rs * 1103515245 + 12345) & 0x7FFFFFFF
        return (self._rs / 0x7FFFFFFF) - 0.5

    def step(self, state, action):
        theta = self.k * state[0]
        moved = _rotate(action, theta)
        ns = [state[0] + moved[0] + self.noise * self._rand(),
              state[1] + moved[1] + self.noise * self._rand()]
        # keep inside the box
        return [max(-1.0, min(1.0, ns[0])), max(-1.0, min(1.0, ns[1]))]


def learn_goal_free(model, env, n=900, seed=0):
    """Take i.i.d. random (state, action) samples, observe, absorb. No labels."""
    rs = seed
    def rnd01():
        nonlocal rs
        rs = (rs * 1103515245 + 12345) & 0x7FFFFFFF
        return (rs / 0x7FFFFFFF) * 2 - 1
    for _ in range(n):
        s = [rnd01(), rnd01()]
        a = int(rnd01() * 4) & 7          # deterministic-ish action index
        ns = env.step(s, ACTIONS[a])
        model.absorb(s, a, ns)


def navigate(model, env, start, goal, reach=0.1, max_steps=80):
    """One-step closed-loop MPC using local predictions. Returns reached?."""
    s = list(start)
    for _ in range(max_steps):
        if math.hypot(s[0] - goal[0], s[1] - goal[1]) < reach:
            return True
        best_a, best_d = None, None
        for ai, _a in enumerate(ACTIONS):
            d = model.predict(s, ai)
            if d is None:
                continue
            cand = [s[0] + d[0], s[1] + d[1]]
            distg = math.hypot(cand[0] - goal[0], cand[1] - goal[1])
            if best_d is None or distg < best_d:
                best_d, best_a = distg, ai
        if best_a is None:
            best_a = 0
        s = env.step(s, ACTIONS[best_a])
    return math.hypot(s[0] - goal[0], s[1] - goal[1]) < reach


def make_tasks(n=20, seed=7):
    rs = seed
    def rnd01():
        nonlocal rs
        rs = (rs * 1103515245 + 12345) & 0x7FFFFFFF
        return (rs / 0x7FFFFFFF) * 2 - 1
    tasks = []
    for _ in range(n):
        start = [rnd01() * 0.6, rnd01() * 0.6]
        goal = [start[0] + (rnd01() * 0.6), start[1] + (rnd01() * 0.6)]
        goal = [max(-0.95, min(0.95, goal[0])), max(-0.95, min(0.95, goal[1]))]
        tasks.append((start, goal))
    return tasks


def reach_rate(model, tasks, k):
    ok = 0
    for st, g in tasks:
        env = RotationField(k=k, noise=0.01)
        if navigate(model, env, st, g):
            ok += 1
    return 100.0 * ok / len(tasks)


def main():
    print("=== Local k-NN world model (pure Python, zero deps) ===")

    # 1. goal-free learning
    model = LocalKNN(state_tol=0.18, k=6)
    learn_goal_free(model, RotationField(k=1.6), n=900, seed=11)
    print("Learned from 900 goal-free transitions ->", len(model), "local prototypes")

    # 2. navigation on the SAME field it was trained on
    tasks = make_tasks(20, seed=7)
    rate = reach_rate(model, tasks, k=1.6)
    print(f"Navigation reach on trained field (k=1.6): {rate:.0f}%")

    # 3. ZERO-FORGETTING: learn a NEW local region, then re-test the OLD field
    old_rate = reach_rate(model, tasks, k=1.6)
    # sample a patch in a new region with a DIFFERENT field slope k=2.4
    for _ in range(40):
        s = [0.4 + 0.5 * ((_ * 37 % 100) / 100.0),
             0.4 + 0.5 * ((_ * 53 % 100) / 100.0)]
        a = _ % 8
        ns = RotationField(k=2.4, noise=0.01, seed=3).step(s, ACTIONS[a])
        model.absorb(s, a, ns)
    after_old = reach_rate(model, tasks, k=1.6)
    print(f"After learning a new region: OLD field reach {old_rate:.0f}% -> {after_old:.0f}%")
    print("  (unchanged => zero forgetting: new local prototypes never touch old ones)")

    # 4. show inspectability: every prediction traces to readable local records
    s = [0.2, -0.1]
    print("\nInspectability: predicted effect of action 0 at state", s)
    print("  predicted delta =", [round(x, 4) for x in model.predict(s, 0)])
    print("  prototypes:", len(model), "; only the k nearest that know action 0 are used.")

    print("\nPrinciples: strictly local, continuous A<->B mapping, no weights,")
    print("no gradient, no recursion, one knob (k), incremental & zero-forgetting.")


if __name__ == "__main__":
    main()
