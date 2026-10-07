"""
Local k-NN model: the simplest rigorous non-parametric learner (v53 result).

Core principle - all knowledge is LOCAL:
  * A state is matched only to its nearest stored prototypes.
  * The predicted effect of an action is the UNWEIGHTED MEAN of the k nearest
    prototypes that have ACTUALLY experienced that action.

Pure Python, nested-list tensors, no numpy, no recursion.
"""


def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class LocalKNN:
    """A local, incremental, zero-forgetting dynamics learner."""

    def __init__(self, state_tol=0.18, k=6):
        self.state_tol = state_tol
        self.k = k
        self.proto = []
        self.eff = {}

    def absorb(self, state, action, next_state):
        delta = [ns - s for s, ns in zip(state, next_state)]
        if not self.proto:
            self.proto.append(list(state))
            self.eff[0] = {action: (delta, 1)}
            return
        best_i, best_d = 0, None
        for i, p in enumerate(self.proto):
            d = _dist(state, p)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_d > self.state_tol:
            best_i = len(self.proto)
            self.proto.append(list(state))
            self.eff[best_i] = {}
        cur = self.eff[best_i].get(action)
        if cur is None:
            self.eff[best_i][action] = (delta, 1)
        else:
            mean, n = cur
            self.eff[best_i][action] = (
                [(m * n + d) / (n + 1) for m, d in zip(mean, delta)], n + 1)

    def predict(self, state, action):
        near = []
        for i, p in enumerate(self.proto):
            rec = self.eff[i].get(action)
            if rec is not None:
                near.append((_dist(state, p), rec[0]))
        if not near:
            return None
        near.sort(key=lambda x: x[0])
        k = min(self.k, len(near))
        dims = len(near[0][1])
        out = [0.0] * dims
        for _, mean_delta in near[:k]:
            for d in range(dims):
                out[d] += mean_delta[d]
        return [v / k for v in out]

    def __len__(self):
        return len(self.proto)
