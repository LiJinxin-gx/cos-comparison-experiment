"""
Local k-NN model: the simplest rigorous non-parametric learner (v53 result).

Core principle - all knowledge is LOCAL:
  * A state is matched only to its nearest stored prototypes.
  * The predicted effect of an action is the UNWEIGHTED MEAN of the k nearest
    prototypes that have ACTUALLY experienced that action.

This is the k-nearest-neighbour estimate of the local conditional mean
E[delta | state, action] - the simplest local regression. As stored prototypes
grow denser, the k-nearest neighbourhood shrinks in physical size and the
estimate converges to the true local mean (standard consistency). No weights,
no gradient, no global function.

Zero-forgetting: absorbing one transition updates the running mean of a SINGLE
prototype (or adds a new prototype) - it never touches other prototypes. So
learning a new local region cannot erase an old one.

Continuous mapping: prototypes are discrete anchors; absorb/predict is the
A<->B bridge between the continuous state space and the discrete prototype
memory (continuous mapping hierarchical isolation, kept local).

Pure Python, nested-list tensors, no numpy, no recursion.
"""


def _dist(a, b):
    """Euclidean distance between two equal-length vectors."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class LocalKNN:
    """A local, incremental, zero-forgetting dynamics learner."""

    def __init__(self, state_tol=0.18, k=6):
        self.state_tol = state_tol   # merge radius: closer = same local prototype
        self.k = k                    # neighbourhood size (only knob)
        self.proto = []               # list of states, each a list[float]
        # proto_idx -> {action_index: (mean_delta(list[float]), count)}
        self.eff = {}

    def absorb(self, state, action, next_state):
        """Record one real transition: state --action--> next_state.

        Quantise state to the nearest prototype (merge if within tolerance,
        else open a new local prototype), then update that prototype's running
        mean delta for this action. Writes ONE prototype only.
        """
        delta = [ns - s for s, ns in zip(state, next_state)]
        if not self.proto:
            self.proto.append(list(state))
            self.eff[0] = {action: (delta, 1)}
            return

        # nearest prototype (iterative, no recursion)
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
        """Mean effect of `action` at `state`.

        Average the k nearest prototypes that have experienced `action`.
        Returns None if no prototype knows this action. Purely local: only
        the k nearest prototypes are consulted.
        """
        near = []
        for i, p in enumerate(self.proto):
            rec = self.eff[i].get(action)
            if rec is not None:
                near.append((_dist(state, p), rec[0]))
        if not near:
            return None
        near.sort(key=lambda x: x[0])          # ascending distance
        k = min(self.k, len(near))
        dims = len(near[0][1])
        out = [0.0] * dims
        for _, mean_delta in near[:k]:         # unweighted mean of k nearest
            for d in range(dims):
                out[d] += mean_delta[d]
        return [v / k for v in out]

    def __len__(self):
        return len(self.proto)
