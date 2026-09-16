# Local k-NN Model (v53)

The simplest rigorous non-parametric learner validated in the exploration
notebook: knowledge is stored as **local prototypes**, and prediction is just
the **average of the k nearest local experiences**.

## The idea

* A `state` is never encoded into a global function. It is matched only to the
  nearest stored prototypes.
* For an `action`, its predicted effect is the **unweighted mean** of the `k`
  nearest prototypes that have actually performed that action.
* That is the k-nearest-neighbour estimate of the local conditional mean
  `E[delta | state, action]` - the simplest local regression.

## Why it is rigorous

* **Consistency**: as prototypes grow denser, the k-nearest neighbourhood
  shrinks in physical size and the estimate converges to the true local mean.
* **Locality**: only the k nearest prototypes are ever consulted - no global
  operation, no global temperature, no shared weights.
* **Zero-forgetting**: absorbing a transition updates (or creates) ONE
  prototype and never touches the others. Learning a new region cannot erase
  an old one (stability-plasticity, for free).
* **One knob**: only the neighbourhood size `k`. No kernel shape, no bandwidth,
  no learning rate, no architecture to tune.
* **No deep learning**: no gradient, no back-propagation, no weight matrix.
  Knowledge stays as inspectable, editable local records.
* **No recursion**: every loop is iterative.

## Continuous mapping

Prototypes are discrete anchors; `absorb` and `predict` are the two directions
of the continuous mapping between the continuous state space and the discrete
prototype memory (continuous mapping hierarchical isolation, kept strictly
local).

## Files

* `core/localknn.py` - the `LocalKNN` learner (pure Python, nested lists).
* `examples/example_local_model.py` - runnable demo: goal-free learning,
  greedy navigation, and a zero-forgetting check.
