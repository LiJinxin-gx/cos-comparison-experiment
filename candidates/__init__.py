"""
candidates -- immature but principle-conforming results tracked publicly.

Each module here is a "candidate" that may break a known bottleneck. It carries
a distributed DEPENDENCIES block and runs independently. See README.md for the
status registry (evidence, open questions, what bottleneck each targets).
"""
DEPENDENCIES = {
    "python": ">=3.8",
    "requires": {"numpy": ">=1.20"},
    "optional": {"cos_comparison": "~=0.5"},
    "backends": ["formal", "numpy"],
    "status": "candidate",
    "targets": "public tracking of promising, immature results",
}
