"""
Continuous mapping hierarchical isolation.

Core mechanism:
  - Low levels store raw / fine-grained templates.
  - High levels store abstract / structural templates.
  - Irrelevant detail at low levels is ISOLATED and does NOT propagate up
    (on-demand extraction).
  - High levels can DRIVE low levels: high-level context guides which
    low-level details to extract (no detail is permanently lost).

This is the foundational principle of the project. This module provides a
generic hierarchical representation that any algorithm can build on.
"""
from .cosine import cosine, as_tensor, flatten
from .multiscale import downsample


class HierarchicalRepresentation:
    """Multi-level tensor representation with isolation and top-down drive.

    Levels are ordered from fine (level 0) to coarse (level N).
    Each level stores:
      - representation: the tensor at this abstraction level
      - templates: learned patterns at this level
      - mask: which positions are "active" (not isolated)
    """

    def __init__(self, tensor, n_levels=4, downsample_factor=2):
        """Build hierarchical representation from a base tensor.

        Level 0 = original (finest). Level k = downsampled k times.
        """
        self.n_levels = n_levels
        self.levels = []
        current = as_tensor(tensor)
        for lv in range(n_levels):
            self.levels.append({
                "representation": flatten(current) if lv > 0 else flatten(current),
                "raw": current,
                "templates": [],
                "mask": [1] * len(flatten(current)),
            })
            if lv < n_levels - 1:
                current = downsample(current, downsample_factor)

    def get_level(self, level):
        """Get representation at a given level."""
        return self.levels[level]["representation"]

    def isolate(self, level, indices):
        """Mark positions at a level as isolated (not propagating upward).

        This is the "isolation" part: low-level detail that is deemed
        irrelevant is masked out.
        """
        for idx in indices:
            if 0 <= idx < len(self.levels[level]["mask"]):
                self.levels[level]["mask"][idx] = 0

    def drive(self, high_level, low_level, template_indices):
        """Top-down drive: high-level templates activate low-level positions.

        Given template indices at high_level, find corresponding regions
        at low_level and unmask them (detail recovery).
        """
        # Simplified: high-level template positions map to low-level regions
        factor = 2 ** (high_level - low_level)
        high_rep = self.levels[high_level]["representation"]
        low_mask = self.levels[low_level]["mask"]
        for ti in template_indices:
            if 0 <= ti < len(high_rep):
                base = ti * factor
                for j in range(factor):
                    idx = base + j
                    if 0 <= idx < len(low_mask):
                        low_mask[idx] = 1

    def add_template(self, level, template, label=None):
        """Add a learned template at a specific level."""
        self.levels[level]["templates"].append({
            "data": as_tensor(template),
            "label": label,
        })

    def match_templates(self, level, tensor=None):
        """Match a tensor (or this level's representation) against templates.

        Returns list of (label, similarity) sorted descending.
        """
        rep = flatten(as_tensor(tensor)) if tensor is not None else self.levels[level]["representation"]
        results = []
        for tpl in self.levels[level]["templates"]:
            sim = cosine(rep, flatten(tpl["data"]))
            results.append((tpl["label"], sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def active_representation(self, level):
        """Get representation with isolation mask applied."""
        rep = self.levels[level]["representation"]
        mask = self.levels[level]["mask"]
        return [rep[i] * mask[i] for i in range(len(rep))]

    def info_loss_report(self):
        """Report how much information is isolated at each level."""
        report = []
        for lv in range(self.n_levels):
            mask = self.levels[lv]["mask"]
            active = sum(mask)
            total = len(mask)
            report.append({
                "level": lv,
                "size": total,
                "active": active,
                "isolated": total - active,
                "isolation_ratio": (total - active) / total if total > 0 else 0.0,
            })
        return report
