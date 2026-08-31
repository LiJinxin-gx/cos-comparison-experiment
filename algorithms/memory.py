"""
Hierarchical Memory System.

Applies continuous mapping hierarchical isolation to memory storage:
  - Low abstraction levels: store fine-grained matching templates (raw detail)
  - High abstraction levels: store high-level matching templates (structure)
  - Levels MUTUALLY DRIVE each other:
      * High-level context guides low-level search (what to look for)
      * Low-level detail feeds high-level abstraction (what exists)
  - Search and organization are DECOUPLED but can COLLABORATE:
      * Search can run in parallel across levels
      * Organization consolidates patterns without blocking search

This increases parallel search capability while maintaining hierarchical
isolation (irrelevant detail stays at low levels).
"""
from core.cosine import cosine, as_tensor, flatten
from core.hierarchical import HierarchicalRepresentation


class HierarchicalMemory:
    """Multi-level memory with bidirectional level driving.

    Each level stores:
      - templates: learned patterns at this abstraction level
      - associations: links to related templates at other levels
      - access_count: how often this template has been retrieved
    """

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.levels = [{"templates": [], "associations": {}} for _ in range(n_levels)]
        self.access_log = []

    def store(self, tensor, label=None, level=0):
        """Store a tensor as a template at a specific level.

        Also creates associations: if similar templates exist at adjacent
        levels, link them (bidirectional driving).
        """
        flat = flatten(as_tensor(tensor))
        entry = {
            "data": flat,
            "label": label,
            "access_count": 0,
        }
        self.levels[level]["templates"].append(entry)
        idx = len(self.levels[level]["templates"]) - 1

        # Create associations with adjacent levels
        for adj_level in [level - 1, level + 1]:
            if 0 <= adj_level < self.n_levels:
                best_sim = 0.0
                best_idx = -1
                for ti, tpl in enumerate(self.levels[adj_level]["templates"]):
                    sim = cosine(flat, tpl["data"])
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = ti
                if best_idx >= 0 and best_sim > 0.5:
                    key = (level, idx)
                    if key not in self.levels[level]["associations"]:
                        self.levels[level]["associations"][key] = []
                    self.levels[level]["associations"][key].append(
                        (adj_level, best_idx, best_sim))

        return idx

    def search(self, query, level=None, top_k=5):
        """Search memory at a specific level (or all levels).

        Returns list of (level, template_idx, label, similarity).
        If level is None, searches all levels in parallel (decoupled search).
        """
        query_flat = flatten(as_tensor(query))
        results = []
        levels_to_search = [level] if level is not None else range(self.n_levels)
        for lv in levels_to_search:
            for ti, tpl in enumerate(self.levels[lv]["templates"]):
                sim = cosine(query_flat, tpl["data"])
                results.append((lv, ti, tpl["label"], sim))
                tpl["access_count"] += 1
        results.sort(key=lambda x: x[3], reverse=True)
        self.access_log.append(("search", len(results)))
        return results[:top_k]

    def drive_search(self, high_level_query, target_level, high_level=None, top_k=5):
        """Top-down driven search: high-level query guides low-level search.

        1. Find matching templates at high level (default: topmost level)
        2. Follow associations to target level
        3. Search only associated low-level templates (narrowed scope)
        """
        if high_level is None:
            high_level = self.n_levels - 1
        high_results = self.search(high_level_query, level=high_level, top_k=3)
        candidate_indices = set()
        for _, h_idx, _, _ in high_results:
            key = (high_level, h_idx)
            if key in self.levels[high_level]["associations"]:
                for adj_lv, adj_idx, _ in self.levels[high_level]["associations"][key]:
                    if adj_lv == target_level:
                        candidate_indices.add(adj_idx)

        if not candidate_indices:
            # Fall back to full search if no associations
            return self.search(high_level_query, level=target_level, top_k=top_k)

        query_flat = flatten(as_tensor(high_level_query))
        results = []
        for ti in candidate_indices:
            if ti < len(self.levels[target_level]["templates"]):
                tpl = self.levels[target_level]["templates"][ti]
                sim = cosine(query_flat, tpl["data"])
                results.append((target_level, ti, tpl["label"], sim))
                tpl["access_count"] += 1
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def consolidate(self, level, threshold=0.85):
        """Organize memory at a level: merge highly similar templates.

        This is the "organization" part — decoupled from search.
        Merges templates with cosine similarity > threshold into one
        (averaged) template.
        """
        templates = self.levels[level]["templates"]
        if len(templates) < 2:
            return 0
        merged = 0
        used = set()
        new_templates = []
        for i in range(len(templates)):
            if i in used:
                continue
            group = [templates[i]["data"]]
            labels = [templates[i]["label"]]
            used.add(i)
            for j in range(i + 1, len(templates)):
                if j in used:
                    continue
                sim = cosine(templates[i]["data"], templates[j]["data"])
                if sim > threshold:
                    group.append(templates[j]["data"])
                    labels.append(templates[j]["label"])
                    used.add(j)
                    merged += 1
            # average group
            if len(group) > 1:
                dim = len(group[0])
                avg = [sum(g[k] for g in group) / len(group) for k in range(dim)]
                new_templates.append({
                    "data": avg,
                    "label": labels[0],  # keep first label
                    "access_count": sum(t["access_count"] for t in [templates[i]] + [templates[j] for j in range(i+1, len(templates)) if j in used and templates[j]["label"] in labels]),
                })
            else:
                new_templates.append(templates[i])
        self.levels[level]["templates"] = new_templates
        return merged

    def get_stats(self):
        """Get memory statistics per level."""
        stats = []
        for lv in range(self.n_levels):
            templates = self.levels[lv]["templates"]
            total_access = sum(t["access_count"] for t in templates)
            stats.append({
                "level": lv,
                "n_templates": len(templates),
                "total_accesses": total_access,
                "n_associations": len(self.levels[lv]["associations"]),
            })
        return stats
