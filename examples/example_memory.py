"""
Example: Hierarchical Memory with bidirectional level driving.

Demonstrates:
  - Multi-level storage (fine detail → coarse structure)
  - Parallel search across levels (search/organization decoupled)
  - Top-down driven search (high-level context narrows low-level search)
  - Consolidation (organize similar templates without blocking search)
"""
import sys
sys.path.insert(0, r"C:\AI项目\explore")
from algorithms import HierarchicalMemory


def main():
    print("=== Creating Hierarchical Memory ===")
    mem = HierarchicalMemory(n_levels=3)

    # Store templates at different levels
    print("\n=== Storing Templates ===")
    # Level 0: fine-grained patterns
    for i in range(5):
        tpl = [float((i + j) % 4) for j in range(20)]
        idx = mem.store(tpl, label=f"fine_{i}", level=0)
        print(f"  Level 0: stored fine_{i} at index {idx}")

    # Level 1: medium patterns (averages of fine groups)
    for i in range(3):
        tpl = [float((i * 2 + j) % 3) for j in range(10)]
        idx = mem.store(tpl, label=f"medium_{i}", level=1)
        print(f"  Level 1: stored medium_{i} at index {idx}")

    # Level 2: global structure
    for i in range(2):
        tpl = [float(i % 2)] * 5
        idx = mem.store(tpl, label=f"global_{i}", level=2)
        print(f"  Level 2: stored global_{i} at index {idx}")

    print("\n=== Parallel Search (all levels) ===")
    query = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    results = mem.search(query, level=None, top_k=5)
    for lv, idx, label, sim in results:
        print(f"  Level {lv}, idx {idx} ({label}): similarity={sim:.4f}")

    print("\n=== Level-Specific Search ===")
    for lv in range(3):
        results = mem.search(query, level=lv, top_k=2)
        print(f"  Level {lv}: {[(l, s) for _, _, l, s in results]}")

    print("\n=== Top-Down Driven Search ===")
    # High-level query drives search at level 0
    high_query = [0.0, 1.0, 0.0, 1.0, 0.0]
    driven = mem.drive_search(high_query, target_level=0, top_k=3)
    print(f"  High-level query: {high_query}")
    for lv, idx, label, sim in driven:
        print(f"  Driven to level {lv}, idx {idx} ({label}): sim={sim:.4f}")

    print("\n=== Consolidation (organize similar templates) ===")
    # Add near-duplicates to level 0
    for i in range(3):
        tpl = [float((0 + j) % 4) + 0.01 for j in range(20)]  # near duplicate of fine_0
        mem.store(tpl, label=f"fine_0_dup_{i}", level=0)
    before = len(mem.levels[0]["templates"])
    merged = mem.consolidate(level=0, threshold=0.95)
    after = len(mem.levels[0]["templates"])
    print(f"  Level 0: {before} → {after} templates ({merged} merged)")

    print("\n=== Memory Statistics ===")
    for stat in mem.get_stats():
        print(f"  Level {stat['level']}: {stat['n_templates']} templates, "
              f"{stat['total_accesses']} accesses, "
              f"{stat['n_associations']} associations")


if __name__ == "__main__":
    main()
