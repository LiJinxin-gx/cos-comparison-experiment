"""
Example: Von Neumann Architecture Agent — instructions as data.

Demonstrates:
  - Fixed code interpreter executes data-stored programs
  - Behavior changes by changing data (program), not code
  - External config generates behavior programs
  - Few-shot learning: store input→program mappings, infer by similarity
"""
import sys
sys.path.insert(0, r"C:\AI项目\explore")
from algorithms import VonNeumannAgent, make_surf_program


def main():
    print("=== Creating Agent (fixed code, no hardcoded behavior) ===")
    agent = VonNeumannAgent(memory_levels=2)

    print("\n=== Program 1: Simple Data Flow ===")
    program1 = [
        ("STORE", "name", "explorer"),
        ("STORE", "task", "analyze patterns"),
        ("OUTPUT", "Agent initialized"),
        ("OUTPUT", "name"),
        ("OUTPUT", "task"),
    ]
    agent.load_program(program1)
    output = agent.run()
    for line in output:
        print(f"  > {line}")

    print("\n=== Program 2: Search and Match Behavior ===")
    # Store some "knowledge" in memory first
    agent.memory.store([1.0, 0.0, 1.0], label="pattern_A", level=0)
    agent.memory.store([0.0, 1.0, 0.0], label="pattern_B", level=0)

    program2 = [
        ("STORE", "query", [1.0, 0.0, 1.0]),
        ("SEARCH", "query"),
        ("OUTPUT", "Search completed"),
        ("MATCH", [1.0, 0.0, 1.0], 0),
        ("OUTPUT", "Match completed"),
    ]
    agent.load_program(program2)
    output = agent.run()
    for line in output:
        print(f"  > {line}")

    print("\n=== Program 3: Conditional Branching ===")
    program3 = [
        ("STORE", "found", True),
        ("BRANCH", "found", 5),   # if found, jump to instruction 5
        ("OUTPUT", "Not found"),   # instruction 3 (skipped)
        ("JUMP", 6),               # instruction 4 (skipped)
        ("OUTPUT", "Found it!"),   # instruction 5 (target)
        ("OUTPUT", "Done"),        # instruction 6
    ]
    agent.load_program(program3)
    output = agent.run()
    for line in output:
        print(f"  > {line}")

    print("\n=== Program from Config (behavior-as-data factory) ===")
    surf_program = make_surf_program(["AI", "algorithms", "patterns"], max_depth=3)
    print(f"  Generated program has {len(surf_program)} instructions")
    agent.load_program(surf_program)
    output = agent.run(max_steps=20)
    for line in output[:5]:
        print(f"  > {line}")
    print(f"  ... ({len(output)} total outputs)")

    print("\n=== Few-Shot Learning: Infer Behavior from Examples ===")
    examples = [
        ([1.0, 0.0, 1.0], [("OUTPUT", "pattern A detected")]),
        ([0.0, 1.0, 0.0], [("OUTPUT", "pattern B detected")]),
    ]
    agent.teach(examples)
    inferred = agent.infer_behavior([0.9, 0.1, 0.9])  # similar to pattern A
    print(f"  Query [0.9, 0.1, 0.9] inferred program: {inferred}")


if __name__ == "__main__":
    main()
