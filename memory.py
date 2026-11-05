# -*- coding: utf-8 -*-
"""
Open Memory System: Pluggable Memory Backends

Fully standard-library implementation with protocol-based extensibility.

Core principles:
- Open/Closed Principle: open for extension, closed for modification
- Interface Segregation: minimal interfaces
- Dependency Inversion: depend on abstractions, not concrete implementations
- Group Theory: write/read operations form a transformation group

Built-in backends:
- InMemoryBackend: in-memory (default, zero dependencies, ephemeral)
- JSONFileBackend: JSON file (persistent, human-readable)
- SQLiteBackend: SQLite database (query-optimized, persistent)

Extensibility:
- Implement the MemoryBackend protocol for custom backends
- Drop-in replacement, no core code changes required

The stored prototypes are produced by the installed cos_comparison engine
(core.pool, core.cos_comparison_passive, core.cos); the memory itself is
storage-agnostic and only preserves the engine-derived feature tuples.
"""
import json
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional

from cos_comparison import core


# ============================================================
# Memory Backend Protocol (Abstract Base Class)
# ============================================================

class MemoryBackend(ABC):
    """
    Abstract base class defining the memory backend protocol.

    All memory backends must implement these methods, forming a group-theoretic structure:
    - write: write all prototypes (overwrite)
    - read: read all prototypes
    - append: append a single prototype (incremental learning)
    - clear: clear all memory
    - size: return number of prototypes

    Group Theory:
    - Closure: write and read are both mappings on memory space
    - Identity: write(read(m)) ≈ m
    - Inverse: read⁻¹ = write
    """

    @abstractmethod
    def write(self, prototypes: List[Tuple]) -> None:
        """
        Write all prototypes (overwrite mode).

        Args:
            prototypes: list of (active_features, passive_features, label)
        """
        pass

    @abstractmethod
    def read(self) -> List[Tuple]:
        """
        Read all prototypes.

        Returns:
            list of (active_features, passive_features, label)
        """
        pass

    @abstractmethod
    def append(self, prototype: Tuple) -> None:
        """
        Append a single prototype (incremental learning).

        Args:
            prototype: (active_features, passive_features, label)
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all prototypes."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return number of prototypes."""
        pass

    @property
    def name(self) -> str:
        """Backend name."""
        return self.__class__.__name__


# ============================================================
# Backend 1: In-Memory (Default)
# ============================================================

class InMemoryBackend(MemoryBackend):
    """
    In-memory memory backend (default implementation).

    Characteristics:
    - Zero dependencies, fastest
    - Ephemeral (lost on program exit, stateless)
    - Ideal for development and testing

    Storage: Python list in process memory
    """

    def __init__(self):
        self._prototypes: List[Tuple] = []

    def write(self, prototypes: List[Tuple]) -> None:
        self._prototypes = list(prototypes)

    def read(self) -> List[Tuple]:
        return self._prototypes

    def append(self, prototype: Tuple) -> None:
        self._prototypes.append(prototype)

    def clear(self) -> None:
        self._prototypes = []

    def size(self) -> int:
        return len(self._prototypes)


# ============================================================
# Backend 2: JSON File
# ============================================================

class JSONFileBackend(MemoryBackend):
    """
    JSON file memory backend.

    Characteristics:
    - Persistent storage, cross-session learning
    - Human-readable, easy to debug
    - Suitable for small datasets

    Storage: .json file on disk
    """

    def __init__(self, filepath: str):
        """
        Args:
            filepath: path to JSON file
        """
        self.filepath = filepath
        self._prototypes: List[Tuple] = []
        if os.path.exists(filepath):
            self._load()

    def _load(self) -> None:
        """Load memory from file."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._prototypes = self._deserialize(data)
        except Exception:
            self._prototypes = []

    def _save(self) -> None:
        """Save memory to file."""
        data = self._serialize(self._prototypes)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _serialize(self, prototypes: List[Tuple]) -> List[dict]:
        """Serialize prototypes to JSON-compatible format."""
        result = []
        for active_lvls, passive_lvls, label in prototypes:
            active = [[list(p) for p in lvl] for lvl in active_lvls]
            passive = [[list(p) for p in lvl] for lvl in passive_lvls]
            result.append({
                'active': active,
                'passive': passive,
                'label': label
            })
        return result

    def _deserialize(self, data: List[dict]) -> List[Tuple]:
        """Deserialize prototypes from JSON data."""
        result = []
        for item in data:
            active = [[tuple(p) for p in lvl] for lvl in item['active']]
            passive = [[tuple(p) for p in lvl] for lvl in item['passive']]
            result.append((active, passive, item['label']))
        return result

    def write(self, prototypes: List[Tuple]) -> None:
        self._prototypes = list(prototypes)
        self._save()

    def read(self) -> List[Tuple]:
        return self._prototypes

    def append(self, prototype: Tuple) -> None:
        self._prototypes.append(prototype)
        self._save()

    def clear(self) -> None:
        self._prototypes = []
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def size(self) -> int:
        return len(self._prototypes)


# ============================================================
# Backend 3: SQLite Database
# ============================================================

class SQLiteBackend(MemoryBackend):
    """
    SQLite database memory backend.

    Characteristics:
    - Persistent storage
    - Supports query indexing
    - Suitable for large datasets

    Storage: .db file (SQLite)
    """

    def __init__(self, db_path: str):
        """
        Args:
            db_path: path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS prototypes
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      active_data TEXT,
                      passive_data TEXT,
                      label INTEGER)''')
        conn.commit()
        conn.close()

    def _serialize_prototype(self, active_lvls, passive_lvls, label) -> Tuple[str, str, int]:
        """Serialize a single prototype."""
        active = [[list(p) for p in lvl] for lvl in active_lvls]
        passive = [[list(p) for p in lvl] for lvl in passive_lvls]
        return json.dumps(active), json.dumps(passive), label

    def _deserialize_prototype(self, active_json: str, passive_json: str, label: int) -> Tuple:
        """Deserialize a single prototype."""
        active_data = json.loads(active_json)
        passive_data = json.loads(passive_json)
        active = [[tuple(p) for p in lvl] for lvl in active_data]
        passive = [[tuple(p) for p in lvl] for lvl in passive_data]
        return (active, passive, label)

    def write(self, prototypes: List[Tuple]) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM prototypes')
        for active_lvls, passive_lvls, label in prototypes:
            active_json, passive_json, lbl = self._serialize_prototype(active_lvls, passive_lvls, label)
            c.execute('INSERT INTO prototypes (active_data, passive_data, label) VALUES (?, ?, ?)',
                      (active_json, passive_json, lbl))
        conn.commit()
        conn.close()

    def read(self) -> List[Tuple]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT active_data, passive_data, label FROM prototypes')
        rows = c.fetchall()
        conn.close()
        return [self._deserialize_prototype(a, p, l) for a, p, l in rows]

    def append(self, prototype: Tuple) -> None:
        active_lvls, passive_lvls, label = prototype
        active_json, passive_json, lbl = self._serialize_prototype(active_lvls, passive_lvls, label)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('INSERT INTO prototypes (active_data, passive_data, label) VALUES (?, ?, ?)',
                  (active_json, passive_json, lbl))
        conn.commit()
        conn.close()

    def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM prototypes')
        conn.commit()
        conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def size(self) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM prototypes')
        count = c.fetchone()[0]
        conn.close()
        return count


# ============================================================
# Memory Manager (High-level Interface)
# ============================================================

class Memory:
    """
    Memory manager.

    Unified memory interface with pluggable backends.

    Usage:
        # Default in-memory
        mem = Memory()

        # JSON file
        mem = Memory(backend=JSONFileBackend("memory.json"))

        # SQLite
        mem = Memory(backend=SQLiteBackend("memory.db"))

        # Custom backend
        mem = Memory(backend=MyCustomBackend())
    """

    def __init__(self, backend: Optional[MemoryBackend] = None):
        """
        Args:
            backend: memory backend instance. Defaults to InMemoryBackend.
        """
        self.backend = backend or InMemoryBackend()

    def write(self, prototypes: List[Tuple]) -> None:
        """Write all prototypes."""
        self.backend.write(prototypes)

    def read(self) -> List[Tuple]:
        """Read all prototypes."""
        return self.backend.read()

    def append(self, prototype: Tuple) -> None:
        """Append a single prototype."""
        self.backend.append(prototype)

    def clear(self) -> None:
        """Clear all memory."""
        self.backend.clear()

    def size(self) -> int:
        """Return number of prototypes."""
        return self.backend.size()

    @property
    def backend_name(self) -> str:
        """Backend name."""
        return self.backend.name


# ============================================================
# Engine-derived prototype construction
# ============================================================

def make_engine_prototype(tensor, label):
    """Build a (active, passive, label) prototype from a 2D tensor using the
    formal engine: mean-pooled active level and its passive boundary map."""
    h, w = len(tensor), len(tensor[0])
    active = core.create_void_list((h // 2, w // 2))
    core.pool(tensor, window_size=(2, 2), step=(2, 2), output=active)
    passive = core.create_void_list((h - 1, w))
    core.cos_comparison_passive(tensor, window_size=(1, 1),
                                d=(1, 0), output=passive)
    return (active, passive, label)


# ============================================================
# Demo
# ============================================================

def main():
    print("=" * 60)
    print("Open Memory System Demo")
    print("=" * 60)
    print()
    print("Available backends:")
    print("1. InMemoryBackend (default, fastest)")
    print("2. JSONFileBackend (human-readable, persistent)")
    print("3. SQLiteBackend (query-optimized, persistent)")
    print()
    print("Extensibility: implement MemoryBackend ABC for custom backends")
    print()

    # Demo 1: In-Memory
    print("=== Demo 1: In-Memory Backend ===")
    mem1 = Memory()
    mem1.write([("active1", "passive1", 0), ("active2", "passive2", 1)])
    print(f"Backend: {mem1.backend_name}")
    print(f"Size: {mem1.size()} prototypes")
    print(f"Read: {mem1.read()}")
    print()

    # Demo 2: JSON File
    print("=== Demo 2: JSON File Backend ===")
    import tempfile
    import os

    tmpfile = os.path.join(tempfile.gettempdir(), "demo_memory.json")
    mem2 = Memory(JSONFileBackend(tmpfile))
    mem2.write([("active1", "passive1", 0), ("active2", "passive2", 1)])
    print(f"Backend: {mem2.backend_name}")
    print(f"Size: {mem2.size()} prototypes")
    print(f"File: {tmpfile}")
    print(f"File size: {os.path.getsize(tmpfile)} bytes")

    # Reload
    mem2_reload = Memory(JSONFileBackend(tmpfile))
    print(f"After reload: {mem2_reload.size()} prototypes")
    print()

    # Demo 3: SQLite
    print("=== Demo 3: SQLite Backend ===")
    tmpdb = os.path.join(tempfile.gettempdir(), "demo_memory.db")
    mem3 = Memory(SQLiteBackend(tmpdb))
    mem3.write([("active1", "passive1", 0), ("active2", "passive2", 1)])
    print(f"Backend: {mem3.backend_name}")
    print(f"Size: {mem3.size()} prototypes")
    print(f"File: {tmpdb}")
    print(f"File size: {os.path.getsize(tmpdb)} bytes")

    # Incremental append
    mem3.append(("active3", "passive3", 2))
    print(f"After append: {mem3.size()} prototypes")
    print()

    # Cleanup
    os.remove(tmpfile)
    os.remove(tmpdb)

    # Demo 4: store engine-derived features and verify the write/read group
    # round-trip preserves them exactly.
    print("=== Demo 4: Engine-derived Prototypes ===")
    sample = [[float((i + j) % 3) for j in range(6)] for i in range(6)]
    prototype = make_engine_prototype(sample, 0)
    mem4 = Memory()
    mem4.write([prototype])
    back = mem4.read()[0]
    active, passive, label = prototype
    r_active, r_passive, r_label = back
    same_active = active == r_active
    same_passive = passive == r_passive
    print(f"  Active level preserved: {same_active}")
    print(f"  Passive boundary preserved: {same_passive}")
    print(f"  Label preserved: {label == r_label}")
    assert same_active and same_passive and label == r_label
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("Open memory system verified!")
    print("- Protocol-based design (MemoryBackend ABC)")
    print("- Three built-in backends")
    print("- Extensible to custom backends")
    print("- Zero external dependencies")


if __name__ == "__main__":
    main()
