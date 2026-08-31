"""
Matching Call Template for cos_comparison.

Template for tensor matching tasks: pair-wise similarity, template matching,
and multi-tensor comparison. Configure the top section or use --config.

Usage:
    python matching_template.py
    python matching_template.py --config my_config.json
"""

import json
import os
import sys
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    # --- Data ---
    "query_dir": "",                         # Query tensors location
    "reference_dir": "",                     # Reference tensors location
    "data_format": "txt",                    # "txt" | "csv" | "nested"
    "output_dir": "",                        # Output location

    # --- Backend ---
    "backend": "cos_comparison_c",
    "backend_fallback": ["cos_comparison"],

    # --- Matching ---
    "algorithm": "cos",                      # "cos" | "mod" | "cosmod"
    "mode": "passive",                       # "passive" | "active" | "cos"
    "top_k": 5,                              # Return top-K matches

    # --- Passive Mode ---
    "passive_window_size": [3, 3],
    "passive_w1": 1.0, "passive_w2": 1.0,
    "passive_b1": 0.0, "passive_b2": 0.0,
    "passive_step": [1, 1],
    "passive_d": [1, 0],

    # --- Active Mode ---
    "active_kernel_path": None,              # Path to kernel tensor file
    "active_w1": 1.0, "active_w2": 1.0,
    "active_b1": 0.0, "active_b2": 0.0,
    "active_step": [1, 1],

    # --- Logging ---
    "verbose": True,
}


# =====================================================================
# 2. UTILITIES
# =====================================================================
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    if not config["output_dir"]:
        config["output_dir"] = os.path.join(os.getcwd(), "output")
    os.makedirs(config["output_dir"], exist_ok=True)
    return config


def setup_backend(config: Dict[str, Any]) -> None:
    from cos_comparison.core import set_mode, get_mode
    backends = [config["backend"]] + config["backend_fallback"]
    try:
        set_mode(backends)
    except Exception:
        set_mode("cos_comparison")
    if config["verbose"]:
        print(f"[Backend] Active: {get_mode()}")


def load_tensor(filepath: str, fmt: str = "txt") -> List:
    if fmt == "nested":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    values = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if fmt == "csv":
                values.extend(float(x) for x in line.split(",") if x.strip())
            else:
                values.append(float(line))
    return values


def load_tensors_from_dir(directory: str, fmt: str = "txt",
                          verbose: bool = True) -> Dict[str, List]:
    """Load all tensors from a directory (flat or one-level subdirs)."""
    tensors: Dict[str, List] = {}
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if os.path.isfile(path):
            try:
                tensors[entry] = load_tensor(path, fmt)
            except Exception as e:
                if verbose:
                    print(f"[Data] Skip {entry}: {e}")
        elif os.path.isdir(path):
            # Subdirectory: load first file as representative
            for sub in sorted(os.listdir(path)):
                subpath = os.path.join(path, sub)
                if os.path.isfile(subpath):
                    try:
                        tensors[f"{entry}/{sub}"] = load_tensor(subpath, fmt)
                        break
                    except Exception:
                        pass
    return tensors


def get_algorithm(name: str) -> Callable:
    from cos_comparison.core import cos_comparison as _cc
    return {"cos": _cc._cos, "mod": _cc._mod, "cosmod": _cc._cosmod}.get(name, _cc._cos)


# =====================================================================
# 3. MATCHING OPERATIONS
# =====================================================================
def match_cos(query: List, references: Dict[str, List],
              config: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Full-tensor cosine similarity matching."""
    from cos_comparison.core import cos
    algo = get_algorithm(config["algorithm"])
    results = []
    for name, ref in references.items():
        try:
            sim = cos(query, ref, algorithm=algo)
            results.append((name, sim))
        except Exception as e:
            if config["verbose"]:
                print(f"[Match] {name} skipped: {e}")
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:config["top_k"]]


def match_passive(query: List, references: Dict[str, List],
                  config: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Passive-mode feature matching: extract passive features then cos-compare."""
    from cos_comparison.core import cos_comparison_passive, cos
    algo = get_algorithm(config["algorithm"])

    q_feat = cos_comparison_passive(
        query,
        window_size=tuple(config["passive_window_size"]),
        w1=config["passive_w1"], w2=config["passive_w2"],
        b1=config["passive_b1"], b2=config["passive_b2"],
        step=tuple(config["passive_step"]),
        d=tuple(config["passive_d"]),
        algorithm=algo,
    )

    results = []
    for name, ref in references.items():
        try:
            r_feat = cos_comparison_passive(
                ref,
                window_size=tuple(config["passive_window_size"]),
                w1=config["passive_w1"], w2=config["passive_w2"],
                b1=config["passive_b1"], b2=config["passive_b2"],
                step=tuple(config["passive_step"]),
                d=tuple(config["passive_d"]),
                algorithm=algo,
            )
            sim = cos(q_feat, r_feat, algorithm=algo)
            results.append((name, sim))
        except Exception as e:
            if config["verbose"]:
                print(f"[Match] {name} skipped: {e}")
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:config["top_k"]]


def match_active(query: List, references: Dict[str, List],
                 kernel: List, config: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Active-mode template matching: match kernel against each reference."""
    from cos_comparison.core import cos_comparison_active
    algo = get_algorithm(config["algorithm"])

    results = []
    for name, ref in references.items():
        try:
            response = cos_comparison_active(
                ref, kernel=kernel,
                w1=config["active_w1"], w2=config["active_w2"],
                b1=config["active_b1"], b2=config["active_b2"],
                step=tuple(config["active_step"]),
                algorithm=algo,
            )
            # Score = max response value
            flat = _flatten(response)
            score = max(flat) if flat else 0.0
            results.append((name, score))
        except Exception as e:
            if config["verbose"]:
                print(f"[Match] {name} skipped: {e}")
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:config["top_k"]]


def _flatten(obj: Any) -> List[float]:
    """Flatten nested list / tensor to 1D floats."""
    result: List[float] = []
    try:
        result.append(float(obj))
        return result
    except (TypeError, ValueError):
        pass
    if isinstance(obj, (list, tuple)):
        for item in obj:
            result.extend(_flatten(item))
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        for item in obj:
            result.extend(_flatten(item))
    return result


# =====================================================================
# 4. MAIN
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="cos_comparison matching template")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--query-dir", type=str, default=None)
    parser.add_argument("--reference-dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["cos", "passive", "active"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.query_dir:
        config["query_dir"] = args.query_dir
    if args.reference_dir:
        config["reference_dir"] = args.reference_dir
    if args.mode:
        config["mode"] = args.mode
    if args.quiet:
        config["verbose"] = False

    if not config["query_dir"] or not config["reference_dir"]:
        raise ValueError("query_dir and reference_dir must be specified")

    print("=== cos_comparison Matching Template ===")
    print(f"Query: {config['query_dir']}")
    print(f"Reference: {config['reference_dir']}")
    print(f"Mode: {config['mode']}")

    setup_backend(config)

    # Load data
    queries = load_tensors_from_dir(config["query_dir"], config["data_format"], config["verbose"])
    references = load_tensors_from_dir(config["reference_dir"], config["data_format"], config["verbose"])
    print(f"[Data] {len(queries)} queries, {len(references)} references")

    # Load kernel for active mode
    kernel = None
    if config["mode"] == "active" and config.get("active_kernel_path"):
        kernel = load_tensor(config["active_kernel_path"], config["data_format"])
        print(f"[Data] Kernel loaded: {len(_flatten(kernel))} elements")

    # Run matching
    all_results: Dict[str, List[Tuple[str, float]]] = {}
    match_fn = {
        "cos": lambda q, r: match_cos(q, r, config),
        "passive": lambda q, r: match_passive(q, r, config),
        "active": lambda q, r: match_active(q, r, kernel, config),
    }[config["mode"]]

    for qname, qtensor in queries.items():
        print(f"\n[Match] Query: {qname}")
        top = match_fn(qtensor, references)
        all_results[qname] = top
        for rank, (rname, score) in enumerate(top, 1):
            print(f"  #{rank}: {rname}  score={score:.6f}")

    # Save
    output = {
        "mode": config["mode"],
        "algorithm": config["algorithm"],
        "results": {q: [(n, float(s)) for n, s in r] for q, r in all_results.items()},
    }
    out_path = os.path.join(config["output_dir"], "matching_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Results saved to {out_path}")


if __name__ == "__main__":
    main()
