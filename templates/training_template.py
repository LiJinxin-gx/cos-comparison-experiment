"""
Training Call Template for cos_comparison.

This template provides a ready-to-use entry point for training tasks.
Configure the top section or pass a JSON config file, then run.

Usage:
    python training_template.py
    python training_template.py --config my_config.json
"""

import json
import os
import sys
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# =====================================================================
# 1. CONFIGURATION (edit here or use --config)
# =====================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    # --- Data ---
    "data_dir": "",                          # Training data location (required)
    "data_format": "txt",                    # "txt" (one value per line) | "csv" (comma-separated) | "nested" (JSON nested list)
    "output_dir": "",                        # Output location (default: ./output)

    # --- Backend ---
    "backend": "cos_comparison_c",           # "cos_comparison_pydll" | "cos_comparison_c" | "cos_comparison"
    "backend_fallback": ["cos_comparison"],  # Fallback backends in priority order

    # --- Passive Mode ---
    "passive_window_size": [3, 3],           # Sliding window dimensions
    "passive_w1": 1.0,                       # Weight for first window
    "passive_w2": 1.0,                       # Weight for second window
    "passive_b1": 0.0,                       # Bias for first window
    "passive_b2": 0.0,                       # Bias for second window
    "passive_step": [1, 1],                  # Step size for sliding
    "passive_d": [1, 0],                     # Offset between two windows (d[0]=1 means adjacent)

    # --- Active Mode ---
    "active_kernel": None,                   # Template tensor (None = auto-generate from first sample)
    "active_w1": 1.0,
    "active_w2": 1.0,
    "active_b1": 0.0,
    "active_b2": 0.0,
    "active_step": [1, 1],

    # --- Algorithm ---
    "algorithm": "cos",                      # "cos" | "mod" | "cosmod"

    # --- Local Statistics ---
    "local_size": [3, 3],                    # Window for mean/variance
    "local_step": [1, 1],

    # --- Threshold ---
    "threshold_low": None,                   # Lower bound (None = no lower bound)
    "threshold_high": None,                  # Upper bound (None = no upper bound)

    # --- Logging ---
    "verbose": True,
    "save_intermediate": False,              # Save intermediate results
}


# =====================================================================
# 2. CONFIG LOADING
# =====================================================================
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration: defaults overridden by JSON file."""
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        config.update(user_config)
    if not config["data_dir"]:
        raise ValueError("data_dir must be specified in config or DEFAULT_CONFIG")
    if not config["output_dir"]:
        config["output_dir"] = os.path.join(os.getcwd(), "output")
    os.makedirs(config["output_dir"], exist_ok=True)
    return config


# =====================================================================
# 3. BACKEND SETUP
# =====================================================================
def setup_backend(config: Dict[str, Any]) -> str:
    """Select and verify the computation backend."""
    from cos_comparison.core import set_mode, get_mode, get_available_backends

    available = get_available_backends()
    if config["verbose"]:
        print(f"[Backend] Available: {available}")

    primary = config["backend"]
    if not primary.startswith("."):
        primary = "." + primary

    backends = [primary] + [
        b if b.startswith(".") else "." + b
        for b in config["backend_fallback"]
    ]

    try:
        set_mode([b.lstrip(".") for b in backends])
    except Exception as e:
        print(f"[Backend] Warning: {e}, falling back to pure Python")
        set_mode("cos_comparison")

    current = get_mode()
    if config["verbose"]:
        print(f"[Backend] Active: {current}")
    return current[0] if current else "unknown"


# =====================================================================
# 4. DATA LOADING
# =====================================================================
def load_tensor_from_file(filepath: str, fmt: str = "txt") -> List:
    """Load a tensor from a file.

    Args:
        filepath: Path to the data file.
        fmt: "txt" (one float per line), "csv" (comma-separated row),
             "nested" (JSON nested list).

    Returns:
        Nested list tensor.
    """
    if fmt == "nested":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    values: List[float] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if fmt == "csv":
                values.extend(float(x.strip()) for x in line.split(",") if x.strip())
            else:
                values.append(float(line))
    return values


def load_training_data(data_dir: str, fmt: str = "txt",
                       verbose: bool = True) -> Dict[str, List[List]]:
    """Load all training samples from a directory.

    Directory structure:
        data_dir/
            class_A/
                sample_001.txt
                sample_002.txt
            class_B/
                sample_001.txt

    Returns:
        Dict mapping class name -> list of tensors.
    """
    classes: Dict[str, List[List]] = {}
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        samples = []
        for fname in sorted(os.listdir(class_path)):
            fpath = os.path.join(class_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                tensor = load_tensor_from_file(fpath, fmt)
                samples.append(tensor)
            except Exception as e:
                if verbose:
                    print(f"[Data] Skipping {fpath}: {e}")
        if samples:
            classes[class_name] = samples
            if verbose:
                print(f"[Data] Class '{class_name}': {len(samples)} samples")

    if not classes:
        raise ValueError(f"No valid samples found in {data_dir}")
    return classes


# =====================================================================
# 5. CORE OPERATIONS (wrappers around cos_comparison API)
# =====================================================================
def get_algorithm(name: str) -> Callable:
    """Get algorithm function by name."""
    from cos_comparison.core import cos_comparison as _cc
    return {
        "cos": _cc._cos,
        "mod": _cc._mod,
        "cosmod": _cc._cosmod,
    }.get(name, _cc._cos)


def run_passive(data: List, config: Dict[str, Any]) -> List:
    """Run passive mode (sliding window self-similarity)."""
    from cos_comparison.core import cos_comparison_passive

    algo = get_algorithm(config["algorithm"])
    result = cos_comparison_passive(
        data,
        window_size=tuple(config["passive_window_size"]),
        w1=config["passive_w1"], w2=config["passive_w2"],
        b1=config["passive_b1"], b2=config["passive_b2"],
        step=tuple(config["passive_step"]),
        d=tuple(config["passive_d"]),
        algorithm=algo,
    )
    return result


def run_active(data: List, kernel: List, config: Dict[str, Any]) -> List:
    """Run active mode (template matching)."""
    from cos_comparison.core import cos_comparison_active

    algo = get_algorithm(config["algorithm"])
    result = cos_comparison_active(
        data, kernel=kernel,
        w1=config["active_w1"], w2=config["active_w2"],
        b1=config["active_b1"], b2=config["active_b2"],
        step=tuple(config["active_step"]),
        algorithm=algo,
    )
    return result


def run_cos(a: List, b: List, config: Dict[str, Any]) -> float:
    """Compute full-tensor cosine similarity."""
    from cos_comparison.core import cos
    algo = get_algorithm(config["algorithm"])
    return cos(a, b, algorithm=algo)


def run_local_mean(data: List, config: Dict[str, Any]) -> List:
    """Compute local mean."""
    from cos_comparison.core import mean_local
    return mean_local(
        data,
        local_size=tuple(config["local_size"]),
        step=tuple(config["local_step"]),
    )


def run_local_variance(data: List, config: Dict[str, Any]) -> List:
    """Compute local variance."""
    from cos_comparison.core import local_variance
    return local_variance(
        data,
        local_size=tuple(config["local_size"]),
        step=tuple(config["local_step"]),
    )


def run_threshold_filter(data: List, config: Dict[str, Any]) -> List:
    """Apply threshold filter."""
    from cos_comparison.core import threshold_filter
    return threshold_filter(
        data,
        low=config["threshold_low"],
        high=config["threshold_high"],
    )


# =====================================================================
# 6. TRAINING PIPELINE
# =====================================================================
def extract_features(tensor: List, config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract multi-level features from a single tensor.

    Returns dict with:
        - passive: passive mode output (boundary/edge response)
        - active: active mode output (template match response)
        - mean: local mean
        - variance: local variance
        - shape: inferred shape
    """
    from cos_comparison.core import infer_shape

    features: Dict[str, Any] = {"shape": infer_shape(tensor)}

    try:
        features["passive"] = run_passive(tensor, config)
    except Exception as e:
        features["passive"] = None
        if config["verbose"]:
            print(f"[Feature] Passive skipped: {e}")

    # Active mode: use first sample's central region as kernel if not specified
    kernel = config.get("active_kernel")
    if kernel is None:
        shape = features["shape"]
        if shape and len(shape) >= 2:
            kh = min(config["passive_window_size"][0], shape[0])
            kw = min(config["passive_window_size"][1], shape[1])
            sh = (shape[0] - kh) // 2
            sw = (shape[1] - kw) // 2
            kernel = [row[sw:sw + kw] for row in tensor[sh:sh + kh]]
    if kernel is not None:
        try:
            features["active"] = run_active(tensor, kernel, config)
        except Exception as e:
            features["active"] = None
            if config["verbose"]:
                print(f"[Feature] Active skipped: {e}")

    try:
        features["mean"] = run_local_mean(tensor, config)
    except Exception as e:
        features["mean"] = None

    try:
        features["variance"] = run_local_variance(tensor, config)
    except Exception as e:
        features["variance"] = None

    return features


def train(classes: Dict[str, List[List]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run training pipeline over all classes.

    For each sample, extract multi-level features. Then compute
    intra-class and inter-class similarity matrices.
    """
    from cos_comparison.core import cos

    print("\n=== Training Pipeline ===")
    all_features: Dict[str, List[Dict]] = {}

    for class_name, samples in classes.items():
        print(f"\n[Train] Processing class '{class_name}' ({len(samples)} samples)...")
        class_features = []
        for i, sample in enumerate(samples):
            feats = extract_features(sample, config)
            class_features.append(feats)
            if config["verbose"] and i % 10 == 0:
                print(f"  Sample {i}/{len(samples)}: shape={feats['shape']}")
        all_features[class_name] = class_features

    # Compute similarity matrix (using passive features as signature)
    print("\n[Train] Computing similarity matrix...")
    class_names = sorted(all_features.keys())
    sim_matrix: Dict[str, Dict[str, float]] = {}

    for ca in class_names:
        sim_matrix[ca] = {}
        for cb in class_names:
            # Average cosine similarity between mean passive features
            feats_a = [f["passive"] for f in all_features[ca] if f.get("passive") is not None]
            feats_b = [f["passive"] for f in all_features[cb] if f.get("passive") is not None]
            if feats_a and feats_b:
                # Flatten and compute average pairwise similarity
                total = 0.0
                count = 0
                for fa in feats_a[:5]:  # sample up to 5 per class for speed
                    for fb in feats_b[:5]:
                        try:
                            total += cos(fa, fb)
                            count += 1
                        except Exception:
                            pass
                sim_matrix[ca][cb] = total / count if count > 0 else 0.0
            else:
                sim_matrix[ca][cb] = 0.0

    # Save results
    model = {
        "config": {k: v for k, v in config.items() if k != "active_kernel"},
        "classes": class_names,
        "similarity_matrix": sim_matrix,
        "feature_count": {cn: len(all_features[cn]) for cn in class_names},
    }

    output_path = os.path.join(config["output_dir"], "training_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    print(f"\n[Train] Results saved to {output_path}")

    # Print similarity matrix
    print("\n[Train] Similarity Matrix:")
    header = " " * 12 + "".join(f"{cn:>12}" for cn in class_names)
    print(header)
    for ca in class_names:
        row = f"{ca:>12}" + "".join(f"{sim_matrix[ca][cb]:>12.4f}" for cb in class_names)
        print(row)

    return model


# =====================================================================
# 7. MAIN
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="cos_comparison training template")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--backend", type=str, default=None, help="Override backend")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.backend:
        config["backend"] = args.backend
    if args.quiet:
        config["verbose"] = False

    print(f"=== cos_comparison Training Template ===")
    print(f"Data dir: {config['data_dir']}")
    print(f"Output dir: {config['output_dir']}")

    # Step 1: Setup backend
    setup_backend(config)

    # Step 2: Load data
    classes = load_training_data(
        config["data_dir"],
        fmt=config["data_format"],
        verbose=config["verbose"],
    )

    # Step 3: Train
    model = train(classes, config)

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
