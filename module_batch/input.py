"""module_batch input helper for the pre-call templates.

Provides the data-location setters and tensor loading helpers that the
instruction templates use (via ``import_module input``): set the
training / query / reference data locations, load tensors from files or
directories, iterate loaded classes / tensors, and rank cosine matches.

Data file formats:
    txt      one float per line
    csv      comma-separated floats per line
    nested   JSON nested list
"""

import json
import os

_DATA_DIR = ""
_QUERY_DIR = ""
_REFERENCE_DIR = ""


def set_data_dir(path):
    """Set the training data location."""
    global _DATA_DIR
    _DATA_DIR = path
    return path


def get_data_dir():
    """The current training data location."""
    return _DATA_DIR


def set_query_dir(path):
    """Set the query data location (matching)."""
    global _QUERY_DIR
    _QUERY_DIR = path
    return path


def set_reference_dir(path):
    """Set the reference data location (matching)."""
    global _REFERENCE_DIR
    _REFERENCE_DIR = path
    return path


def get_query_dir():
    """The current query data location."""
    return _QUERY_DIR


def get_reference_dir():
    """The current reference data location."""
    return _REFERENCE_DIR


def load_tensor(path, fmt="txt"):
    """Load one tensor from a file (txt / csv / nested JSON)."""
    if fmt == "nested":
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    values = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if fmt == "csv":
                values.extend(float(x) for x in line.split(",")
                              if x.strip())
            else:
                values.append(float(line))
    return values


def load_all(directory=None, fmt="txt"):
    """Load tensors from a directory (flat or one-level subdirs) into
    {name: tensor}; the first file represents a subdirectory."""
    directory = _DATA_DIR if directory is None else directory
    tensors = {}
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if os.path.isfile(path):
            try:
                tensors[entry] = load_tensor(path, fmt)
            except Exception:
                pass
        elif os.path.isdir(path):
            for sub in sorted(os.listdir(path)):
                subpath = os.path.join(path, sub)
                if os.path.isfile(subpath):
                    try:
                        tensors[entry + "/" + sub] = load_tensor(
                            subpath, fmt)
                    except Exception:
                        pass
                    break
    return tensors


def load_training_data(directory=None, fmt="txt"):
    """Load training samples (class directory structure) into
    {class: [tensors...]}."""
    directory = _DATA_DIR if directory is None else directory
    classes = {}
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)
    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue
        samples = []
        for fname in sorted(os.listdir(class_path)):
            fpath = os.path.join(class_path, fname)
            if os.path.isfile(fpath):
                try:
                    samples.append(load_tensor(fpath, fmt))
                except Exception:
                    pass
        if samples:
            classes[class_name] = samples
    if not classes:
        raise ValueError("no samples in " + directory)
    return classes


def load_queries(fmt="txt"):
    """Load query tensors from the query data location."""
    return load_all(_QUERY_DIR, fmt)


def load_references(fmt="txt"):
    """Load reference tensors from the reference data location."""
    return load_all(_REFERENCE_DIR, fmt)


def count(data):
    """Number of entries of a mapping or a list."""
    return len(data)


def key_at(data, index):
    """Key of the index-th entry of a mapping (sorted keys)."""
    return sorted(data)[index]


def value_at(data, index):
    """Value of the index-th entry of a mapping (sorted keys)."""
    return data[sorted(data)[index]]


def sample_count(classes, class_index):
    """Number of samples of the class_index-th class."""
    return len(value_at(classes, class_index))


def sample_at(classes, class_index, sample_index):
    """The sample_index-th sample of the class_index-th class."""
    return value_at(classes, class_index)[sample_index]


def top_matches(query, references, k=5):
    """Rank references against a query by cosine similarity (top-k);
    returns [(name, score)...] sorted descending."""
    from cos_comparison.core import cos
    scored = []
    for name, ref in references.items():
        try:
            scored.append((name, cos(query, ref)))
        except Exception:
            pass
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
