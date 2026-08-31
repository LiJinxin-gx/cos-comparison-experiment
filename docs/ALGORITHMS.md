# Algorithm Reference

## Core (`core/`)

### cosine.py
- `cosine(a, b)` — cosine similarity of 1D tensors (strings auto-converted)
- `cosine_2d(a, b)` — cosine similarity of 2D tensors (flattened)
- `as_tensor(data)` — convert str/list/tuple/number to tensor
- `flatten(tensor)` — flatten nested lists
- `normalize(vec)` — L2 normalization
- `tensor_shape(tensor)` — get shape tuple
- `zeros(shape)` — create zero tensor

### multiscale.py
- `downsample(tensor, factor)` — average-pool by integer factor
- `multiscale_represent(tensor, scales)` — multi-scale representations
- `multiscale_cosine(a, b, scales, weights)` — weighted multi-scale similarity
- `multiscale_match(query, database, top_k)` — match against database

### passive_active.py
- `local_variance(tensor, window)` — local variance (boundary detection)
- `local_mean(tensor, window)` — local mean (smoothing)
- `passive_extract(tensor, window, threshold)` — extract boundary points
- `passive_extract_2d(matrix, ...)` — 2D boundary extraction
- `active_match(tensor, templates)` — sliding template matching
- `joint_extract(tensor, templates)` — passive-guided active extraction

### hierarchical.py
- `HierarchicalRepresentation` — multi-level tensor with isolation and top-down drive
  - `isolate(level, indices)` — mask positions
  - `drive(high, low, template_indices)` — top-down recovery
  - `add_template(level, template, label)` — store template
  - `match_templates(level)` — match against templates
  - `info_loss_report()` — isolation statistics

## Algorithms (`algorithms/`)

### bhsm.py — Bidirectional Hierarchical Similarity Matching
- `extract_keypoints(tensor, n_thresholds, top_k)` — multi-threshold boundary points
- `geometric_signature(keypoints, n_bins)` — distance histogram signature
- `build_template(keypoints_list, n_bins)` — class template from samples
- `coarse_rank(query, templates)` — signature-based ranking
- `fine_rerank(query, candidates, database)` — multi-scale re-ranking
- `bhsm_match(query, templates, database)` — full pipeline

### thcn.py — Pure Template Network (no residual)
- `k_means(data, k)` — zero-dependency clustering
- `PureTemplateNetwork`
  - `train(data_by_class)` — build templates per level
  - `classify(tensor)` — hierarchical weighted voting
  - `get_template_path(tensor)` — nearest template per level
  - `generate(template_path, alpha)` — cross-class generation

### cross_modal.py — Paired Bridging
- `quantize_attributes(tensor)` — discrete attribute vector
- `paired_bridge_retrieval(query, database)` — two-stage cross-modal search
- `cross_modal_match(query, database, direction)` — generic cross-modal

### martian_encoding.py — Language-Agnostic Encoding
- `anonymize_symbols(sequence)` — first-occurrence renumbering
- `ngram_anonymize(text, n)` — n-gram level anonymization
- `normalized_compression_distance(a, b)` — NCD similarity
- `martian_encode(text, n)` — structural feature vector
- `martian_similarity(text_a, text_b)` — combined similarity

### memory.py — Hierarchical Memory
- `HierarchicalMemory`
  - `store(tensor, label, level)` — store template
  - `search(query, level, top_k)` — parallel search
  - `drive_search(high_query, target_level)` — top-down guided search
  - `consolidate(level, threshold)` — merge similar templates
  - `get_stats()` — memory statistics

### von_neumann_agent.py — Instructions as Data
- `VonNeumannAgent`
  - `load_program(instructions)` — load behavior program
  - `run(max_steps)` — execute (fixed interpreter)
  - `teach(examples)` — few-shot learning
  - `infer_behavior(input)` — retrieve similar program
- `make_surf_program(interests, max_depth)` — config→program factory

## Examples (`examples/`)

Each example is self-contained and runnable with `python example_*.py`.
No external dependencies. All inputs are tensors (lists) or strings.
