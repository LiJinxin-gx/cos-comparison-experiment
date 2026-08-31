"""Algorithm implementations — zero-dependency, tensor-in / tensor-out."""

from .bhsm import bhsm_match, extract_keypoints, geometric_signature, build_template
from .thcn import PureTemplateNetwork, k_means
from .cross_modal import paired_bridge_retrieval, cross_modal_match, quantize_attributes
from .martian_encoding import (martian_encode, martian_similarity,
                                anonymize_symbols, ngram_anonymize,
                                normalized_compression_distance)
from .memory import HierarchicalMemory
from .von_neumann_agent import VonNeumannAgent, make_surf_program

__all__ = [
    "bhsm_match", "extract_keypoints", "geometric_signature", "build_template",
    "PureTemplateNetwork", "k_means",
    "paired_bridge_retrieval", "cross_modal_match", "quantize_attributes",
    "martian_encode", "martian_similarity", "anonymize_symbols", "ngram_anonymize",
    "normalized_compression_distance",
    "HierarchicalMemory",
    "VonNeumannAgent", "make_surf_program",
]
