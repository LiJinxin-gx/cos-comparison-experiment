"""Core algorithm package — zero dependency, tensor-in / tensor-out."""

from .cosine import cosine, cosine_2d, as_tensor, flatten, normalize, tensor_shape, zeros
from .multiscale import multiscale_cosine, multiscale_match, multiscale_represent, downsample
from .passive_active import passive_extract, active_match, joint_extract, local_variance, local_mean
from .hierarchical import HierarchicalRepresentation

__all__ = [
    "cosine", "cosine_2d", "as_tensor", "flatten", "normalize",
    "tensor_shape", "zeros",
    "multiscale_cosine", "multiscale_match", "multiscale_represent", "downsample",
    "passive_extract", "active_match", "joint_extract",
    "local_variance", "local_mean",
    "HierarchicalRepresentation",
]
