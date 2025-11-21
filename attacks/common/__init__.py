"""Common utilities for gradient-based reconstruction attacks."""

from .signals import compute_param_diff, average_param_diffs, select_layers, flatten_param_diff
from .gradients import compute_model_gradients, gradient_matching_loss, build_attack_loss
from .diffusion_prior import DiffusionPrior
from .guidance_base import BaseGuidance
from .scoring import CandidateRecord, compute_candidate_score, update_topk_candidates, select_best_candidates
from . import utils

__all__ = [
    "BaseGuidance",
    "CandidateRecord",
    "DiffusionPrior",
    "average_param_diffs",
    "build_attack_loss",
    "compute_candidate_score",
    "compute_model_gradients",
    "compute_param_diff",
    "flatten_param_diff",
    "gradient_matching_loss",
    "select_best_candidates",
    "select_layers",
    "update_topk_candidates",
    "utils",
]
