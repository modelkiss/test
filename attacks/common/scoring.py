"""Candidate scoring helpers for attack selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, MutableSequence

import torch


@dataclass
class CandidateRecord:
    image: torch.Tensor
    loss_gradient_match: float
    loss_classification: float | None = None
    score_total: float | None = None
    extra: dict | None = None


def compute_candidate_score(
    loss_gradient: float, loss_cls: float | None = None, cfg: dict | None = None
) -> float:
    """Combine different loss components into a single score."""

    cfg = cfg or {}
    weight_grad = float(cfg.get("weight_grad", 1.0))
    weight_cls = float(cfg.get("weight_cls", 0.0 if loss_cls is None else 1.0))

    score = weight_grad * loss_gradient
    if loss_cls is not None:
        score += weight_cls * loss_cls
    return score


def update_topk_candidates(
    pool: MutableSequence[CandidateRecord], new_candidates: Iterable[CandidateRecord], k: int
) -> None:
    """Maintain a pool of the top-k candidates with the lowest scores."""

    for cand in new_candidates:
        pool.append(cand)
    pool.sort(key=lambda c: c.score_total if c.score_total is not None else float("inf"))
    del pool[k:]


def select_best_candidates(pool: Iterable[CandidateRecord], k: int) -> List[CandidateRecord]:
    """Return the k candidates with the lowest scores."""

    sorted_pool = sorted(
        pool, key=lambda c: c.score_total if c.score_total is not None else float("inf")
    )
    return list(sorted_pool[:k])
