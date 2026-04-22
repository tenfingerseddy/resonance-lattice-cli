# SPDX-License-Identifier: BUSL-1.1
"""Trainable EML circuits for learned scoring functions.

The EML operator eml(x, y) = exp(x) - ln(|y|) is a Sheffer stroke for continuous
mathematics (Odrzywołek, 2603.21852): every elementary function is a composition
of EML trees. This module lets us SEARCH for optimal scoring functions by
training EML trees via gradient descent.

Use case: given per-band similarities [sim_0, ..., sim_{B-1}] between a source
and a query, discover the best scoring function f([sim_0, ..., sim_{B-1}]) -> R
that maximises retrieval quality. The learned tree is interpretable — its
leaves are affine combinations of the inputs, its internal nodes are eml gates.

Depth-1 tree (6 params for B=2): eml(a·sim_0+b·sim_1+c, d·sim_0+e·sim_1+f)
Depth-2 tree (12 params for B=2): eml(eml(...), eml(...))

Training:
    - Input: SciFact per-query per-source similarities, shape (Q, N, B)
    - Labels: qrels (relevance scores) per query
    - Loss: pairwise hinge on (relevant, irrelevant) doc pairs
    - Optimiser: Adam

Inference:
    - Load trained circuit
    - Compute per-band similarities for (query, source)
    - Score = circuit.forward(similarities)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Numerical safety bounds for exp/log to prevent overflow/underflow
EXP_CLAMP_MAX = 20.0   # exp(20) ≈ 4.85e8, safe in fp32
EXP_CLAMP_MIN = -50.0  # exp(-50) ≈ 2e-22
LOG_MIN = 1e-6         # ln(1e-6) ≈ -13.8


def safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Bounded exponential: clamps input to [-50, 20] to prevent overflow."""
    return torch.exp(torch.clamp(x, min=EXP_CLAMP_MIN, max=EXP_CLAMP_MAX))


def safe_log(x: torch.Tensor) -> torch.Tensor:
    """Safe log on absolute value (preserves magnitude, drops sign)."""
    return torch.log(torch.clamp(x.abs(), min=LOG_MIN))


def eml_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """EML operator: eml(x, y) = exp(x) - ln(|y|), with safe bounds."""
    return safe_exp(x) - safe_log(y)


class EMLCircuit(nn.Module):
    """Learnable balanced EML tree.

    A balanced binary tree of depth D has 2^D leaves. Each leaf is an affine
    linear projection of the input vector to a scalar. Internal nodes are
    EML gates.

    Depth=0: single linear projection (equivalent to linear scoring)
    Depth=1: eml(lin_0(x), lin_1(x))
    Depth=2: eml(eml(lin_0, lin_1), eml(lin_2, lin_3))

    Args:
        input_dim: number of input features (e.g., B bands).
        depth: tree depth. 0 = linear baseline, 1+ = EML tree.
        bias: whether leaf projections include bias.
    """

    def __init__(self, input_dim: int, depth: int = 1, bias: bool = True) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.input_dim = input_dim
        self.depth = depth
        if depth == 0:
            self.leaves = nn.ModuleList([nn.Linear(input_dim, 1, bias=bias)])
        else:
            n_leaves = 2 ** depth
            self.leaves = nn.ModuleList([
                nn.Linear(input_dim, 1, bias=bias) for _ in range(n_leaves)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute circuit output.

        Args:
            x: shape (..., input_dim). Batch dims are preserved.

        Returns:
            shape (...) — scalar score per batch element.
        """
        # Compute leaf outputs: list of (...,) tensors
        nodes = [lin(x).squeeze(-1) for lin in self.leaves]
        if self.depth == 0:
            return nodes[0]
        # Reduce pairwise via EML gates
        while len(nodes) > 1:
            new_nodes = []
            for i in range(0, len(nodes), 2):
                new_nodes.append(eml_op(nodes[i], nodes[i + 1]))
            nodes = new_nodes
        return nodes[0]

    def to_expression(self) -> str:
        """Human-readable expression of the learned circuit."""
        def leaf_expr(lin: nn.Linear, idx: int) -> str:
            w = lin.weight.data.squeeze().tolist()
            if not isinstance(w, list):
                w = [w]
            b = float(lin.bias.data.item()) if lin.bias is not None else 0.0
            terms = [f"{wi:.3f}·x{i}" for i, wi in enumerate(w)]
            if abs(b) > 1e-6:
                terms.append(f"{b:.3f}")
            return "(" + " + ".join(terms) + ")" if terms else "0"

        if self.depth == 0:
            return leaf_expr(self.leaves[0], 0)

        exprs = [leaf_expr(lin, i) for i, lin in enumerate(self.leaves)]
        while len(exprs) > 1:
            new = []
            for i in range(0, len(exprs), 2):
                new.append(f"eml({exprs[i]}, {exprs[i + 1]})")
            exprs = new
        return exprs[0]


@dataclass
class TrainConfig:
    """Configuration for EML circuit training."""
    learning_rate: float = 0.01
    epochs: int = 50
    batch_size: int = 32
    margin: float = 1.0
    weight_decay: float = 1e-5
    early_stop_patience: int = 10
    pairs_per_query: int = 4  # random (rel, irrel) pairs per query per epoch


def pairwise_hinge_loss(
    scores_pos: torch.Tensor,
    scores_neg: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pairwise hinge: max(0, margin - score_pos + score_neg)."""
    return torch.clamp(margin - scores_pos + scores_neg, min=0.0).mean()


def train_circuit(
    circuit: EMLCircuit,
    train_data: TrainData,
    config: TrainConfig,
    val_data: TrainData | None = None,
    verbose: bool = True,
) -> dict:
    """Train an EML circuit with pairwise hinge loss.

    Returns training history dict with 'train_loss', 'val_ndcg10' lists.
    """
    optimizer = torch.optim.Adam(
        circuit.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = {"train_loss": [], "val_ndcg10": [], "best_val_ndcg10": 0.0, "best_epoch": 0}
    best_state = None
    patience = 0

    for epoch in range(config.epochs):
        circuit.train()
        losses = []
        # Build (pos, neg) similarity tensors per query, then batch
        pos_sims, neg_sims = train_data.sample_pairs(config.pairs_per_query)
        if pos_sims.numel() == 0:
            break

        n = pos_sims.size(0)
        for start in range(0, n, config.batch_size):
            end = min(start + config.batch_size, n)
            batch_pos = pos_sims[start:end]
            batch_neg = neg_sims[start:end]
            optimizer.zero_grad()
            score_pos = circuit(batch_pos)
            score_neg = circuit(batch_neg)
            loss = pairwise_hinge_loss(score_pos, score_neg, config.margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(circuit.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses)) if losses else 0.0
        history["train_loss"].append(mean_loss)

        # Validation
        if val_data is not None:
            circuit.eval()
            with torch.no_grad():
                val_ndcg = evaluate_ndcg(circuit, val_data, k=10)
            history["val_ndcg10"].append(val_ndcg)
            if val_ndcg > history["best_val_ndcg10"]:
                history["best_val_ndcg10"] = val_ndcg
                history["best_epoch"] = epoch
                best_state = {k: v.detach().clone() for k, v in circuit.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if verbose:
                print(f"  epoch {epoch:3d}  loss={mean_loss:.4f}  val_nDCG@10={val_ndcg:.4f}")
            if patience >= config.early_stop_patience:
                if verbose:
                    print(f"  early stop at epoch {epoch}")
                break
        else:
            if verbose and epoch % 5 == 0:
                print(f"  epoch {epoch:3d}  loss={mean_loss:.4f}")

    if best_state is not None:
        circuit.load_state_dict(best_state)

    return history


@dataclass
class TrainData:
    """Per-query similarities + relevance labels.

    similarities: (Q, N, B) — per-band similarities between query q and source n
    qrels: dict[q_idx][n_idx] -> relevance score (0 = irrelevant, >0 = relevant)
    """
    similarities: torch.Tensor   # (Q, N, B)
    qrels: list[dict[int, float]]  # len Q, each dict maps n_idx -> relevance

    def sample_pairs(self, pairs_per_query: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (pos, neg) similarity vectors for training.

        Returns:
            pos_sims: (P, B) — similarities for relevant pairs
            neg_sims: (P, B) — similarities for irrelevant pairs (same query)
        """
        Q, N, B = self.similarities.shape
        pos_list = []
        neg_list = []
        for q_idx, rel_map in enumerate(self.qrels):
            if not rel_map:
                continue
            rel_ns = list(rel_map.keys())
            # All non-relevant indices
            rng = np.random.default_rng()
            # Sample pairs_per_query pairs
            for _ in range(pairs_per_query):
                n_pos = rel_ns[rng.integers(len(rel_ns))]
                # Sample a random irrelevant index (rejection sample if needed)
                for _ in range(20):
                    n_neg = int(rng.integers(N))
                    if n_neg not in rel_map:
                        break
                else:
                    continue
                pos_list.append(self.similarities[q_idx, n_pos])
                neg_list.append(self.similarities[q_idx, n_neg])
        if not pos_list:
            return torch.empty(0, B), torch.empty(0, B)
        return torch.stack(pos_list), torch.stack(neg_list)


def evaluate_ndcg(circuit: EMLCircuit, data: TrainData, k: int = 10) -> float:
    """Compute mean nDCG@k over all queries in data."""
    circuit.eval()
    ndcgs = []
    Q, N, B = data.similarities.shape
    with torch.no_grad():
        # Score all (query, doc) pairs
        all_scores = circuit(data.similarities)  # (Q, N)
        all_scores_np = all_scores.cpu().numpy()
    for q_idx in range(Q):
        rel_map = data.qrels[q_idx]
        if not rel_map:
            continue
        scores = all_scores_np[q_idx]
        # Sort docs by predicted score desc
        order = np.argsort(-scores)
        # Ideal order: by true relevance desc
        ideal = sorted(rel_map.values(), reverse=True)
        # DCG@k
        dcg = 0.0
        for rank, doc_idx in enumerate(order[:k]):
            rel = rel_map.get(int(doc_idx), 0.0)
            dcg += (2 ** rel - 1) / np.log2(rank + 2)
        # IDCG@k
        idcg = 0.0
        for rank, rel in enumerate(ideal[:k]):
            idcg += (2 ** rel - 1) / np.log2(rank + 2)
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def enumerate_topologies(input_dim: int, max_depth: int = 2) -> list[EMLCircuit]:
    """A4 symbolic regression: enumerate circuit topologies of increasing depth.

    For now, just produces balanced trees of each depth. A richer search
    would include unbalanced trees and different leaf sharing strategies.
    """
    return [EMLCircuit(input_dim=input_dim, depth=d) for d in range(max_depth + 1)]
