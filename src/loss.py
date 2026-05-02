"""
src/loss.py
===========
InfoNCE (Noise Contrastive Estimation) loss for dense retrieval.

The loss treats each query's positive document as the target class in a
B-way classification problem, where the B-1 other documents in the batch
serve as in-batch negatives.

Mathematical derivation:
    sim_matrix[i,j] = cos_sim(query_i, doc_j) / τ     shape (B, B)
    labels           = [0, 1, 2, ..., B-1]             diagonal = positives
    L = CrossEntropy(sim_matrix, labels)

This is equivalent to:
    L = -log [ exp(sim(q_i, p_i)/τ) / Σ_j exp(sim(q_i, p_j)/τ) ]

averaged over the batch.
"""

import torch
import torch.nn.functional as F


def infonce_loss(
    query_emb: torch.Tensor,   # (B, D) — MUST be L2-normalized
    doc_emb: torch.Tensor,     # (B, D) — MUST be L2-normalized
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    InfoNCE loss with in-batch negatives.

    Inputs MUST be L2-normalized (cosine similarity = dot product).

    Args:
        query_emb   : (B, D) L2-normalized query embeddings
        doc_emb     : (B, D) L2-normalized document embeddings
        temperature : τ scalar, controls sharpness of softmax
                      smaller τ → stronger contrastive signal, sharper distribution
                      larger  τ → weaker signal, flatter distribution
                      τ=0.05 is empirically good for dense retrieval

    Returns:
        scalar loss — negative log likelihood of identifying the positive doc

    Why temperature τ matters:
        As τ → 0: softmax becomes argmax (hard, non-differentiable)
        As τ → ∞: softmax becomes uniform (no learning signal)
        τ=0.05 keeps gradients informative without instability.
        Requires L2 normalization to prevent gradient explosion at small τ.

    Limitation — In-batch false negatives:
        With batch_size=B=16 and corpus_size≈6,000, for each query_i
        there is a ~(B-1)/corpus_size = 15/6000 = 0.25% chance that
        another product in the batch is ALSO relevant for that query.
        When this occurs, we incorrectly penalize a valid retrieval,
        corrupting the gradient signal (a "false negative").

        Expected false negatives per batch: B*(B-1)/corpus_size ≈ 0.04
        At our corpus size this is negligible, but grows with batch size
        and shrinks with corpus size. For larger corpora (100M products)
        the rate approaches zero and becomes irrelevant.

        Mitigation strategies:
          1. Larger batch size (reduces the relative false-negative rate)
          2. Gold label filtering (remove known positives from negatives)
          3. MNR loss with deduplication
    """
    assert query_emb.shape == doc_emb.shape, (
        f"Shape mismatch: query {query_emb.shape} vs doc {doc_emb.shape}"
    )
    B = query_emb.size(0)

    # Similarity matrix: (B, B), entry [i,j] = sim(query_i, doc_j)
    # After L2 norm, dot product = cosine similarity
    sim = torch.matmul(query_emb, doc_emb.T) / temperature   # (B, B)

    # Labels: diagonal entries are the positives
    labels = torch.arange(B, device=query_emb.device)

    # Cross-entropy: treats each row as a B-class classification problem
    return F.cross_entropy(sim, labels)


def infonce_loss_with_hard_negatives(
    query_emb: torch.Tensor,    # (B, D)
    pos_doc_emb: torch.Tensor,  # (B, D)
    neg_doc_emb: torch.Tensor,  # (B, D)  — one hard negative per query
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    InfoNCE with both in-batch negatives AND one explicit hard negative per query.

    The hard negative is concatenated to the doc matrix, giving (B+1) or (2B)
    candidate documents per query. The positive is still at position [i, i].

    Combines in-batch negatives (random, easy) with BM25 hard negatives:
        - In-batch: B-1 random products (likely easy negatives)
        - Hard: 1 BM25-retrieved product per query (harder semantic negative)
    """
    B = query_emb.size(0)

    # Stack positives and hard negatives: (2B, D)
    all_docs = torch.cat([pos_doc_emb, neg_doc_emb], dim=0)

    # Sim matrix: (B, 2B)
    sim = torch.matmul(query_emb, all_docs.T) / temperature

    # Labels: positive for query i is at position i in all_docs
    labels = torch.arange(B, device=query_emb.device)

    return F.cross_entropy(sim, labels)
