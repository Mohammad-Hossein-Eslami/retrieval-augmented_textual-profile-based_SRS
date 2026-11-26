# src/training/losses.py

import torch
import torch.nn.functional as F


def recommendation_loss(
    logits: torch.Tensor,
    target_items: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:

    selected_logits = logits[position_mask]
    selected_targets = target_items[position_mask]

    if selected_logits.shape[0] == 0:
        raise ValueError(
            "No valid positions were selected for "
            "recommendation loss."
        )

    return F.cross_entropy(
        selected_logits,
        selected_targets,
    )


def contrastive_loss(
    text_hidden: torch.Tensor,
    graph_hidden: torch.Tensor,
    position_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:

    text_vectors = text_hidden[position_mask]
    graph_vectors = graph_hidden[position_mask]

    if text_vectors.shape[0] < 2:
        raise ValueError(
            "At least two valid positions are required "
            "for contrastive learning."
        )

    text_vectors = F.normalize(text_vectors, dim=-1)

    graph_vectors = F.normalize(graph_vectors, dim=-1)

    # Pairwise similarity matrix.
    #
    # [M, H] @ [H, M]
    #       ↓
    #     [M, M]
    #
    # similarity[i, j] tells us how similar:
    #
    #     text position i
    #
    # is to:
    #
    #     graph position j
    similarity = (text_vectors @ graph_vectors.T) / temperature

    labels = torch.arange(similarity.shape[0], device=similarity.device)

    # Text -> Graph
    text_to_graph = F.cross_entropy(similarity, labels)

    # Graph -> Text
    graph_to_text = F.cross_entropy(similarity.T, labels)

    return 0.5 * (text_to_graph + graph_to_text)


def total_loss(
    text_loss: float,
    graph_loss: float,
    text_hidden: torch.Tensor,
    graph_hidden: torch.Tensor,
    contrastive_mask: torch.Tensor,
    contrastive_weight: float = 0.1,
    contrastive_temperature: float = 0.07,
):

    cl_loss = contrastive_loss(
        text_hidden=text_hidden,
        graph_hidden=graph_hidden,
        position_mask=contrastive_mask,
        temperature=contrastive_temperature,
    )

    loss = (text_loss + graph_loss + contrastive_weight * cl_loss)

    losses = {
        "total": loss,
        "text_recommendation": text_loss,
        "graph_recommendation": graph_loss,
        "contrastive": cl_loss,
    }

    return loss, losses