import torch
import torch.nn as nn
import torch.nn.functional as F


class RecommendationHead(nn.Module):
    """
    Scores each sequence hidden state against every candidate item.
    """

    def __init__(
        self,
        hidden_dim: int,
        item_embedding_dim: int,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.query_projection = nn.Linear(
            hidden_dim,
            item_embedding_dim,
            bias=False,
        )

        self.item_projection = nn.Linear(
            item_embedding_dim,
            item_embedding_dim,
            bias=False,
        )

        self.temperature = temperature

    def forward(
        self,
        hidden_states: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:
                [B, L, H]

            item_embeddings:
                [N, D]

        Returns:
            logits:
                [B, L, N]
        """

        queries = self.query_projection(hidden_states)

        candidates = self.item_projection(item_embeddings)

        queries = F.normalize(queries, dim=-1)

        candidates = F.normalize(candidates, dim=-1,)

        logits = torch.einsum("blh,nh->bln", queries, candidates)

        logits = logits / self.temperature

        return logits