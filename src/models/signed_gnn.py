# src/models/signed_gnn.py

from typing import Optional

import torch
import torch.nn as nn


class SignedGraphLayer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Separate trainable transformations for positive/negative
        # relations.
        self.positive_transform = nn.Linear(hidden_dim, hidden_dim)
        self.negative_transform = nn.Linear(hidden_dim, hidden_dim)

        # Transform the node's own representation.
        self.self_transform = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    @staticmethod
    def aggregate(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
       
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges]."
            )

        src = edge_index[0]
        dst = edge_index[1]

        # Messages coming from source nodes.
        messages = x[src]

        if edge_weight is not None:
            if edge_weight.ndim != 1:
                raise ValueError(
                    "edge_weight must have shape [num_edges]."
                )

            if edge_weight.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    "edge_weight and edge_index must contain "
                    "the same number of edges."
                )

            messages = messages * edge_weight.unsqueeze(-1)

        # Sum messages for every destination node.
        aggregated = torch.zeros_like(x)

        aggregated.index_add_(
            dim=0,
            index=dst,
            source=messages,
        )

        # Degree normalization.
        degree = torch.zeros(
            x.size(0),
            device=x.device,
            dtype=x.dtype,
        )

        if edge_weight is None:
            degree.index_add_(
                dim=0,
                index=dst,
                source=torch.ones(
                    edge_index.shape[1],
                    device=x.device,
                    dtype=x.dtype,
                ),
            )
        else:
            degree.index_add_(
                dim=0,
                index=dst,
                source=edge_weight,
            )

        degree = degree.clamp(min=1.0)

        aggregated = aggregated / degree.unsqueeze(-1)

        return aggregated

    def forward(
        self,
        x: torch.Tensor,
        positive_edge_index: torch.Tensor,
        negative_edge_index: torch.Tensor,
        positive_edge_weight: Optional[torch.Tensor] = None,
        negative_edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply one signed message-passing layer.
        """

        positive_message = self.aggregate(
            x=x,
            edge_index=positive_edge_index,
            edge_weight=positive_edge_weight,
        )

        negative_message = self.aggregate(
            x=x,
            edge_index=negative_edge_index,
            edge_weight=negative_edge_weight,
        )

        updated = (
            self.self_transform(x)
            + self.positive_transform(positive_message)
            + self.negative_transform(negative_message)
        )

        updated = self.norm(updated)
        updated = self.activation(updated)
        updated = self.dropout(updated)

        return updated


class SignedGNN(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                SignedGraphLayer(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        positive_edge_index: torch.Tensor,
        negative_edge_index: torch.Tensor,
        positive_edge_weight: Optional[torch.Tensor] = None,
        negative_edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if x.ndim != 2:
            raise ValueError(
                "x must have shape [num_nodes, hidden_dim]."
            )

        if x.shape[1] != self.hidden_dim:
            raise ValueError(
                f"x has dimension {x.shape[1]}, but the GNN "
                f"expects hidden_dim={self.hidden_dim}."
            )

        h = x

        for layer in self.layers:
            h = layer(
                x=h,
                positive_edge_index=positive_edge_index,
                negative_edge_index=negative_edge_index,
                positive_edge_weight=positive_edge_weight,
                negative_edge_weight=negative_edge_weight,
            )

        return h
