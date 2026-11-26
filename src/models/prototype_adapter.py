"""Prototype-based enrichment of static item embeddings (Stage 1).

The module intentionally does not use user profiles, interaction labels, or graph
information. The prototypes are learned jointly with the existing text-branch
recommendation objective. A separate user adapter and explicit signed auxiliary
objective are reserved for later controlled experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeItemAdapter(nn.Module):
    """Residual, trainable soft-prototype memory for item embeddings.

    Args:
        input_dim: Dimension of the (already MLP-adapted) item embeddings.
        num_prototypes: Number of shared item prototypes.
        prototype_dim: Dimension of the latent prototype space.
        temperature: Soft assignment temperature; must be positive.
        gate_init: Initial residual scale as a probability in (0, 1).

    Inputs:
        item_embeddings: [num_items, input_dim]

    Returns:
        [num_items, input_dim]. If return_assignments=True, also returns
        memberships [num_items, num_prototypes].
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_prototypes: int = 64,
        prototype_dim: int = 256,
        temperature: float = 0.2,
        gate_init: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0 or num_prototypes <= 0 or prototype_dim <= 0:
            raise ValueError("Dimensions and num_prototypes must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must lie strictly between 0 and 1.")

        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.query = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, prototype_dim),
        )
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, prototype_dim))
        nn.init.normal_(self.prototypes, std=prototype_dim ** -0.5)
        self.output = nn.Linear(prototype_dim, input_dim, bias=False)
        self.output_norm = nn.LayerNorm(input_dim)
        self.gate_logit = nn.Parameter(
            torch.logit(torch.tensor(float(gate_init)))
        )

    def forward(self, item_embeddings: torch.Tensor, return_assignments: bool = False):
        if item_embeddings.ndim != 2 or item_embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected item_embeddings shape [N, {self.input_dim}], "
                f"got {tuple(item_embeddings.shape)}"
            )
        query = F.normalize(self.query(item_embeddings), dim=-1)
        keys = F.normalize(self.prototypes, dim=-1)
        memberships = torch.softmax((query @ keys.T) / self.temperature, dim=-1)
        prototype_context = memberships @ self.prototypes
        correction = self.output_norm(self.output(prototype_context))
        adapted = item_embeddings + torch.sigmoid(self.gate_logit) * correction
        if return_assignments:
            return adapted, memberships
        return adapted
