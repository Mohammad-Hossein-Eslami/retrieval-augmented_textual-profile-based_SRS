# src/training/trainer.py

from typing import Dict

import torch
import torch.nn as nn

from .losses import total_loss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.07,
        batch_size: int = 64
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.batch_size = batch_size

    def train_step(self, graph_data, users_embedding, items_embedding):
        self.model.train()

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        node_embeddings = torch.cat([users_embedding, items_embedding], dim=0)

        outputs = self.model(
            graph_data=graph_data,
            node_embeddings=node_embeddings,
        )

        # --------------------------------------------------
        # Compute losses
        # --------------------------------------------------

        loss, losses = total_loss(
            text_loss=outputs["text_loss"],
            graph_loss=outputs["graph_loss"],
            text_hidden=outputs["text_hidden"],
            graph_hidden=outputs["graph_hidden"],
            contrastive_mask=outputs["contrastive_mask"],
            contrastive_weight=self.contrastive_weight,
            contrastive_temperature=self.contrastive_temperature,
        )

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # --------------------------------------------------
        # Return values for logging
        # --------------------------------------------------

        return {
            name: value.detach().item()
            for name, value in losses.items()
        }



    @torch.no_grad()
    def eval_step(self, graph_data, users_embedding, items_embedding, split="valid", seed=42, num_negatives=100, fusion_weight=0.5):
        """Baseline-agnostic evaluation entry point."""
        self.model.eval()
        nodes = torch.cat([users_embedding, items_embedding], dim=0).to(self.device)
        return self.model.evaluate(
            node_embeddings=nodes,
            graph_data=graph_data,
            split=split,
            seed=seed,
            num_negatives=num_negatives,
            fusion_weight=fusion_weight,
        )
