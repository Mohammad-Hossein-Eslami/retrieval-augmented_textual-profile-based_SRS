from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseSequenceEncoder(nn.Module, ABC):
    """
    Common interface for all sequential recommendation backbones.

    Every backbone receives a sequence of item embeddings and returns
    a hidden state for every sequence position.

    Input:
        x: [batch_size, seq_len, item_dim]

    Output:
        hidden_states: [batch_size, seq_len, hidden_dim]
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        masked_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError