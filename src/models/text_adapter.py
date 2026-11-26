import torch
import torch.nn as nn


class TextAdapter(nn.Module):
    def __init__(self, dim=768, bottleneck=256, dropout=0.1):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )

        self.beta = nn.Parameter(torch.tensor(0.1))

        # Start as an identity transformation.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        return x + self.beta * self.adapter(x)