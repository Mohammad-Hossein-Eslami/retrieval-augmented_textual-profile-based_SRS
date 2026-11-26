import torch
import torch.nn as nn


class DualViewSequenceEncoder(nn.Module):
    """
    Runs two item-embedding views through the SAME sequence backbone.

    The backbone parameters are shared between the two views.
    """

    def __init__(
        self,
        backbone: nn.Module,
    ):
        super().__init__()



    def forward(
        self,
        text_item_embeddings: torch.Tensor,
        graph_item_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_positions: torch.Tensor | None = None,
    ):
        """
        Args:
            text_item_embeddings:
                [num_items, item_dim]

            graph_item_embeddings:
                [num_items, item_dim]

            sequence_item_indices:
                [batch_size, seq_len]

            attention_mask:
                [batch_size, seq_len]

            masked_positions:
                [batch_size, seq_len]
                Used by BERT4Rec.
        """

        if self.backbone.model_name == 'sasrec':
            u, seq, pos, neg = self.backbone.sampler.nex_batch()
            text_hidden, text_loss = self.backbone(
                text_item_embeddings,
                u,
                seq,
                pos,
                neg,
                self.device
            )
            
            # SAME backbone instance again.
            graph_hidden, graph_loss = self.backbone(
                graph_item_embeddings,
                u,
                seq,
                pos,
                neg,
                self.device
            )


        return {
            "text_hidden": text_hidden,
            "graph_hidden": graph_hidden,
            "text_loss": text_loss,
            "graph_loss": graph_loss
        }