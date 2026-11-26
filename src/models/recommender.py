# src/models/recommender.py

from typing import Optional

import torch
import torch.nn as nn

from .signed_gnn import SignedGNN
from .sequence.dual_encoder import DualViewSequenceEncoder
from .recommendation_head import RecommendationHead
from .text_adapter import TextAdapter
from .prototype_adapter import PrototypeItemAdapter

class GraphSequentialRecommender(nn.Module):

    def __init__(
        self,
        gnn: SignedGNN,
        backbone,
        text_recommendation_head: RecommendationHead,
        graph_recommendation_head: RecommendationHead,
        num_users: int,
        num_prototypes: int = 64,
        prototype_dim: int = 256,
        prototype_enabled: bool = True,
    ):
        super().__init__()

        if num_users <= 0:
            raise ValueError("num_users must be positive.")

        self.gnn = gnn
        self.backbone = backbone

        self.text_recommendation_head = (text_recommendation_head)

        self.graph_recommendation_head = (graph_recommendation_head)

        self.text_adapter = TextAdapter(dim=768, bottleneck=256, dropout=0.1)
        # Stage 1: prototype enrichment after the original residual MLP.
        # Set prototype_enabled=False for the unchanged MLP baseline.
        self.prototype_enabled = prototype_enabled
        self.prototype_adapter = PrototypeItemAdapter(
            input_dim=768,
            num_prototypes=num_prototypes,
            prototype_dim=prototype_dim,
        ) if prototype_enabled else None

        self.num_users = num_users

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def split_nodes(
        self,
        node_embeddings: torch.Tensor,
    ):


        if node_embeddings.ndim != 2:
            raise ValueError(
                "node_embeddings must have shape "
                "[num_nodes, hidden_dim]."
            )

        if node_embeddings.shape[0] <= self.num_users:
            raise ValueError(
                "The number of nodes must be greater than "
                "num_users."
            )

        user_embeddings = node_embeddings[:self.num_users]

        item_embeddings = node_embeddings[self.num_users:]

        return user_embeddings, item_embeddings


    def extract_graph_data(self, graph_data, node_embeddings, u, seq, pos, neg):
        num_users = graph_data["num_users"]
        num_items = graph_data["num_items"]
        item_node_offset = graph_data["item_node_offset"]

        positive_edge_index = graph_data["positive_edge_index"].to(self.device)
        negative_edge_index = graph_data["negative_edge_index"].to(self.device)

        positive_edge_weight = graph_data["positive_edge_weight"].to(self.device)
        negative_edge_weight = graph_data["negative_edge_weight"].to(self.device)

        device = node_embeddings.device

        u = torch.as_tensor(u, dtype=torch.long, device=device) - 1
        seq = torch.as_tensor(seq, dtype=torch.long, device=device)
        pos = torch.as_tensor(pos, dtype=torch.long, device=device)
        neg = torch.as_tensor(neg, dtype=torch.long, device=device)

        seq_graph = seq[seq > 0] - 1 + item_node_offset
        pos_graph = pos[pos > 0] - 1 + item_node_offset
        neg_graph = neg[neg > 0] - 1 + item_node_offset

        batch_node_ids = torch.unique(
            torch.cat([
                u.reshape(-1),
                seq_graph.reshape(-1),
                pos_graph.reshape(-1),
                neg_graph.reshape(-1),
            ])
        )

        # Start with batch users and items
        subgraph_mask = torch.zeros(num_users + num_items, dtype=torch.bool, device=self.device)

        subgraph_mask[batch_node_ids] = True

        positive_src, positive_dst = positive_edge_index
        negative_src, negative_dst = negative_edge_index

        # Expand the incoming neighborhood by one hop
        # for each GNN layer.
        for _ in range(self.gnn.num_layers):

            pos_mask = subgraph_mask[positive_dst]
            neg_mask = subgraph_mask[negative_dst]

            new_nodes = torch.cat([positive_src[pos_mask], negative_src[neg_mask]])

            subgraph_mask[new_nodes] = True

        # Get all selected node IDs
        subgraph_node_ids = torch.where(subgraph_mask)[0]

        # Keep edges whose endpoints are both in the subgraph
        positive_mask = (subgraph_mask[positive_src] & subgraph_mask[positive_dst])

        negative_mask = (subgraph_mask[negative_src] & subgraph_mask[negative_dst])

        positive_edge_index_batch = positive_edge_index[:, positive_mask]
        positive_edge_weight_batch = positive_edge_weight[positive_mask]

        negative_edge_index_batch = negative_edge_index[:, negative_mask]
        negative_edge_weight_batch = negative_edge_weight[negative_mask]

        positive_nodes = positive_edge_index_batch.reshape(-1)
        negative_nodes = negative_edge_index_batch.reshape(-1)

        subgraph_node_ids = torch.unique(torch.cat([batch_node_ids, positive_nodes, negative_nodes]))

        global_to_local = torch.full((num_users + num_items,), -1, dtype=torch.long, device=self.device)

        global_to_local[subgraph_node_ids] = torch.arange(subgraph_node_ids.size(0), device=self.device)

        positive_edge_index_batch = global_to_local[positive_edge_index_batch]

        negative_edge_index_batch = global_to_local[negative_edge_index_batch]

        subgraph_node_embeddings = node_embeddings[subgraph_node_ids]

        subgraph_info = [
            subgraph_node_embeddings, 
            positive_edge_index_batch, 
            negative_edge_index_batch,
            positive_edge_weight_batch,
            negative_edge_weight_batch,
            subgraph_node_ids
        ]

        for info in subgraph_info:
            info.to(self.device)

        return subgraph_info
    

    def forward(
        self,
        node_embeddings,
        graph_data
    ):
    
        # ---------------------------------------------------------
        # 0. Data Batch Generation
        # ---------------------------------------------------------

       

        if self.backbone.model_name == 'sasrec':
            u, seq, pos, neg = self.backbone.sampler.next_batch()
            node_embeddings = node_embeddings.to(self.device)
            graph_output = self.extract_graph_data(graph_data, node_embeddings, u, seq, pos, neg)

        elif self.backbone.model_name == 'gru4rec':
            u, seq, pos, neg = self.backbone.sampler.next_batch()
            node_embeddings = node_embeddings.to(self.device)
            graph_output = self.extract_graph_data(graph_data, node_embeddings, u, seq, pos, neg)



        # ---------------------------------------------------------
        # 1. Graph representation learning
        # ---------------------------------------------------------

        graph_node_embeddings = self.gnn(
            x=graph_output[0],
            positive_edge_index=graph_output[1],
            negative_edge_index=graph_output[2],
            positive_edge_weight=graph_output[3],
            negative_edge_weight=graph_output[4],
        )

        # Initial/static representations.
        text_user_embeddings, text_item_embeddings = (
            self.split_nodes(node_embeddings)
        )

        text_item_embeddings = self.text_adapter(text_item_embeddings)
        if self.prototype_adapter is not None:
            text_item_embeddings = self.prototype_adapter(text_item_embeddings)

        # Graph-enhanced representations.
        subgraph_node_ids = graph_output[5]

        graph_node_embeddings_full = node_embeddings.index_copy(
            dim=0,
            index=subgraph_node_ids,
            source=graph_node_embeddings,
        )

        graph_user_embeddings, graph_item_embeddings = (
            self.split_nodes(graph_node_embeddings_full)
        )
                # ---------------------------------------------------------
        # 2. Run the two item views through the SAME
        #    sequential backbone
        # ---------------------------------------------------------

        if self.backbone.model_name == 'sasrec':
            text_hidden, text_loss = self.backbone(text_item_embeddings, u, seq, pos, neg, self.device)
            graph_hidden, graph_loss = self.backbone(graph_item_embeddings, u,seq, pos, neg, self.device)
        elif self.backbone.model_name == 'gru4rec':
            text_hidden, text_loss = self.backbone(text_item_embeddings, u, seq, pos, neg, self.device)
            graph_hidden, graph_loss = self.backbone(graph_item_embeddings, u,seq, pos, neg, self.device)

    
        # ---------------------------------------------------------
        # 3. Return everything required by the loss functions
        # ---------------------------------------------------------

        contrastive_mask = torch.as_tensor(
            seq,
            dtype=torch.long,
            device=text_hidden.device,
        ) != 0

        return {
            "text_hidden": text_hidden,
            "graph_hidden": graph_hidden,

            "text_loss": text_loss,
            "graph_loss": graph_loss,

            "text_user_embeddings": text_user_embeddings,
            "graph_user_embeddings": graph_user_embeddings,

            "text_item_embeddings": text_item_embeddings,
            "graph_item_embeddings": graph_item_embeddings,

            "contrastive_mask": contrastive_mask
        }



    @torch.no_grad()
    def evaluate(self, node_embeddings, graph_data, split="valid", seed=42, num_negatives=100, fusion_weight=0.5, fusion_weights=None):
        """Full training-graph inference ONCE, then backbone-specific ranking.

        graph_data MUST be built exclusively from training interactions.
        """
        self.eval()
        device = next(self.parameters()).device
        node_embeddings = node_embeddings.to(device)

        _, text_items = self.split_nodes(node_embeddings)

        text_items = self.text_adapter(text_items)
        if self.prototype_adapter is not None:
            text_items = self.prototype_adapter(text_items)

        graph_nodes = self.gnn(
            x=node_embeddings,
            positive_edge_index=graph_data["positive_edge_index"].to(device),
            negative_edge_index=graph_data["negative_edge_index"].to(device),
            positive_edge_weight=graph_data["positive_edge_weight"].to(device),
            negative_edge_weight=graph_data["negative_edge_weight"].to(device),
        )
        _, graph_items = self.split_nodes(graph_nodes)

        # Other backbones can implement their own evaluate() without changes here.
        return self.backbone.evaluate(
            text_item_embeddings=text_items,
            graph_item_embeddings=graph_items,
            split=split,
            seed=seed,
            num_negatives=num_negatives,
            fusion_weight=fusion_weight,
            fusion_weights=fusion_weights
        )