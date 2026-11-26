# scripts/train.py

import torch
import yaml
from pathlib import Path
import argparse

from src.models.signed_gnn import SignedGNN
from src.models.sequence.factory import build_sequence_encoder
from src.models.sequence.dual_encoder import DualViewSequenceEncoder
from src.models.recommendation_head import RecommendationHead
from src.models.recommender import GraphSequentialRecommender
from src.training.trainer import Trainer



# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


# ============================================================
# CONFIG
# ============================================================

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(path_value):
    """Resolve paths relative to the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path




def build_model(
    num_users: int,
    item_embedding_dim: int,
    sequence_hidden_dim: int,
    batch_size: int,
    device,
    interaction_path,
    backbone_name: str = "sasrec",
):
    """
    Build the complete graph + sequential recommender.

    This function is responsible only for constructing the model.
    """

    # --------------------------------------------------
    # 1. Signed GNN
    # --------------------------------------------------

    gnn = SignedGNN(
        hidden_dim=item_embedding_dim,
        num_layers=2,
        dropout=0.1,
    )

    # --------------------------------------------------
    # 2. Choose the sequential backbone
    # --------------------------------------------------

    backbone = build_sequence_encoder(
        name=backbone_name,
        item_dim=item_embedding_dim,
        hidden_dim=sequence_hidden_dim,
        batch_size=batch_size,
        device=device,
        interaction_path=interaction_path,
    )

    # --------------------------------------------------
    # 3. Shared dual-view sequence encoder
    # --------------------------------------------------

    # dual_view_encoder = DualViewSequenceEncoder(
    #     backbone=backbone,
    # )

    # --------------------------------------------------
    # 4. Recommendation heads
    # --------------------------------------------------

    text_head = RecommendationHead(
        hidden_dim=sequence_hidden_dim,
        item_embedding_dim=item_embedding_dim,
    )

    graph_head = RecommendationHead(
        hidden_dim=sequence_hidden_dim,
        item_embedding_dim=item_embedding_dim,
    )

    # --------------------------------------------------
    # 5. Complete recommender
    # --------------------------------------------------

    model = GraphSequentialRecommender(
        gnn=gnn,
        backbone=backbone,
        text_recommendation_head=text_head,
        graph_recommendation_head=graph_head,
        num_users=num_users,
    )

    return model


def main():

    parser = argparse.ArgumentParser(description="Define your training configs here.")

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., Office_Products or Electronics")
    parser.add_argument("--backbone", type=str, default="sasrec")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--fusion_weight", type=float, default=0.5)

    
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}"
            )
    
    config = load_config(config_path)


    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # --------------------------------------------------
    # These will eventually come from our processed data
    # and configuration files.
    # --------------------------------------------------

    dataset = args.dataset

    graph_data_path = PROJECT_ROOT / "graphs" / dataset / "signed_graph.pt"

    user_embedding_path = (PROJECT_ROOT / "embeddings" / dataset / "user_embeddings.pt")

    item_embedding_path = (PROJECT_ROOT / "embeddings" / dataset / "item_embeddings.pt")

    interaction_path = PROJECT_ROOT / "data" / dataset / "processed"

    graph_data = torch.load(graph_data_path, map_location="cpu", weights_only=False)
    users_embedding = torch.load(user_embedding_path, map_location="cpu", weights_only=False)
    items_embedding = torch.load(item_embedding_path, map_location="cpu", weights_only=False)


    
    num_users = int(graph_data["num_users"])
    num_items = int(graph_data["num_items"])

    if users_embedding.shape[0] != num_users:
        raise ValueError(
            f"User embedding count ({users_embedding.shape[0]}) "
            f"does not match graph user count ({num_users})."
        )

    if items_embedding.shape[0] != num_items:
        raise ValueError(
            f"Item embedding count ({items_embedding.shape[0]}) "
            f"does not match graph item count ({num_items})."
        )

    # Dimension of the precomputed text embeddings.
    item_embedding_dim = config["item_embedding_dim"]

    # Hidden dimension used by the sequence backbone.
    sequence_hidden_dim = config["sequence_hidden_dim"]

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    model = build_model(
        num_users=num_users,
        item_embedding_dim=item_embedding_dim,
        sequence_hidden_dim=sequence_hidden_dim,
        batch_size=args.batch_size,
        device=device,
        interaction_path=interaction_path,
        backbone_name=args.backbone,
    )

    model = model.to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        contrastive_weight=0.1,
        contrastive_temperature=0.07,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    num_epochs = args.epochs
    num_batch = (num_users - 1) // args.batch_size + 1
    best_ndcg = float("-inf")
    best_checkpoint = (PROJECT_ROOT / "checkpoints" / dataset / args.backbone / "best_model.pt")
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):

        print(
            f"\n===== Epoch {epoch + 1}/{num_epochs} ====="
        )

        epoch_losses = []
        for _ in range(num_batch):
            train_loss = trainer.train_step(graph_data, users_embedding, items_embedding)
            epoch_losses.append(train_loss["total"])

        print(train_loss)

        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == num_epochs:
            metrics = trainer.eval_step(
                graph_data, users_embedding, items_embedding, split="valid", seed=42, fusion_weight=args.fusion_weight)
            print("Validation:", metrics)

            if metrics["fused_ndcg@10"] > best_ndcg:
                best_ndcg = metrics["fused_ndcg@10"]
                torch.save(model.state_dict(), best_checkpoint)
                print(f"Saved best checkpoint: {best_checkpoint}")

    # Use the BEST validation checkpoint for test, never the last epoch.
    model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
    test_metrics = trainer.eval_step(graph_data, users_embedding, items_embedding, split="test", seed=42, fusion_weight=args.fusion_weight)
    print("Test:", test_metrics)


if __name__ == "__main__":
    main()