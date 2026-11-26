"""Evaluate a Stage 1 prototype-enhanced two-view SASRec checkpoint.

Run from the project root:
    python -m scripts.eval_stage1 --dataset Office_Products

Or specify a checkpoint explicitly:
    python -m scripts.eval_stage1 --dataset Office_Products \
        --checkpoint checkpoints/Office_Products/sasrec/stage1_protos_64_256/best_model.pt

Uses a fixed 0.5 text / 0.5 graph fusion. By default evaluates the TEST set.
Pass --split valid or --split both to evaluate other splits.
Requires the Stage 1 prototype adapter in src/models/ and the existing
GraphSequentialRecommender.evaluate / SASRecEncoder.evaluate implementations.
"""

import argparse
import csv
from pathlib import Path

import torch
import yaml

from src.models.signed_gnn import SignedGNN
from src.models.sequence.factory import build_sequence_encoder
from src.models.recommendation_head import RecommendationHead
from src.models.recommender import GraphSequentialRecommender


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"
FUSION_WEIGHT = 0.5


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 prototype-enhanced SASRec at fixed 0.5 fusion"
    )
    parser.add_argument("--dataset", required=True, help="Dataset folder name, e.g. Office_Products")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--checkpoint", default=None,
        help="Stage 1 checkpoint. If omitted, infer from dataset and prototype settings.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_prototypes", type=int, default=64)
    parser.add_argument("--prototype_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_negatives", type=int, default=100)
    parser.add_argument(
        "--split", choices=("valid", "test", "both"), default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--output", default=None,
        help="CSV output path (default: next to the checkpoint).",
    )
    return parser.parse_args()


def require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def build_stage1_model(num_users, item_dim, hidden_dim, batch_size,
                       device, interaction_path, num_prototypes, prototype_dim):
    # Build Stage 1 directly, independently of later changes in scripts/train.py.
    gnn = SignedGNN(hidden_dim=item_dim, num_layers=2, dropout=0.1)
    backbone = build_sequence_encoder(
        name="sasrec",
        item_dim=item_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        device=device,
        interaction_path=interaction_path,
    )
    text_head = RecommendationHead(
        hidden_dim=hidden_dim, item_embedding_dim=item_dim,
    )
    graph_head = RecommendationHead(
        hidden_dim=hidden_dim, item_embedding_dim=item_dim,
    )
    model = GraphSequentialRecommender(
        gnn=gnn,
        backbone=backbone,
        text_recommendation_head=text_head,
        graph_recommendation_head=graph_head,
        num_users=num_users,
        prototype_enabled=True,
        num_prototypes=num_prototypes,
        prototype_dim=prototype_dim,
    )
    return model.to(device)


def print_metrics(split, result):
    print(f"\n{split.upper()} | evaluated users: {result['evaluated_users']}")
    print(f"Fusion: text={FUSION_WEIGHT:.2f}, graph={1.0 - FUSION_WEIGHT:.2f}")
    print(f"{'View':<12}{'HR@10':>12}{'NDCG@10':>12}")
    for view in ("text", "graph", "fused"):
        print(
            f"{view:<12}"
            f"{result[f'{view}_hr@10']:>12.5f}"
            f"{result[f'{view}_ndcg@10']:>12.5f}"
        )


def main():
    args = parse_args()
    config_path = require_file(resolve_path(args.config), "training configuration")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset = args.dataset
    graph_path = PROJECT_ROOT / "graphs" / dataset / "signed_graph.pt"
    user_path = PROJECT_ROOT / "embeddings" / dataset / "user_embeddings.pt"
    item_path = PROJECT_ROOT / "embeddings" / dataset / "item_embeddings.pt"
    interaction_path = PROJECT_ROOT / "data" / dataset / "processed"

    if args.checkpoint is None:
        checkpoint_path = (
            PROJECT_ROOT / "checkpoints" / dataset / "sasrec"
            / f"stage1_protos_{args.num_prototypes}_{args.prototype_dim}"
            / "best_model.pt"
        )
    else:
        checkpoint_path = resolve_path(args.checkpoint)

    for description, path in (
        ("signed graph", graph_path),
        ("user embeddings", user_path),
        ("item embeddings", item_path),
        ("Stage 1 checkpoint", checkpoint_path),
    ):
        require_file(path, description)
    for filename in ("train_interaction.txt", "valid_interaction.txt", "test_interaction.txt"):
        require_file(interaction_path / filename, "interaction data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)
    users_embedding = torch.load(user_path, map_location="cpu", weights_only=False)
    items_embedding = torch.load(item_path, map_location="cpu", weights_only=False)

    num_users = int(graph_data["num_users"])
    num_items = int(graph_data["num_items"])
    item_dim = int(config["item_embedding_dim"])
    hidden_dim = int(config["sequence_hidden_dim"])

    if users_embedding.ndim != 2 or items_embedding.ndim != 2:
        raise ValueError("User and item embeddings must both be 2D tensors")
    if users_embedding.shape[0] != num_users or items_embedding.shape[0] != num_items:
        raise ValueError(
            "Embedding counts differ from graph counts: "
            f"users {users_embedding.shape[0]}/{num_users}, "
            f"items {items_embedding.shape[0]}/{num_items}"
        )
    if users_embedding.shape[1] != item_dim or items_embedding.shape[1] != item_dim:
        raise ValueError(
            f"Embedding widths must match item_embedding_dim={item_dim}; "
            f"got users={users_embedding.shape[1]}, items={items_embedding.shape[1]}"
        )
    if item_dim != 768:
        raise ValueError("Your Stage 1 TextAdapter/PrototypeItemAdapter expects 768-dimensional embeddings")
    if graph_data.get("source_split") not in (None, "train"):
        raise ValueError("Evaluation graph must be constructed from training interactions only")

    model = None
    try:
        model = build_stage1_model(
            num_users, item_dim, hidden_dim, args.batch_size,
            device, interaction_path, args.num_prototypes, args.prototype_dim,
        )
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint does not match the Stage 1 architecture. "
                "Check that this is a Stage 1 checkpoint and that --num_prototypes "
                "and --prototype_dim match training. Also ensure Stage 2/3/4 "
                "modules are disabled or your Stage 1 source files are restored."
            ) from exc
        model.eval()
        node_embeddings = torch.cat((users_embedding, items_embedding), dim=0).to(device)

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {device} | Dataset: {dataset} | Fixed fusion weight: {FUSION_WEIGHT}")
        splits = ("valid", "test") if args.split == "both" else (args.split,)
        csv_rows = []
        with torch.inference_mode():
            for split in splits:
                result = model.evaluate(
                    node_embeddings=node_embeddings,
                    graph_data=graph_data,
                    split=split,
                    seed=args.seed,
                    num_negatives=args.num_negatives,
                    fusion_weight=FUSION_WEIGHT,
                )
                print_metrics(split, result)
                for view in ("text", "graph", "fused"):
                    csv_rows.append((
                        dataset, split, FUSION_WEIGHT, view,
                        result["evaluated_users"],
                        result[f"{view}_hr@10"],
                        result[f"{view}_ndcg@10"],
                    ))

        output_path = (
            resolve_path(args.output) if args.output is not None
            else checkpoint_path.parent / f"eval_fixed_0.5_{args.split}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("dataset", "split", "text_weight", "view", "evaluated_users", "hr@10", "ndcg@10"))
            writer.writerows(csv_rows)
        print(f"\nSaved results to: {output_path}")
    finally:
        # SASRec launches sampler subprocesses even when used for evaluation.
        sampler = getattr(getattr(model, "backbone", None), "sampler", None)
        if sampler is not None and hasattr(sampler, "close"):
            sampler.close()


if __name__ == "__main__":
    main()
