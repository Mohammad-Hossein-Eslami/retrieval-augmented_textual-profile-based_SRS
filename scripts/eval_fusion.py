"""Load a trained two-view SASRec checkpoint and sweep validation fusion weights.

Run from the project root:
    python -m scripts.eval_fusion --checkpoint checkpoints/best_model.pt

Prerequisites: apply the SASRecEncoder.evaluate and GraphSequentialRecommender.evaluate
keyword additions described in README.md.
"""

import argparse
import csv
from pathlib import Path

import torch

from scripts.train import PROJECT_ROOT, DEFAULT_CONFIG_PATH, load_config, resolve_path, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-only fusion-weight sweep")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weights", type=str, default=None,
                        help="Comma-separated text weights, e.g. '0,0.05,0.1,0.25,0.5,1'")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Text-weight grid step when --weights is not specified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_users", type=int, default=10000)
    parser.add_argument("--num_negatives", type=int, default=100)
    parser.add_argument("--output", type=str, default="checkpoints/fusion_sweep_valid.csv")
    parser.add_argument("--evaluate_test", action="store_true",
                        help="Evaluate the single validation-selected alpha on the test set")
    return parser.parse_args()


def make_weights(args):
    if args.weights:
        weights = [float(value.strip()) for value in args.weights.split(",")]
    else:
        if not (0 < args.step <= 1):
            raise ValueError("--step must be in (0, 1]")
        weights = [round(i * args.step, 10) for i in range(int(1 / args.step) + 1)]
        if weights[-1] != 1.0:
            weights.append(1.0)
    weights = sorted(set(weights))
    if not weights or any(weight < 0 or weight > 1 for weight in weights):
        raise ValueError("All fusion weights must lie in [0, 1]")
    return weights


def main():
    args = parse_args()
    weights = make_weights(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = resolve_path(args.config)
    config = load_config(config_path)

    # These are the same data sources used by scripts/train.py.
    graph_data = torch.load(resolve_path(config["graph_data_path"]),
                            map_location="cpu", weights_only=False)
    users_embedding = torch.load(resolve_path(config["user_embedding_path"]),
                                 map_location="cpu", weights_only=False)
    items_embedding = torch.load(resolve_path(config["item_embedding_path"]),
                                 map_location="cpu", weights_only=False)

    with open(resolve_path(config["user_profile_path"]), encoding="utf-8") as file:
        num_users = sum(1 for _ in file)
    if num_users != users_embedding.shape[0]:
        raise ValueError("User-profile count differs from user-embedding row count")

    # Use the SAME model architecture and hyperparameters as training.
    model = build_model(
        num_users=num_users,
        item_embedding_dim=config["item_embedding_dim"],
        sequence_hidden_dim=config["sequence_hidden_dim"],
        batch_size=args.batch_size,
        device=device,
        backbone_name="sasrec",
    ).to(device)

    checkpoint = torch.load(resolve_path(args.checkpoint),
                            map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model.eval()  # Essential: turns off dropout.

    node_embeddings = torch.cat([users_embedding, items_embedding], dim=0).to(device)
    print(f"Loaded checkpoint: {resolve_path(args.checkpoint)}")
    print(f"Device: {device}; weights: {weights}")

    try:
        # The recommender runs the GNN once; SASRec computes each pair of
        # text/graph scores once and scores all alpha values on identical candidates.
        with torch.inference_mode():
            result = model.evaluate(
                node_embeddings=node_embeddings,
                graph_data=graph_data,
                split="valid",
                seed=args.seed,
                num_negatives=args.num_negatives,
                fusion_weight=0.5,
                fusion_weights=weights,
            )

        sweep = result["fusion_sweep"]
        print(f"\nValidation users: {result['evaluated_users']}")
        print("alpha(text)  graph weight      HR@10    NDCG@10")
        for alpha in weights:
            scores = sweep[alpha]
            print(f"{alpha:10.2f}  {1-alpha:12.2f}  "
                  f"{scores['hr@10']:9.5f}  {scores['ndcg@10']:9.5f}")

        # Select on validation NDCG@10, with smaller alpha as a tie-break.
        best_alpha = max(weights, key=lambda a: (sweep[a]["ndcg@10"], -a))
        print(f"\nSelected alpha (text): {best_alpha:.2f}")
        print("Best validation metrics:", sweep[best_alpha])

        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["alpha_text", "weight_graph", "hr@10", "ndcg@10"])
            for alpha in weights:
                writer.writerow([alpha, 1-alpha, sweep[alpha]["hr@10"], sweep[alpha]["ndcg@10"]])
        print(f"Saved validation sweep: {output_path}")

        if args.evaluate_test:
            with torch.inference_mode():
                test = model.evaluate(
                    node_embeddings=node_embeddings,
                    graph_data=graph_data,
                    split="test",
                    seed=args.seed,
                    num_negatives=args.num_negatives,
                    fusion_weight=best_alpha,
                )
            print("\nTest results with validation-selected alpha:", test)

    finally:
        # SASRec creates sampler workers at construction, even for evaluation.
        sampler = getattr(getattr(model, "backbone", None), "sampler", None)
        if sampler is not None and hasattr(sampler, "close"):
            sampler.close()


if __name__ == "__main__":
    main()
