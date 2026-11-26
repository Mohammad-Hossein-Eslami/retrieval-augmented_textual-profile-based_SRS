import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "graph.yaml"


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


# ============================================================
# VALIDATION
# ============================================================

def validate_train_dataframe(df):
    required_columns = [
        "user_idx",
        "item_idx",
        "rating",
        "timestamp",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"train.csv is missing required columns: {missing}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "train.csv contains missing values in graph-critical columns."
        )

    if not pd.api.types.is_integer_dtype(df["user_idx"]):
        df["user_idx"] = df["user_idx"].astype(int)

    if not pd.api.types.is_integer_dtype(df["item_idx"]):
        df["item_idx"] = df["item_idx"].astype(int)

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="raise",
    )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="raise",
    ).astype("int64")

    return df


# ============================================================
# EDGE BUILDING
# ============================================================

def build_signed_edges(
    df,
    num_users,
    positive_rating_min=4,
    negative_rating_max=2,
):
    """
    Build a signed bipartite graph.

    User nodes:
        0 ... num_users - 1

    Item nodes:
        num_users ... num_users + num_items - 1

    Each relation is stored bidirectionally so that message passing
    can move User -> Item and Item -> User.
    """

    positive = df[df["rating"] >= positive_rating_min].copy()
    negative = df[df["rating"] <= negative_rating_max].copy()

    # Rating == 3 is intentionally ignored.
    neutral = df[
        (df["rating"] > negative_rating_max)
        & (df["rating"] < positive_rating_min)
    ]

    def make_bidirectional_edge_index(rows):
        if rows.empty:
            return torch.empty((2, 0), dtype=torch.long)

        users = torch.tensor(
            rows["user_idx"].to_numpy(),
            dtype=torch.long,
        )

        items = torch.tensor(
            rows["item_idx"].to_numpy(),
            dtype=torch.long,
        ) + num_users

        forward = torch.stack([users, items], dim=0)
        backward = torch.stack([items, users], dim=0)

        return torch.cat([forward, backward], dim=1)

    positive_edge_index = make_bidirectional_edge_index(positive)
    negative_edge_index = make_bidirectional_edge_index(negative)

    positive_edge_weight = torch.ones(
        positive_edge_index.shape[1],
        dtype=torch.float32,
    )

    negative_edge_weight = torch.ones(
        negative_edge_index.shape[1],
        dtype=torch.float32,
    )

    return {
        "positive_edge_index": positive_edge_index,
        "negative_edge_index": negative_edge_index,
        "positive_edge_weight": positive_edge_weight,
        "negative_edge_weight": negative_edge_weight,
        "positive_interaction_count": len(positive),
        "negative_interaction_count": len(negative),
        "neutral_interaction_count": len(neutral),
    }


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graph(config, dataset):
    train_path = (PROJECT_ROOT / "data" / dataset / "processed" / "train.csv")

    item_profiles_path = (PROJECT_ROOT / "profiles" / dataset / "item" / "item_profiles.jsonl")

    output_path = (PROJECT_ROOT / "graphs" / dataset / "signed_graph.pt")

    positive_rating_min = int(config["positive_rating_min"])
    negative_rating_max = int(config["negative_rating_max"])
    use_temporal_weights = bool(config.get("use_temporal_weights", False))

    if use_temporal_weights:
        raise NotImplementedError(
            "Temporal edge weighting is planned but is intentionally "
            "disabled in the first graph implementation."
        )

    print(f"Loading training data from: {train_path}")

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}"
        )

    df = pd.read_csv(train_path)
    df = validate_train_dataframe(df)


    with open(item_profiles_path, "r", encoding="utf-8") as f:
        items_num = sum(1 for _ in f)

    print(items_num)

    # Graph structure uses training interactions only.
    num_users = int(df["user_idx"].max()) + 1
    num_items = items_num
    num_nodes = num_users + num_items

    # Sanity-check that IDs are already dense.
    expected_users = set(range(num_users))
    expected_items = set(range(num_items))

    actual_users = set(df["user_idx"].unique().tolist())
    actual_items = set(df["item_idx"].unique().tolist())

    if actual_users != expected_users:
        raise ValueError(
            "user_idx values in train.csv are not contiguous "
            "from 0 to num_users - 1."
        )

    # if actual_items != expected_items:
    #     raise ValueError(
    #         "item_idx values in train.csv are not contiguous "
    #         "from 0 to num_items - 1."
    #     )

    # Avoid accidentally creating duplicate graph edges.
    duplicate_count = df.duplicated(
        subset=["user_idx", "item_idx", "timestamp"]
    ).sum()

    if duplicate_count:
        print(
            f"Warning: found {duplicate_count} duplicate "
            "user-item-timestamp rows. Keeping the first occurrence."
        )

        df = df.drop_duplicates(
            subset=["user_idx", "item_idx", "timestamp"],
            keep="first",
        ).copy()

    graph_parts = build_signed_edges(
        df=df,
        num_users=num_users,
        positive_rating_min=positive_rating_min,
        negative_rating_max=negative_rating_max,
    )

    graph = {
        # -----------------------------
        # Node information
        # -----------------------------
        "num_users": num_users,
        "num_items": num_items,
        "num_nodes": num_nodes,
        "item_node_offset": num_users,

        # -----------------------------
        # Signed graph
        # -----------------------------
        "positive_edge_index": graph_parts["positive_edge_index"],
        "negative_edge_index": graph_parts["negative_edge_index"],
        "positive_edge_weight": graph_parts["positive_edge_weight"],
        "negative_edge_weight": graph_parts["negative_edge_weight"],

        # -----------------------------
        # Statistics
        # -----------------------------
        "positive_interaction_count": graph_parts[
            "positive_interaction_count"
        ],
        "negative_interaction_count": graph_parts[
            "negative_interaction_count"
        ],
        "neutral_interaction_count": graph_parts[
            "neutral_interaction_count"
        ],

        # -----------------------------
        # Construction metadata
        # -----------------------------
        "positive_rating_min": positive_rating_min,
        "negative_rating_max": negative_rating_max,
        "use_temporal_weights": False,
        "source_split": "train",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output_path)

    print("\nGraph created.")
    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Nodes: {num_nodes}")
    print(
        f"Positive interactions: "
        f"{graph['positive_interaction_count']}"
    )
    print(
        f"Negative interactions: "
        f"{graph['negative_interaction_count']}"
    )
    print(
        f"Ignored neutral interactions: "
        f"{graph['neutral_interaction_count']}"
    )
    print(
        f"Positive directed edges: "
        f"{graph['positive_edge_index'].shape[1]}"
    )
    print(
        f"Negative directed edges: "
        f"{graph['negative_edge_index'].shape[1]}"
    )
    print(f"Saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build a signed user-item interaction graph."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to graph.yaml",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g., Office_Products or Electronics",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    config = load_config(config_path)
    build_graph(config, args.dataset)


if __name__ == "__main__":
    main()
