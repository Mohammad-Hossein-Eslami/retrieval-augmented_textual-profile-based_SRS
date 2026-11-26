import ast
import gzip
import json
import re
from pathlib import Path
import argparse
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5


def configure_paths(dataset):
    global RAW_DIR, PROCESSED_DIR
    global REVIEWS_PATH, META_PATH
    global TRAIN_PATH, VAL_PATH, TEST_PATH
    global USER_MAPPING_PATH, ITEM_MAPPING_PATH, ITEM_METADATA_PATH

    dataset_dir = PROJECT_ROOT / "data" / dataset

    RAW_DIR = dataset_dir / "raw"
    PROCESSED_DIR = dataset_dir / "processed"

    REVIEWS_PATH = RAW_DIR / f"reviews_{dataset}.json.gz"
    META_PATH = RAW_DIR / f"meta_{dataset}.json.gz"

    TRAIN_PATH = PROCESSED_DIR / "train.csv"
    VAL_PATH = PROCESSED_DIR / "val.csv"
    TEST_PATH = PROCESSED_DIR / "test.csv"

    USER_MAPPING_PATH = PROCESSED_DIR / "user_mapping.json"
    ITEM_MAPPING_PATH = PROCESSED_DIR / "item_mapping.json"
    ITEM_METADATA_PATH = PROCESSED_DIR / "item_metadata.json"

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    """Convert a value to a compact, readable string."""
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    text = str(value)

    # Remove known junk found in the source text.
    text = re.sub(r"\[\[VIDEOID:[^\]]+\]\]", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def stringify_meta_field(value):
    """Flatten list/dict metadata fields into a clean string."""
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.extend(str(v) for v in item.values())
            else:
                parts.append(str(item))
        return clean_text("; ".join(parts))

    if isinstance(value, dict):
        return clean_text("; ".join(str(v) for v in value.values()))

    return clean_text(value)


def parse_json_line(line):
    """Parse both standard JSON lines and Python-literal style lines."""
    line = line.strip()
    if not line:
        return None

    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def load_json_gz(path):
    """Load a gzipped line-delimited JSON/JSON-like file."""
    with gzip.open(path, "rt", encoding="utf-8") as file:
        rows = []
        for line in file:
            parsed = parse_json_line(line)
            if parsed is not None:
                rows.append(parsed)

    return pd.DataFrame(rows)


# ============================================================
# LOAD REVIEWS
# ============================================================

def load_reviews():
    print("Loading reviews...")

    reviews = load_json_gz(REVIEWS_PATH)

    required_columns = [
        "overall",
        "reviewText",
        "summary",
        "asin",
        "reviewerID",
        "unixReviewTime",
        "helpful",
    ]

    missing = [c for c in required_columns if c not in reviews.columns]
    if missing:
        raise ValueError(f"Missing review columns: {missing}")

    reviews = reviews[required_columns].copy()

    reviews = reviews.rename(
        columns={
            "reviewerID": "user_id",
            "overall": "rating",
            "unixReviewTime": "timestamp",
            "helpful": "helpful_vote",
            "reviewText": "review_text",
            "summary": "review_title",
        }
    )

    # Keep identifiers as strings throughout preprocessing.
    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews["item_id"] = reviews["asin"].astype(str)

    # Clean review text fields before saving them into train/val/test.
    reviews["review_title"] = reviews["review_title"].apply(clean_text)
    reviews["review_text"] = reviews["review_text"].apply(clean_text)

    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews["timestamp"] = pd.to_numeric(
        reviews["timestamp"], errors="coerce"
    )

    # The Amazon source commonly stores helpful as something like "3/5".
    reviews["helpful_vote"] = (
        reviews["helpful" if "helpful" in reviews.columns else "helpful_vote"]
        .astype(str)
        .str.split("/")
        .str[0]
    )
    reviews["helpful_vote"] = (
        pd.to_numeric(reviews["helpful_vote"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Remove rows that cannot participate in chronological preprocessing.
    reviews = reviews.dropna(
        subset=["user_id", "item_id", "rating", "timestamp"]
    ).copy()

    reviews["rating"] = reviews["rating"].astype(float)
    reviews["timestamp"] = reviews["timestamp"].astype("int64")

    # Remove exact duplicate interactions while preserving repeated
    # purchases/reviews of the same item at different timestamps.
    reviews = (
        reviews.sort_values(
            ["user_id", "timestamp"],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=["user_id", "item_id", "timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(f"Reviews loaded: {reviews.shape}")
    return reviews


# ============================================================
# 5-CORE FILTERING
# ============================================================

def apply_k_core(df):
    """Iteratively enforce minimum user/item interaction counts."""
    df = df.copy()

    iteration = 0

    while True:
        iteration += 1

        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()

        valid_users = user_counts[
            user_counts >= MIN_USER_INTERACTIONS
        ].index
        valid_items = item_counts[
            item_counts >= MIN_ITEM_INTERACTIONS
        ].index

        filtered = df[
            df["user_id"].isin(valid_users)
            & df["item_id"].isin(valid_items)
        ].copy()

        print(
            f"5-core iteration {iteration}: "
            f"{len(filtered)} interactions, "
            f"{filtered['user_id'].nunique()} users, "
            f"{filtered['item_id'].nunique()} items"
        )

        if len(filtered) == len(df):
            break

        df = filtered

    return df.reset_index(drop=True)


# ============================================================
# LOAD ITEM METADATA
# ============================================================

def load_metadata():
    print("Loading metadata...")

    meta = load_json_gz(META_PATH)

    if "asin" not in meta.columns:
        raise ValueError("Metadata file does not contain 'asin'.")

    wanted_columns = [
        "asin",
        "title",
        "description",
        "categories",
        "brand",
    ]

    present = [c for c in wanted_columns if c in meta.columns]
    meta = meta[present].copy()

    meta = meta.rename(
        columns={
            "asin": "item_id",
            "title": "item_title",
            "description": "item_description",
            "categories": "item_categories",
            "brand": "item_brand",
        }
    )

    meta["item_id"] = meta["item_id"].astype(str)

    for column in [
        "item_title",
        "item_description",
        "item_categories",
        "item_brand",
    ]:
        if column not in meta.columns:
            meta[column] = ""
        else:
            meta[column] = meta[column].apply(stringify_meta_field)

    # One metadata row per item.
    meta = meta.drop_duplicates(subset=["item_id"], keep="first")

    print(f"Metadata loaded: {meta.shape}")
    return meta


# ============================================================
# MERGE REVIEWS + METADATA
# ============================================================

def merge_data(reviews, metadata):
    print("Merging reviews and metadata...")

    df = reviews.merge(
        metadata,
        how="left",
        on="item_id",
        validate="many_to_one",
    )

    for column in [
        "item_title",
        "item_description",
        "item_categories",
        "item_brand",
    ]:
        if column not in df.columns:
            df[column] = ""
        else:
            df[column] = df[column].fillna("")

    return df


# ============================================================
# CREATE INTERNAL IDS
# ============================================================

def create_mappings(df):
    # Sorted raw identifiers give deterministic mappings across runs.
    unique_users = sorted(df["user_id"].astype(str).unique())
    unique_items = sorted(df["item_id"].astype(str).unique())

    user_mapping = {
        raw_id: idx for idx, raw_id in enumerate(unique_users)
    }
    item_mapping = {
        raw_id: idx for idx, raw_id in enumerate(unique_items)
    }

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_mapping).astype(int)
    df["item_idx"] = df["item_id"].map(item_mapping).astype(int)

    return df, user_mapping, item_mapping


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(df):
    """For each user: all but last 2 -> train; second-last -> val; last -> test."""
    train_rows = []
    val_rows = []
    test_rows = []

    for user_idx, group in df.groupby("user_idx", sort=True):
        group = group.sort_values("timestamp", kind="mergesort")

        if len(group) < 3:
            # This should not happen after 5-core, but keep the guard.
            continue

        train_rows.append(group.iloc[:-2])
        val_rows.append(group.iloc[-2: -1])
        test_rows.append(group.iloc[-1:])

    if not train_rows:
        raise ValueError("No users have enough interactions for splitting.")

    train = pd.concat(train_rows, ignore_index=True)
    val = pd.concat(val_rows, ignore_index=True)
    test = pd.concat(test_rows, ignore_index=True)

    sort_columns = ["user_idx", "timestamp"]

    train = train.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    val = val.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    test = test.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    return train, val, test


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(df, train, val, test, user_mapping, item_mapping):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Columns needed by the recommender and by static profile generation.
    output_columns = [
        "user_idx",
        "item_idx",
        "user_id",
        "item_id",
        "rating",
        "timestamp",
        "helpful_vote",
        "review_title",
        "review_text",
        "item_title",
        "item_brand",
        "item_description",
        "item_categories",
    ]

    train[output_columns].to_csv(TRAIN_PATH, index=False)
    val[output_columns].to_csv(VAL_PATH, index=False)
    test[output_columns].to_csv(TEST_PATH, index=False)

    with open(USER_MAPPING_PATH, "w", encoding="utf-8") as file:
        json.dump(user_mapping, file, ensure_ascii=False, indent=2)

    with open(ITEM_MAPPING_PATH, "w", encoding="utf-8") as file:
        json.dump(item_mapping, file, ensure_ascii=False, indent=2)

    # Store metadata for every item that survived filtering, even when
    # that item only appears in validation/test. This will be useful later
    # for item profiles and item-text representations.
    metadata_rows = (
        df[
            [
                "item_idx",
                "item_id",
                "item_title",
                "item_brand",
                "item_description",
                "item_categories",
            ]
        ]
        .drop_duplicates(subset=["item_idx"])
        .sort_values("item_idx")
    )

    item_metadata = {}
    for row in metadata_rows.to_dict(orient="records"):
        item_metadata[str(int(row["item_idx"]))] = {
            "item_id": row["item_id"],
            "title": row["item_title"],
            "brand": row["item_brand"],
            "description": row["item_description"],
            "categories": row["item_categories"],
        }

    with open(ITEM_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(item_metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved: {TRAIN_PATH}")
    print(f"Saved: {VAL_PATH}")
    print(f"Saved: {TEST_PATH}")
    print(f"Saved: {USER_MAPPING_PATH}")
    print(f"Saved: {ITEM_MAPPING_PATH}")
    print(f"Saved: {ITEM_METADATA_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess a dataset for recommendation.")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., Office_Products or Electronics")

    args = parser.parse_args()

    configure_paths(args.dataset)

    if not REVIEWS_PATH.is_file():
        raise FileNotFoundError(
            f"Reviews file not found: {REVIEWS_PATH}"
        )

    if not META_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {META_PATH}"
        )

    print(f"Processing dataset: {args.dataset}")

    reviews = load_reviews()

    print("Applying 5-core filtering...")
    reviews = apply_k_core(reviews)

    metadata = load_metadata()
    df = merge_data(reviews, metadata)

    df, user_mapping, item_mapping = create_mappings(df)

    train, val, test = chronological_split(df)

    print("\nFinal dataset:")
    print(f"  Interactions: {len(df)}")
    print(f"  Users:        {df['user_idx'].nunique()}")
    print(f"  Items:        {df['item_idx'].nunique()}")
    print(f"  Train:        {len(train)}")
    print(f"  Validation:   {len(val)}")
    print(f"  Test:         {len(test)}")

    save_outputs(
        df=df,
        train=train,
        val=val,
        test=test,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )


if __name__ == "__main__":
    main()
