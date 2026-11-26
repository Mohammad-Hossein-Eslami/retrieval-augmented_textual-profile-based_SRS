import json
import re
from pathlib import Path
import argparse

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def configure_paths(dataset):
    global ITEM_METADATA_PATH, OUTPUT_PATH

    ITEM_METADATA_PATH = (
        PROJECT_ROOT / "data" / dataset / "processed" / "item_metadata.json"
    )

    OUTPUT_PATH = (
        PROJECT_ROOT / "profiles" / dataset / "item" / "item_profiles.jsonl"
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):
    """Convert a metadata value into compact, readable text."""
    if value is None:
        return ""

    text = str(value)

    # Remove a few common pieces of source-data noise.
    text = re.sub(r"\[\[VIDEOID:[^\]]+\]\]", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_list_field(value):
    """Clean an already-flattened metadata field."""
    if value is None:
        return ""

    if isinstance(value, list):
        parts = [clean_text(item) for item in value]
        parts = [item for item in parts if item]
        return "; ".join(parts)

    return clean_text(value)


# ============================================================
# ITEM PROFILE
# ============================================================

def build_item_profile(item):
    """
    Build the single text string that will later be passed to the
    text encoder.

    The profile is intentionally template-based and deterministic.
    """
    title = clean_text(item.get("title", ""))
    brand = clean_text(item.get("brand", ""))
    categories = clean_list_field(item.get("categories", ""))
    description = clean_text(item.get("description", ""))

    fields = []

    if title:
        fields.append(f"Title: {title}")

    if brand:
        fields.append(f"Brand: {brand}")

    if categories:
        fields.append(f"Categories: {categories}")

    if description:
        fields.append(f"Description: {description}")

    # This is the encoder-ready representation.
    return ". ".join(fields)


# ============================================================
# SAVE
# ============================================================

def generate_item_profiles():
    if not ITEM_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Item metadata not found: {ITEM_METADATA_PATH}\n"
            "Run prepare_data.py first."
        )

    print(f"Loading item metadata from: {ITEM_METADATA_PATH}")

    with open(ITEM_METADATA_PATH, "r", encoding="utf-8") as file:
        item_metadata = json.load(file)

    if not isinstance(item_metadata, dict):
        raise ValueError("item_metadata.json must contain a JSON object.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    empty_profiles = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        # Sort by item_idx so the file is deterministic.
        for item_idx_str, item in sorted(
            item_metadata.items(),
            key=lambda pair: int(pair[0]),
        ):
            item_idx = int(item_idx_str)

            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid metadata for item_idx={item_idx}: expected object."
                )

            profile = build_item_profile(item)

            if not profile:
                empty_profiles += 1

            record = {
                "item_idx": item_idx,
                "item_id": clean_text(item.get("item_id", "")),
                "title": clean_text(item.get("title", "")),
                "brand": clean_text(item.get("brand", "")),
                "categories": clean_list_field(item.get("categories", "")),
                "description": clean_text(item.get("description", "")),
                "profile": profile,
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            processed += 1

            if processed % 1000 == 0:
                print(f"Processed {processed} items...")

    print("\nItem profile generation complete.")
    print(f"Items processed: {processed}")
    print(f"Empty profiles:   {empty_profiles}")
    print(f"Saved to:         {OUTPUT_PATH}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate item profiles for a dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g., Office_Products or Electronics"
    )

    args = parser.parse_args()

    configure_paths(args.dataset)

    print(f"Generating item profiles for dataset: {args.dataset}")

    generate_item_profiles()
