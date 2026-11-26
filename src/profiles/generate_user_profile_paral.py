import json
import re
from pathlib import Path
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_paths(dataset, start, end):
    global TRAIN_PATH, ITEM_METADATA_PATH
    global OUTPUT_PATH, CHECKPOINT_PATH

    dataset_dir = PROJECT_ROOT / "data" / dataset / "processed"
    profile_dir = PROJECT_ROOT / "profiles" / dataset / "user"

    TRAIN_PATH = dataset_dir / "train.csv"
    ITEM_METADATA_PATH = dataset_dir / "item_metadata.json"

    # Unique filenames for each user range
    suffix = f"{start}_{end}"

    OUTPUT_PATH = profile_dir / f"user_profiles_{suffix}.jsonl"

    CHECKPOINT_PATH = (
        profile_dir
        / "checkpoints"
        / f"user_profiles_checkpoint_{suffix}.jsonl"
    )

MODEL_PATH = PROJECT_ROOT / "../llms" / "qwen-2.5-7b"

LIKE_TH = 4
DISLIKE_TH = 2

MAX_SIDE_REVIEWS = 10

GPU_INDEX = 0
CHECKPOINT_INTERVAL = 200

MAX_INPUT_TOKENS = 1024


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):
    """Convert a value into compact, readable text."""
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(r"\[\[VIDEOID:[^\]]+\]\]", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def truncate_text(value, max_chars):
    """Clean and truncate text without cutting in the middle of a word."""
    text = clean_text(value)

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + "..."


def clean_response(text):
    """Clean common LLM formatting/prompt leakage without over-truncating valid text."""
    text = clean_text(text)

    text = re.sub(
        r"^(Human|Assistant|User|AI|Model)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"^#+\s*", "", text)

    # Remove labels only when they occur at the beginning.
    text = re.sub(
        r"^(Preference statement|Profile|Final profile|Response|Output)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    
    continuation_markers = [
        r"Human\s*:",
        r"Assistant\s*:",
        r"Profile\s*:",
        r"Final profile\s*:",
        r"Response\s*:",
        r"Output\s*:",
    ]

    for marker in continuation_markers:
        match = re.search(marker, text, flags=re.IGNORECASE)

        if match:
            text = text[:match.start()]

    # Cut everything from "(xx words)" onward.
    text = re.split(
        r"\s*\(\d+\s+words?\)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]




    # If the model starts reproducing the prompt, stop before the prompt section.
    leakage_markers = [
        "\nReview title:",
        "\nReview text:",
        "\nItem:",
        "\nPreference statements:",
        "\nPositive profile:",
        "\nNegative profile:",
        "\nFinal profile:",
    ]

    for marker in leakage_markers:
        if marker in text:
            text = text.split(marker, 1)[0]

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# RATING / INTERACTION HELPERS
# ============================================================

def rating_side(rating):
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None

    if value >= LIKE_TH:
        return "positive"

    if value <= DISLIKE_TH:
        return "negative"

    return None


def select_best_interactions(rows, max_items=10):
    """Select diverse, useful interactions for profile construction."""
    ranked = sorted(
        rows,
        key=lambda row: (
            int(row.get("helpful_vote", 0)),
            abs(float(row.get("rating", 3)) - 3),
            len(clean_text(row.get("review_text", ""))),
        ),
        reverse=True,
    )

    selected = []
    seen_items = set()

    for row in ranked:
        item_id = str(row.get("item_id", ""))

        if item_id in seen_items:
            continue

        # Skip interactions that contain virtually no textual signal.
        review_text = clean_text(row.get("review_text", ""))
        review_title = clean_text(row.get("review_title", ""))

        if not review_text and not review_title:
            continue

        seen_items.add(item_id)
        selected.append(row)

        if len(selected) >= max_items:
            break

    return selected


# ============================================================
# PROMPTS
# ============================================================

def build_interaction_prompt(row, side):
    focus = "likes" if side == "positive" else "dislikes"

    review_title = truncate_text(row.get("review_title", ""), 80)
    review_text = truncate_text(row.get("review_text", ""), 400)
    item_title = truncate_text(row.get("item_title", ""), 100)
    item_brand = truncate_text(row.get("item_brand", ""), 100)
    item_description = truncate_text(row.get("item_description", ""), 260)
    item_categories = truncate_text(row.get("item_categories", ""), 180)

    prompt = f"""
Extract one short statement about what the user {focus}.

RULES:
- Output ONLY the preference statement.
- Maximum 30 words.
- Use one or two sentences.
- Focus on the user's preference, not the product itself.
- Do not mention the product name, brand, or item identity.
- Do not use marketing language.
- Do not explain your reasoning.
- Use the review and item information together when useful.

Review title:
{review_title}

Review text:
{review_text}

Item title:
{item_title}

Item brand:
{item_brand}

Item description:
{item_description}

Item categories:
{item_categories}

Preference statement:
""".strip()

    return prompt


def build_side_prompt(side_name, preference_statements):
    focus = "likes" if side_name == "positive" else "dislikes"

    joined = "\n".join(
        f"{index}. {statement}"
        for index, statement in enumerate(preference_statements, start=1)
    )

    prompt = f"""
Generate a concise profile describing what the user {focus}.

RULES:
- Output ONLY the profile.
- One paragraph.
- 40-70 words.
- Keep recurring and meaningful preferences.
- Do not mention products, brands, or specific item names.
- Do not add explanations.
- Do not use marketing language.
- Do not simply repeat the statements.

Preference statements:
{joined}

Profile:
""".strip()

    return prompt


def build_final_prompt(positive_profile, negative_profile):
    prompt = f"""
Create ONE final profile of this user's preferences by combining the positive and negative profiles.

RULES:
- Output ONLY the final profile.
- 35-60 words.
- One paragraph only.
- Natural language.
- Describe stable user preferences rather than individual purchases.
- Do not mention product names, brands, or specific items.
- Mention both positive and negative preferences when supported by the evidence.
- Do not add explanations.
- Do not use marketing language.

Positive profile:
{positive_profile}

Negative profile:
{negative_profile}

Final profile:
""".strip()

    return prompt


# ============================================================
# MODEL
# ============================================================


def load_model():
    if torch.cuda.is_available():
        if GPU_INDEX >= torch.cuda.device_count():
            raise ValueError(
                f"GPU_INDEX={GPU_INDEX} but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        device = torch.device(f"cuda:{GPU_INDEX}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, device


def generate_text(tokenizer, model, device, prompt, max_new_tokens):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return clean_response(text)


# ============================================================
# CHECKPOINTS
# ============================================================


def load_checkpoint():
    """Load completed profiles from an existing checkpoint."""
    profiles = []
    processed_users = set()

    if not CHECKPOINT_PATH.exists():
        return profiles, processed_users

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            profile = json.loads(line)
            user_idx = int(profile["user_idx"])

            profiles.append(profile)
            processed_users.add(user_idx)

    print(f"Checkpoint contains {len(processed_users)} users.")
    return profiles, processed_users


def save_checkpoint(profiles):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as file:
        for profile in profiles:
            file.write(
                json.dumps(profile, ensure_ascii=False) + "\n"
            )


# ============================================================
# DATA LOADING
# ============================================================


def load_training_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    train = pd.read_csv(TRAIN_PATH)

    required_columns = [
        "user_idx",
        "item_idx",
        "item_id",
        "rating",
        "helpful_vote",
        "review_title",
        "review_text",
        "item_title",
        "item_brand",
        "item_description",
        "item_categories",
    ]

    missing = [column for column in required_columns if column not in train.columns]
    if missing:
        raise ValueError(
            f"train.csv is missing required profile columns: {missing}"
        )

    for column in [
        "review_title",
        "review_text",
        "item_title",
        "item_brand",
        "item_description",
        "item_categories",
    ]:
        train[column] = train[column].fillna("")

    train["helpful_vote"] = pd.to_numeric(
        train["helpful_vote"], errors="coerce"
    ).fillna(0).astype(int)

    train["rating"] = pd.to_numeric(
        train["rating"], errors="coerce"
    )

    train = train.dropna(subset=["user_idx", "rating"]).copy()
    train["user_idx"] = train["user_idx"].astype(int)

    return train


def load_item_metadata():
    """Load metadata as a fallback for rows whose train CSV fields are empty."""
    if not ITEM_METADATA_PATH.exists():
        return {}

    with open(ITEM_METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


def fill_metadata_from_item_file(train, item_metadata):
    if not item_metadata:
        return train

    train = train.copy()

    field_map = {
        "item_title": "title",
        "item_brand": "brand",
        "item_description": "description",
        "item_categories": "categories",
    }

    for row_index, row in train.iterrows():
        item_key = str(int(row["item_idx"]))
        metadata = item_metadata.get(item_key, {})

        for train_field, metadata_field in field_map.items():
            if not clean_text(row[train_field]):
                train.at[row_index, train_field] = clean_text(
                    metadata.get(metadata_field, "")
                )

    return train


# ============================================================
# SINGLE USER PROFILE
# ============================================================


def generate_user_profile(
    user_idx,
    user_rows,
    tokenizer,
    model,
    device,
):
    positive_rows = []
    negative_rows = []

    for row in user_rows:
        side = rating_side(row["rating"])

        if side == "positive":
            positive_rows.append(row)
        elif side == "negative":
            negative_rows.append(row)

    positive_rows = select_best_interactions(
        positive_rows,
        MAX_SIDE_REVIEWS,
    )

    negative_rows = select_best_interactions(
        negative_rows,
        MAX_SIDE_REVIEWS,
    )

    positive_statements = []
    for row in positive_rows:
        prompt = build_interaction_prompt(row, "positive")
        statement = generate_text(
            tokenizer,
            model,
            device,
            prompt,
            max_new_tokens=35,
        )
        if statement:
            positive_statements.append(statement)

    negative_statements = []
    for row in negative_rows:
        prompt = build_interaction_prompt(row, "negative")
        statement = generate_text(
            tokenizer,
            model,
            device,
            prompt,
            max_new_tokens=35,
        )
        if statement:
            negative_statements.append(statement)

    positive_profile = ""
    if positive_statements:
        prompt = build_side_prompt("positive", positive_statements)
        positive_profile = generate_text(
            tokenizer,
            model,
            device,
            prompt,
            max_new_tokens=100,
        )

    negative_profile = ""
    if negative_statements:
        prompt = build_side_prompt("negative", negative_statements)
        negative_profile = generate_text(
            tokenizer,
            model,
            device,
            prompt,
            max_new_tokens=100,
        )

    final_profile = ""
    if positive_profile or negative_profile:
        prompt = build_final_prompt(
            positive_profile,
            negative_profile,
        )
        final_profile = generate_text(
            tokenizer,
            model,
            device,
            prompt,
            max_new_tokens=70,
        )

    # Sensible fallbacks if an intermediate generation is empty.
    if not final_profile:
        final_profile = positive_profile or negative_profile or ""

    return {
        "user_idx": int(user_idx),
        "positive_profile": positive_profile,
        "negative_profile": negative_profile,
        "profile": final_profile,
    }


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Generate user profiles for a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g., Office_Products or Electronics",
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Starting user_idx (inclusive)",
    )

    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Ending user_idx (exclusive)",
    )

    args = parser.parse_args()

    if args.start < 0 or args.end <= args.start:
        parser.error("Require 0 <= --start < --end")

    configure_paths(args.dataset, args.start, args.end)

    print(f"Generating user profiles for dataset: {args.dataset}")

    print("Loading training data...")
    train = load_training_data()

    print(f"Training interactions: {len(train)}")
    print(f"Training users: {train['user_idx'].nunique()}")

    item_metadata = load_item_metadata()
    train = fill_metadata_from_item_file(train, item_metadata)

    user_groups = {
        int(user_idx): group.to_dict(orient="records")
        for user_idx, group in train.groupby("user_idx", sort=True)
        if args.start <= int(user_idx) < args.end
    }
    tokenizer, model, device = load_model()

    profiles, processed_users = load_checkpoint()

    remaining_users = [
        user_idx
        for user_idx in sorted(user_groups)
        if user_idx not in processed_users
    ]

    print(
        f"Users to process: {len(remaining_users)} "
        f"(already completed: {len(processed_users)})"
    )

    for count, user_idx in enumerate(remaining_users, start=1):
        profile = generate_user_profile(
            user_idx,
            user_groups[user_idx],
            tokenizer,
            model,
            device,
        )
    
        profiles.append(profile)

        if count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(profiles)
            print(
                f"Checkpoint saved: {len(profiles)} profiles total."
            )

        if count % 50 == 0 or count == 1:
            print(
                f"Processed {count}/{len(remaining_users)} "
                f"remaining users."
            )

            if count == 1:
                print("\n--- Example profile ---")
                print(f"User: {profile['user_idx']}")
                print(f"Positive: {profile['positive_profile']}")
                print(f"Negative: {profile['negative_profile']}")
                print(f"Final:    {profile['profile']}")
                print("-----------------------\n")

    profiles.sort(key=lambda row: int(row["user_idx"]))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        for profile in profiles:
            file.write(
                json.dumps(profile, ensure_ascii=False) + "\n"
            )

    print(f"Saved {len(profiles)} user profiles to:")
    print(OUTPUT_PATH)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint removed after successful completion.")


if __name__ == "__main__":
    main()


