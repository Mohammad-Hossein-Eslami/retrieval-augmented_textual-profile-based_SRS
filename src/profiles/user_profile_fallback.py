import json
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import argparse


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local Qwen model
MODEL_PATH = PROJECT_ROOT / "../llms" / "qwen-2.5-7b"


def configure_paths(dataset):
    global JSONL_PATH, TRAIN_CSV_PATH, OUTPUT_JSONL_PATH

    profile_dir = PROJECT_ROOT / "profiles" / dataset / "user"
    processed_dir = PROJECT_ROOT / "data" / dataset / "processed"

    JSONL_PATH = profile_dir / "user_profiles.jsonl"
    TRAIN_CSV_PATH = processed_dir / "train.csv"

    OUTPUT_JSONL_PATH = profile_dir / "user_profiles_completed.jsonl"

MAX_INTERACTIONS = 10
MAX_NEW_TOKENS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)


parser = argparse.ArgumentParser(description="Generate fallback user profiles for a dataset.")

parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., Office_Products or Electronics")

args = parser.parse_args()

configure_paths(args.dataset)

print(f"Generating fallback profiles for dataset: {args.dataset}")

if not JSONL_PATH.is_file():
    raise FileNotFoundError(
        f"User profiles not found: {JSONL_PATH}"
    )

if not TRAIN_CSV_PATH.is_file():
    raise FileNotFoundError(
        f"Training data not found: {TRAIN_CSV_PATH}"
    )


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """You create concise user preference profiles from interaction history.

Create ONE final profile of this user's preferences based only on the provided interaction history.

RULES:
- Output ONLY the final profile.
- Write 35-60 words.
- One paragraph only.
- Use natural language.
- Describe stable user preferences and patterns rather than individual purchases.
- Do not mention product names, brands, or specific items.
- Mention both positive and negative preferences when supported by the evidence.
- Do not invent preferences that are not supported by the interaction history.
- Do not use marketing language.
- Do not add explanations, headings, labels, or bullet points.
"""


def clean_response(text):

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


def build_user_prompt(interactions):
    """
    Convert the user's last interactions into text for Qwen.
    """

    interaction_text = []

    for i, row in enumerate(interactions, start=1):

        review_title = str(row.get("review_title", "")).strip()
        review_text = str(row.get("review_text", "")).strip()
        item_title = str(row.get("item_title", "")).strip()
        item_description = str(row.get("item_description", "")).strip()

        # Handle pandas NaN values
        if review_title.lower() == "nan":
            review_title = ""

        if review_text.lower() == "nan":
            review_text = ""

        if item_title.lower() == "nan":
            item_title = ""

        if item_description.lower() == "nan":
            item_description = ""

        interaction_text.append(
            f"""Interaction {i}:
Review title: {review_title}
Review text: {review_text}
Item title: {item_title}
Item description: {item_description}"""
        )

    return (
    "Here is the user's recent interaction history:\n\n"
    + "\n\n".join(interaction_text)
    + "\n\nBased only on this interaction history, create the user's final preference profile."
)


# ============================================================
# Load Qwen
# ============================================================

print(f"Loading Qwen model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=(
        torch.float16
        if DEVICE == "cuda"
        else torch.float32
    ),
    local_files_only=True,
)

if torch.cuda.is_available():
    model.to(device)

model.eval()

print("Model loaded.")


# ============================================================
# Read JSONL
# ============================================================

print(f"Reading JSONL: {JSONL_PATH}")

jsonl_rows = []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        jsonl_rows.append(json.loads(line))

print(f"Loaded {len(jsonl_rows)} JSONL rows.")


# ============================================================
# Find users with empty profiles
# ============================================================

missing_user_ids = set()

for row in jsonl_rows:

    profile = row.get("profile", "")

    # Treat None, empty string, and whitespace-only strings as empty.
    if profile is None or str(profile).strip() == "":
        user_idx = row.get("user_idx")

        if user_idx is not None:
            missing_user_ids.add(user_idx)

print(f"Users with empty profiles: {len(missing_user_ids)}")


# ============================================================
# Read train.csv
# ============================================================

print(f"Reading train CSV: {TRAIN_CSV_PATH}")

train_df = pd.read_csv(TRAIN_CSV_PATH)

print(f"Loaded {len(train_df)} interaction rows.")


# ============================================================
# Validate required columns
# ============================================================

required_columns = [
    "user_idx",
    "review_title",
    "review_text",
    "item_title",
    "item_description",
]

missing_columns = [
    col for col in required_columns
    if col not in train_df.columns
]

if missing_columns:
    raise ValueError(
        f"train.csv is missing required columns: {missing_columns}"
    )


# ============================================================
# Prepare interactions for missing-profile users
# ============================================================

# Keep only users whose profile is missing.
missing_users_df = train_df[
    train_df["user_idx"].isin(missing_user_ids)
].copy()

print(
    f"Found {len(missing_users_df)} interaction rows "
    f"belonging to users with missing profiles."
)


# ============================================================
# Generate profiles
# ============================================================

generated_profiles = {}

for user_idx in missing_user_ids:

    user_interactions = missing_users_df[
        missing_users_df["user_idx"] == user_idx
    ]

    if len(user_interactions) == 0:
        print(
            f"[WARNING] No interactions found for user {user_idx}. "
            f"Skipping."
        )
        continue

    # --------------------------------------------------------
    # Take the LAST 10 interactions.
    #
    # IMPORTANT:
    # This assumes train.csv is already ordered chronologically.
    # If your CSV contains a timestamp column, it is better to
    # explicitly sort by timestamp instead.
    # --------------------------------------------------------

    user_interactions = user_interactions.tail(MAX_INTERACTIONS)

    print(
        f"Generating profile for user {user_idx} "
        f"using {len(user_interactions)} interactions..."
    )

    interactions = user_interactions[
        [
            "review_title",
            "review_text",
            "item_title",
            "item_description",
        ]
    ].to_dict("records")

    user_prompt = build_user_prompt(interactions)

    text = SYSTEM_PROMPT + "\n\n" + user_prompt

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove the prompt tokens from the generated output.
    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    profile = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    # --------------------------------------------------------
    # Basic cleanup
    # --------------------------------------------------------

    # Sometimes models may put the profile inside quotes.
    if (
        len(profile) >= 2
        and profile[0] == '"'
        and profile[-1] == '"'
    ):
        profile = profile[1:-1].strip()

    profile = clean_response(profile)

    generated_profiles[user_idx] = profile

    print(f"Profile for user {user_idx}:")
    print(profile)
    print("-" * 80)


# ============================================================
# Replace empty profiles
# ============================================================

updated_count = 0

for row in jsonl_rows:

    profile = row.get("profile", "")

    if profile is None or str(profile).strip() == "":

        user_idx = row.get("user_idx")

        if user_idx in generated_profiles:

            row["profile"] = generated_profiles[user_idx]
            updated_count += 1


# ============================================================
# Save completed JSONL
# ============================================================

output_path = Path(OUTPUT_JSONL_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:

    for row in jsonl_rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

print()
print("=" * 80)
print("DONE")
print("=" * 80)
print(f"Generated profiles: {len(generated_profiles)}")
print(f"Updated JSONL rows: {updated_count}")
print(f"Output: {OUTPUT_JSONL_PATH}")