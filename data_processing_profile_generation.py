#####################
# This is the preprocessing file.
# This file could be applied to any of the Amazon Review dataset categories.
# Change the policies regarding meta columns and user interactions in any way you desire
#####################

import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_DIR = Path("")

# ============================================================
# Dataset configuration
# Change only this name when using another Amazon category.
# ============================================================

DATASET_NAME = "Video_Games"

REVIEWS_PATH = DATA_DIR / f"../data/{DATASET_NAME}.json"
META_PATH = DATA_DIR / f"../data/meta_{DATASET_NAME}.json"

LIKE_TH = 4
DISLIKE_TH = 2

MIN_USER_INTERACTIONS = 5

CORE_USER_INTERACTIONS = 15

# Desired number of users
TARGET_NUM_USERS = 12000

USER_SAMPLE_SEED = 42

MAX_SIDE_REVIEWS = 10

MODEL_PATH = "../../llms/qwen-2.5-7b"

GPU_INDEX = 1

CHECKPOINT_INTERVAL = 200 

START_USER_IDX = 0
END_USER_IDX = 4000

PROFILE_OUT_PATH = f"user_profiles_{DATASET_NAME}_{START_USER_IDX}_{END_USER_IDX}.jsonl"
CHECKPOINT_PATH = f"user_profiles_{DATASET_NAME}_checkpoint_{START_USER_IDX}_{END_USER_IDX}.jsonl"


def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"\[\[VIDEOID:[^\]]+\]\]", " ", text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    return text


def truncate_text(text, max_chars=180):
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def stringify_meta_field(value):
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    if isinstance(value, list):
        parts = []
        for x in value:
            if isinstance(x, dict):
                parts.extend(str(v) for v in x.values())
            else:
                parts.append(str(x))
        return clean_text("; ".join(parts))

    if isinstance(value, dict):
        return clean_text("; ".join(str(v) for v in value.values()))

    return clean_text(value)


def rating_side(rating):
    try:
        r = float(rating)
    except:
        return None

    if r >= LIKE_TH:
        return "positive"

    if r <= DISLIKE_TH:
        return "negative"

    return None


print("Loading reviews...")

with open(REVIEWS_PATH, "r", encoding="utf-8") as f:
    review_rows = json.load(f)

reviews = pd.DataFrame(review_rows)

# verified purchase only
if "verified_purchase" in reviews.columns:
    vp = reviews["verified_purchase"]

    if vp.dtype == "object":
        vp = (
            vp.astype(str).str.lower().map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )

    reviews["verified_purchase"] = vp.fillna(False).astype(bool)

    reviews = reviews[reviews["verified_purchase"]].copy()

reviews = reviews[[
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
    "helpful_vote"
]].copy()

reviews = reviews.rename(columns={
    "title": "review_title",
    "text": "review_text"
})

reviews = reviews.dropna(subset=[
    "user_id",
    "asin",
    "rating",
    "timestamp"
])

reviews["timestamp"] = (reviews["timestamp"] // 1000).astype("int64")

reviews["join_asin"] = (reviews["parent_asin"].fillna(reviews["asin"]).astype(str))

print("Reviews:", reviews.shape)

print("Loading meta...")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta_rows = json.load(f)

meta = pd.DataFrame(meta_rows)

# IMPORTANT: Add whichever field you think is necessary
meta_cols = [
    "parent_asin",
    "title",
    "features",
    "description",
    "categories",
]

meta_cols = [c for c in meta_cols if c in meta.columns]

meta = meta[meta_cols].copy()

meta = meta.rename(columns={
    "parent_asin": "meta_parent_asin",
    "title": "item_title",
    "features": "item_features",
    "description": "item_description",
    "categories": "item_categories",
})

meta["meta_parent_asin"] = (meta["meta_parent_asin"].astype(str))

for col in ["item_title", "item_features", "item_description", "item_categories"]:
    if col not in meta.columns:
        meta[col] = ""

    meta[col] = meta[col].apply(stringify_meta_field)

meta = meta.drop_duplicates(subset=["meta_parent_asin"])

print("Meta:", meta.shape)

print("Merging...")

df = reviews.merge(
    meta,
    how="left",
    left_on="join_asin",
    right_on="meta_parent_asin"
)

df["item_id"] = df["join_asin"].astype(str)

df["helpful_vote"] = (pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int))

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

df = (
    df.sort_values(["user_id", "timestamp"]).drop_duplicates( subset=["user_id", "item_id"], keep="last")
)

print("Merged:", df.shape)


user_counts = df["user_id"].value_counts()

# Apply whatever policy you desire for user selection
####################
core_users = user_counts[user_counts >= CORE_USER_INTERACTIONS].index


filler_candidates = user_counts[(user_counts >= MIN_USER_INTERACTIONS) & (user_counts < CORE_USER_INTERACTIONS)].index

needed_fillers = max(0, TARGET_NUM_USERS - len(core_users))

if needed_fillers > len(filler_candidates):
    print(
        f"Warning: Need {needed_fillers} filler users, "
        f"but only {len(filler_candidates)} are available. "
        f"Using all available filler users."
    )

    sampled_fillers = filler_candidates
else:
    sampled_fillers = pd.Index(filler_candidates).to_series().sample(n=needed_fillers, random_state=USER_SAMPLE_SEED).index

good_users = pd.Index(core_users).append(pd.Index(sampled_fillers))

df = df[df["user_id"].isin(good_users)].copy()

print("Core users >= 15 interactions:", len(core_users))
print("Sampled filler users 5-14 interactions:", len(sampled_fillers))
print("Filtered users:", df["user_id"].nunique())
########################################

unique_users = df["user_id"].unique()
unique_items = df["item_id"].unique()

user2id = {
    u: i for i, u in enumerate(unique_users)
}

item2id = {
    a: i for i, a in enumerate(unique_items)
}

df["user_idx"] = df["user_id"].map(user2id)
df["item_idx"] = df["item_id"].map(item2id)

df["user_idx"] = df["user_idx"].astype(int)
df["item_idx"] = df["item_idx"].astype(int)

df = df.sort_values(["user_idx", "timestamp"])


train_by_user = {}

interaction_rows = []

for uid, g in df.groupby("user_idx"):

    g = g.sort_values("timestamp")

    if len(g) < 3:
        continue

    train_g = g.iloc[:-2]

    val_row = g.iloc[-2]

    test_row = g.iloc[-1]

    train_by_user[uid] = train_g

    # train
    for _, row in train_g.iterrows():

        interaction_rows.append({
            "user_idx": uid,
            "item_idx": int(row["item_idx"]),
            "timestamp": int(row["timestamp"]),
            "split": "train"
        })

    # val
    interaction_rows.append({
        "user_idx": uid,
        "item_idx": int(val_row["item_idx"]),
        "timestamp": int(val_row["timestamp"]),
        "split": "val"
    })

    # test
    interaction_rows.append({
        "user_idx": uid,
        "item_idx": int(test_row["item_idx"]),
        "timestamp": int(test_row["timestamp"]),
        "split": "test"
    })

interaction_df = pd.DataFrame(interaction_rows)

interaction_df = interaction_df.sort_values(["user_idx", "timestamp"])

interaction_df.to_csv(
    f"sasrec_interactions_{DATASET_NAME}.txt",
    sep="\t",
    index=False
)

print("Saved interactions")

print("Loading model...")

if torch.cuda.is_available():
    device = torch.device(f"cuda:{GPU_INDEX}")
else:
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=(
        torch.float16
        if device.type == "cuda"
        else torch.float32
    ),
    local_files_only=True,
)

model.to(device)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def clean_response(text):

    text = str(text)
    text = re.sub(
        r"(Human|Assistant|User|AI|Model):",
        "",
        text,
        flags=re.I
    )

    text = re.sub(r"#+", " ", text)

    bad_patterns = [
        "critical rules",
        "task:",
        "generate",
        "output:",
        "response:",
        "instruction:",
        "note:",
        "Extract"
    ]

    lower = text.lower()

    for pat in bad_patterns:
        idx = lower.find(pat)

        if idx != -1:
            text = text[:idx]

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def generate_text(prompt, max_new_tokens=40):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,

            do_sample=False,
            temperature=0.0,

            repetition_penalty=1.1,

            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(generated, skip_special_tokens=True)

    return clean_response(text)

def select_best_interactions(rows, max_items=10):

    rows = sorted(
        rows,
        key=lambda x: (
            int(x.get("helpful_vote", 0)),
            abs(float(x.get("rating", 3)) - 3),
            len(clean_text(
                x.get("review_text", "")
            ))
        ),
        reverse=True
    )

    seen = set()

    final = []

    for r in rows:

        item = r["item_id"]

        if item in seen:
            continue

        seen.add(item)

        final.append(r)

        if len(final) >= max_items:
            break

    return final


def build_interaction_prompt(row, side):

    review_title = truncate_text(row.get("review_title", ""), 80)

    review_text = truncate_text(row.get("review_text", ""), 400)

    item_title = truncate_text(row.get("item_title", ""), 80)

    item_features = truncate_text(row.get("item_features", ""), 220)

    item_description = truncate_text(row.get("item_description", ""), 260)

    item_categories = truncate_text(row.get("item_categories", ""), 180)

    focus = (
        "likes"
        if side == "positive"
        else "dislikes")

    prompt = f"""
Extract ONE short user preference statement.

TASK:
Describe what the user {focus}.

RULES:
- Output ONLY one or two sentences
- Maximum 35 words total
- No explanations
- No product names
- No marketing language
- Focus on preferences only
- Use natural language
- Be specific
- Use review text and item metadata when useful

Review title:
{review_title}

Review text:
{review_text}

Item:
{item_title}

Item features:
{item_features}

Item description:
{item_description}

Item categories:
{item_categories}

Preference statement:
""".strip()

    return prompt

def build_side_prompt(side_name, small_profiles):

    joined = "\n".join([
        f"{i+1}. {x}"
        for i, x in enumerate(small_profiles)
    ])

    focus = (
        "likes"
        if side_name == "likes"
        else "dislikes"
    )
    prompt = f"""
Generate a concise user preference profile.

TASK:
Summarize what the user {focus}.

RULES:
- Output ONE paragraph
- 40-70 words
- No product names
- No explanations
- No marketing language
- Keep only recurring preferences
- Use natural natural language
- Be direct and concise

Preference statements:
{joined}

Profile:
""".strip()

    return prompt

def build_final_prompt(pos_profile, neg_profile):

    prompt = f"""
Combine the positive and negative preferences into ONE final profile.

RULES:
- 35-60 words
- One paragraph only
- Natural language
- No product names
- Keep only important preferences
- Mention both likes and dislikes if possible

Positive profile:
{pos_profile}

Negative profile:
{neg_profile}

Final profile:
""".strip()

    return prompt

print("Generating profiles...")

processed_user_indices = set()
final_profiles = []

if Path(CHECKPOINT_PATH).exists():
    print(f"Found checkpoint file. Loading previously processed profiles...")
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                profile = json.loads(line)
                final_profiles.append(profile)
                processed_user_indices.add(profile["user_idx"])
    print(f"Loaded {len(processed_user_indices)} previously processed users")


remaining_users = [(uid, train_g) for uid, train_g in train_by_user.items() 
                   if uid not in processed_user_indices]

remaining_users = [
    (uid, train_g)
    for uid, train_g in remaining_users
    if START_USER_IDX <= uid < END_USER_IDX
]

print(f"Processing {len(remaining_users)} remaining users (already processed: {len(processed_user_indices)})")

batch_counter = 0
total_processed_in_session = 0

for idx, (uid, train_g) in enumerate(remaining_users, start=1):

    positive_rows = []
    negative_rows = []

    for _, row in train_g.iterrows():

        side = rating_side(row["rating"])

        if side is None:
            continue

        if side == "positive":
            positive_rows.append(row.to_dict())
        else:
            negative_rows.append(row.to_dict())

    positive_rows = select_best_interactions(
        positive_rows,
        MAX_SIDE_REVIEWS
    )

    negative_rows = select_best_interactions(
        negative_rows,
        MAX_SIDE_REVIEWS
    )

    positive_small = []

    for row in positive_rows:

        prompt = build_interaction_prompt(row, "positive")

        small = generate_text(prompt, max_new_tokens=25)

        positive_small.append(small)

    negative_small = []

    for row in negative_rows:

        prompt = build_interaction_prompt(row, "negative")

        small = generate_text(prompt, max_new_tokens=25)

        negative_small.append(small)

    positive_profile = ""

    if positive_small:

        prompt = build_side_prompt("likes", positive_small)

        positive_profile = generate_text(prompt, max_new_tokens=120)

    negative_profile = ""

    if negative_small:

        prompt = build_side_prompt("dislikes", negative_small)

        negative_profile = generate_text(prompt, max_new_tokens=120)

    final_profile = ""

    if positive_profile or negative_profile:

        prompt = build_final_prompt(positive_profile, negative_profile)

        final_profile = generate_text(prompt, max_new_tokens=50)

    if not final_profile:
        final_profile = (
            positive_profile
            or negative_profile
            or ""
        )

    current_total = len(processed_user_indices) + idx
    
    profile_obj = {
        "user_idx": int(uid),
        "positive_interactions": positive_small,
        "negative_interactions": negative_small,
        "positive_profile": positive_profile,
        "negative_profile": negative_profile,
        "profile": final_profile,
    }

    final_profiles.append(profile_obj)
    total_processed_in_session += 1

    if total_processed_in_session % CHECKPOINT_INTERVAL == 0:
        print(f"Checkpoint: Saving {len(final_profiles)} profiles so far...")
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            for row in final_profiles:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Checkpoint saved. Processed {current_total} total users")

    if current_total % 100 == 0:
        print(f"Processed {current_total} users total ({total_processed_in_session} in this session)")


print(f"Saving all {len(final_profiles)} profiles to {PROFILE_OUT_PATH}...")

with open(PROFILE_OUT_PATH,"w", encoding="utf-8") as f:

    for row in final_profiles:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Done.")

if Path(CHECKPOINT_PATH).exists():
    print("Cleaning up checkpoint file...")
    Path(CHECKPOINT_PATH).unlink()

with open(f"user2id_{DATASET_NAME}.json", "w") as f:
    json.dump(user2id, f)

with open(f"item2id_{DATASET_NAME}.json", "w") as f:
    json.dump(item2id, f)

print("Saved mappings.")