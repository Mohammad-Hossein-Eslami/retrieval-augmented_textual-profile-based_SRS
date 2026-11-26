
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(
        "sentence-transformers is required. Install it with: "
        "pip install sentence-transformers"
    ) from exc


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_paths(dataset):
    global USER_PROFILE_PATH, ITEM_PROFILE_PATH
    global USER_OUTPUT_PATH, ITEM_OUTPUT_PATH, CONFIG_OUTPUT_PATH

    profile_dir = PROJECT_ROOT / "profiles" / dataset
    embedding_dir = PROJECT_ROOT / "embeddings" / dataset

    USER_PROFILE_PATH = (
        profile_dir / "user" / "user_profiles.jsonl"
    )

    ITEM_PROFILE_PATH = (
        profile_dir / "item" / "item_profiles.jsonl"
    )

    USER_OUTPUT_PATH = (
        embedding_dir / "user_embeddings.pt"
    )

    ITEM_OUTPUT_PATH = (
        embedding_dir / "item_embeddings.pt"
    )

    CONFIG_OUTPUT_PATH = (
        embedding_dir / "metadata" / "embedding_config.json"
    )


DEFAULT_MODEL = str(
    (PROJECT_ROOT / "../llms" / "bge-base-en-v1.5").resolve()
)

DEFAULT_BATCH_SIZE = 64


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode user and/or item text profiles into semantic embeddings."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g., Office_Products or Electronics",
    )

    parser.add_argument(
        "--type",
        choices=["user", "item", "both"],
        default="both",
        help="Which profiles to encode.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer-compatible embedding model. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Encoding batch size. Default: {DEFAULT_BATCH_SIZE}",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device such as cuda, cuda:0, or cpu. Defaults to automatic selection.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Optional encoder max sequence length. Leave unset to use the model default.",
    )

    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Do not L2-normalize embeddings before saving.",
    )

    return parser.parse_args()


# ============================================================
# PROFILE LOADING
# ============================================================


def load_profiles(path: Path, id_field: str) -> tuple[list[int], list[str]]:
    """Load internal IDs and encoder-ready profile strings from JSONL."""

    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    ids: list[int] = []
    texts: list[str] = []
    seen_ids: set[int] = set()
    empty_profiles = 0

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

            if id_field not in row:
                raise KeyError(
                    f"Missing '{id_field}' in {path} at line {line_number}."
                )

            if "profile" not in row:
                raise KeyError(
                    f"Missing 'profile' in {path} at line {line_number}."
                )

            try:
                idx = int(row[id_field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {id_field}={row[id_field]!r} in {path} at line {line_number}."
                ) from exc

            if idx in seen_ids:
                raise ValueError(
                    f"Duplicate {id_field}={idx} found in {path} at line {line_number}."
                )

            profile = str(row["profile"] or "").strip()
            if not profile:
                empty_profiles += 1

            seen_ids.add(idx)
            ids.append(idx)
            texts.append(profile)

    if not ids:
        raise ValueError(f"No profiles were found in {path}.")

    print(f"Loaded {len(ids):,} profiles from {path}")
    print(f"Empty profiles: {empty_profiles:,}")

    return ids, texts


# ============================================================
# MODEL
# ============================================================


def load_encoder(
    model_name: str,
    device: str | None,
    max_seq_length: int | None,
) -> SentenceTransformer:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(model_name).expanduser().resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {model_path}"
        )

    print(f"Loading local embedding model: {model_path}")
    print(f"Device: {device}")

    model = SentenceTransformer(
        str(model_path),
        device=device,
        trust_remote_code=True,
    )

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    print(f"Max sequence length: {model.max_seq_length}")

    return model


# ============================================================
# ENCODING
# ============================================================


def encode_profiles(
    model: SentenceTransformer,
    ids: list[int],
    texts: list[str],
    batch_size: int,
    normalize: bool,
) -> torch.Tensor:
    """Encode profiles and return a float32 tensor ordered by profile IDs."""

    if len(ids) != len(texts):
        raise ValueError("IDs and profile texts have different lengths.")

    # Empty profiles are legal at the file level, but we should not silently
    # turn them into meaningful-looking semantic vectors.
    empty_count = sum(not text for text in texts)
    if empty_count:
        raise ValueError(
            f"Found {empty_count:,} empty profiles. "
            "Fix profile generation before encoding."
        )

    print(f"Encoding {len(texts):,} profiles...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=normalize,
    )

    embeddings = embeddings.detach().cpu().float()

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Expected [N, D] embeddings, got shape {tuple(embeddings.shape)}."
        )

    # IDs should match tensor row positions so downstream code can use
    # embedding[user_idx] / embedding[item_idx] directly.
    if ids != sorted(ids):
        print("Profile IDs are not sorted; sorting embeddings by internal ID.")
        order = sorted(range(len(ids)), key=ids.__getitem__)
        embeddings = embeddings[order]
        ids[:] = [ids[i] for i in order]

    expected_ids = list(range(len(ids)))
    if ids != expected_ids:
        raise ValueError(
            "Internal IDs must be contiguous and zero-based for direct tensor "
            "indexing. Expected IDs 0..N-1."
        )

    return embeddings


# ============================================================
# SAVING
# ============================================================


def save_embeddings(
    embeddings: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)

    print(f"Saved: {output_path}")
    print(f"Shape: {tuple(embeddings.shape)}")
    print(f"Dtype: {embeddings.dtype}")


# ============================================================
# CONFIG METADATA
# ============================================================


def write_metadata(
    args: argparse.Namespace,
    model: SentenceTransformer,
    encoded_types: list[str],
) -> None:
    CONFIG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    embedding_dim = model.get_sentence_embedding_dimension()

    metadata = {
        "dataset": args.dataset,
        "model_name": args.model,
        "embedding_dimension": int(embedding_dim),
        "normalize_embeddings": not args.no_normalize,
        "profile_field": "profile",
        "encoded_types": encoded_types,
        "user_profile_file": str(USER_PROFILE_PATH.relative_to(PROJECT_ROOT)),
        "item_profile_file": str(ITEM_PROFILE_PATH.relative_to(PROJECT_ROOT)),
        "user_embedding_file": str(USER_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "item_embedding_file": str(ITEM_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "batch_size": int(args.batch_size),
        "max_seq_length": int(model.max_seq_length),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with CONFIG_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata: {CONFIG_OUTPUT_PATH}")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()

    configure_paths(args.dataset)

    print(f"Generating embeddings for dataset: {args.dataset}")

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0.")

    encode_user = args.type in {"user", "both"}
    encode_item = args.type in {"item", "both"}

    model = load_encoder(
        model_name=args.model,
        device=args.device,
        max_seq_length=args.max_seq_length,
    )

    encoded_types: list[str] = []

    if encode_user:
        print("\n========== USER PROFILES ==========")
        user_ids, user_texts = load_profiles(
            USER_PROFILE_PATH,
            id_field="user_idx",
        )

        user_embeddings = encode_profiles(
            model=model,
            ids=user_ids,
            texts=user_texts,
            batch_size=args.batch_size,
            normalize=not args.no_normalize,
        )

        save_embeddings(user_embeddings, USER_OUTPUT_PATH)
        encoded_types.append("user")

    if encode_item:
        print("\n========== ITEM PROFILES ==========")
        item_ids, item_texts = load_profiles(
            ITEM_PROFILE_PATH,
            id_field="item_idx",
        )

        item_embeddings = encode_profiles(
            model=model,
            ids=item_ids,
            texts=item_texts,
            batch_size=args.batch_size,
            normalize=not args.no_normalize,
        )

        save_embeddings(item_embeddings, ITEM_OUTPUT_PATH)
        encoded_types.append("item")

    write_metadata(args, model, encoded_types)

    print("\nEncoding complete.")


if __name__ == "__main__":
    main()
