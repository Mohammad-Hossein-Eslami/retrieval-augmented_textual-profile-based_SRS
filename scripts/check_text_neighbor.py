
import torch
import torch.nn.functional as F

# Change this to your item embeddings file.
EMBEDDINGS_PATH = "path/to/item_embeddings.pt"

NUM_SAMPLES = 5
TOP_K = 5
SEED = 42


def load_embeddings(path):
    embeddings = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    # Expected shape: [num_items, 768]
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(
            "Expected a tensor. Adjust loading if your file "
            "contains a dictionary."
        )

    return embeddings.float()


@torch.no_grad()
def inspect_neighbors(embeddings, num_samples=5, top_k=5):
    num_items = embeddings.shape[0]

    # Normalize to compute cosine similarity.
    normalized = F.normalize(embeddings, p=2, dim=-1)

    # Choose random items reproducibly.
    generator = torch.Generator().manual_seed(SEED)
    sampled_ids = torch.randperm(
        num_items, generator=generator
    )[:num_samples]

    for item_id in sampled_ids.tolist():

        # Similarity to all catalog items.
        similarities = normalized[item_id] @ normalized.T

        # Exclude the item itself.
        similarities[item_id] = -float("inf")

        scores, neighbors = torch.topk(
            similarities, k=min(top_k, num_items - 1)
        )

        print(f"\nItem {item_id}:")
        print("-" * 40)

        for rank, (neighbor, score) in enumerate(
            zip(neighbors.tolist(), scores.tolist()), 1
        ):
            print(
                f"{rank}. Item {neighbor:5d} "
                f"| Cosine similarity: {score:.4f}"
            )


if __name__ == "__main__":
    embeddings = load_embeddings(EMBEDDINGS_PATH)

    print("Embeddings shape:", embeddings.shape)

    inspect_neighbors(
        embeddings,
        num_samples=NUM_SAMPLES,
        top_k=TOP_K,
    )
