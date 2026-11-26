from .sasrec import SASRecEncoder
from .gru4rec import GRU4RecEncoder
from .bert4rec import BERT4RecEncoder


def build_sequence_encoder(
    name: str,
    item_dim: int,
    hidden_dim: int,
    batch_size: int,
    device,
    interaction_path,
):
    name = name.lower()

    if name == "sasrec":
        return SASRecEncoder(
            interaction_path=interaction_path,
            item_dim=item_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=2,
            max_len=200,
            dropout=0.2,
            batch_size=batch_size,
            device=device
        )

    if name == "gru4rec":
        return GRU4RecEncoder(
            interaction_path=interaction_path,
            item_dim=item_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            num_layers=2,
            max_len=200,
            dropout=0.2,
            device=device,
        )

    if name == "bert4rec":
        return BERT4RecEncoder(
            item_dim=item_dim,
            hidden_dim=hidden_dim,
        )

    raise ValueError(
        f"Unknown sequence backbone: {name}"
    )