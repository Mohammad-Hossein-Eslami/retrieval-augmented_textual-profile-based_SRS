# Research Extensions

This branch contains exploratory research conducted following the completion of my bachelor's thesis. The objective is to investigate richer user and item representations by combining semantic information, graph-based representations, and sequential modeling.

## Architectural Extensions

Compared to the original thesis implementation, this framework introduces the following components:

- **Signed Graph Neural Networks (GNNs):** Modeling positive and negative user-item interactions through separate graph message-passing mechanisms.
- **Semantic Representations:** Utilizing textual embeddings of both users and items, with trainable residual adapters.
- **Dual-View Sequential Modeling:** Processing semantic and graph-enhanced representations through a shared SASRec backbone.
- **Contrastive Learning:** Aligning semantic and graph-based sequential representations using a bidirectional contrastive objective.
- **Prototype-Based Learning:** Enriching item representations through learnable prototypes and soft assignments.
- **Prediction Fusion:** Combining recommendations from semantic and graph-based branches.

## Repository Structure

- `src/data/` — Dataset preparation and preprocessing.
- `src/profiles/` — User and item profile generation.
- `src/embeddings/` — Semantic embedding generation.
- `src/graph/` — Signed interaction graph construction.
- `src/models/` — Model architectures and representation learning modules.
- `src/training/` — Training procedures and loss functions.
- `scripts/` — Training and evaluation scripts.

## Development Status

This branch represents an ongoing exploratory extension of the original thesis. Although several architectural components have been implemented and preliminary experiments have been conducted, the overall framework remains incomplete and has not been comprehensively evaluated.

The `main` branch contains the completed undergraduate thesis implementation and its corresponding experimental results.
