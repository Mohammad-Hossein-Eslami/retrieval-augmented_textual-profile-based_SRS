# LLM-Enhanced Sequential Recommendation with Semantic Neighbor Retrieval

This repository implements a sequential recommender based on **SASRec**, augmented with **LLM-generated user preference profiles** and **retrieval of items from semantically similar users**. The model combines behavioral sequences, text-derived user representations, and neighbor-item information for next-item prediction.

The implementation uses Qwen2.5-7B to summarize preferences from historical reviews, `sentence-transformers/all-MiniLM-L6-v2` to encode those profiles, and a self-attention sequential recommender to score candidate items. Experiments cover three Amazon Reviews categories: Beauty, Video Games, and Movies & TV. The reported results compare vanilla SASRec, SASRec with textual profiles, and the combined retrieval-and-text model.

## Dataset and preprocessing

Experiments were conducted on three **Amazon Reviews 2023** categories: **Beauty**, **Video Games**, and **Movies & TV**. The dataset is available from the [Amazon Reviews 2023 project website](https://amazon-reviews-2023.github.io/). The preprocessing script defaults to `DATASET_NAME = "Video_Games"`, with inputs `Video_Games.json` and `meta_Video_Games.json`. Other categories can be processed by changing `DATASET_NAME` and supplying review and metadata files with the expected schema.

Review fields include user ID, item ASIN, rating, timestamp, review title/text, and helpful-vote count. Metadata fields include item title, features, description, and categories. If a `verified_purchase` column exists, only verified purchases are retained. Interactions without essential identifiers, ratings, or timestamps are dropped, and repeated user–item interactions are deduplicated by keeping the latest record.

## Requirements

- Python with `torch`, `transformers`, `numpy`, and `pandas`
- Access to the `sentence-transformers/all-MiniLM-L6-v2` model
- Local Qwen2.5-7B model files for profile generation
- CUDA-capable GPU recommended for LLM inference and training

## Execution guide

Run the following scripts in order from a working directory where their relative paths resolve correctly.

### Step 1 — Preprocess reviews and generate profiles

Configure `DATASET_NAME`, input paths, `MODEL_PATH`, `GPU_INDEX`, and `START_USER_IDX` / `END_USER_IDX` at the top of `data_processing_profile_generation.py`.

```bash
python data_processing_profile_generation.py
```

Main outputs:

- `sasrec_interactions_<dataset>.txt` — tab-separated user/item indices, timestamps, and initial split labels
- `user_profiles_<dataset>_<START>_<END>.jsonl` — user profiles for the selected half-open index range `[START, END)`
- `user2id_<dataset>.json` and `item2id_<dataset>.json` — initial ID mappings

Profile generation supports periodic checkpointing and resuming from its checkpoint file. Profiles generated in separate index ranges can be combined into a single file, with one profile per user.

### Step 2 — Prepare interaction files

Set `INPUT_FILE` in `data_post_processing.py` to the generated interaction file, then run:

```bash
python data_post_processing.py
```

Outputs are `train.txt`, `test.txt`, and `item_mapping.csv`. The script selects one held-out test item per eligible user, shifts the test position backward when needed to choose an item present in the original training-item set, and randomly remaps item indices with seed 42.

The resulting `train.txt` and `test.txt` files are placed under `data/` before training. This postprocessing step changes item IDs; the downstream loader expects zero-based IDs in these files and adds one internally to reserve zero for padding.

### Step 3 — Prepare profiles for training

The training code reads `data/user_profiles.jsonl`. Generated profile files are combined or renamed to that path, with one profile per expected user and alignment between `user_idx` and embedding-row order. The embedding loader sorts available profile records by user ID but does not insert placeholders for missing users.

### Step 4 — Train

```bash
python main.py --dataset Video_Games --train_dir experiment_01
```

The `--dataset` argument is used to name the output directory; the current `data_partition()` implementation reads fixed `data/train.txt` and `data/test.txt` paths regardless of that argument.

Selected command-line options and defaults:

| Argument | Default |
| --- | --- |
| `--batch_size` | `128` |
| `--lr` | `0.001` |
| `--maxlen` | `200` |
| `--hidden_units` | `200` |
| `--num_blocks` | `2` |
| `--num_heads` | `1` |
| `--dropout_rate` | `0.2` |
| `--num_epochs` | `500` |
| `--device` | `cuda` |

Outputs are stored in `<dataset>_<train_dir>/`, including `args.txt`, `log.txt`, and model checkpoints (`.pth`). Evaluation is performed every 20 epochs; checkpoints are saved when validation NDCG@10 improves and at the final epoch.

## Evaluation protocol

The implementation reports **Hit Rate at 10 (HR@10)** and **Normalized Discounted Cumulative Gain at 10 (NDCG@10)**. For each evaluated user, the positive target is ranked against **1,000 sampled negative item IDs**. This is **sampled-candidate evaluation**, not full-catalog ranking. The negative-sampling loops do not explicitly deduplicate sampled negatives.

The training loader additionally reserves the last item of sufficiently long `train.txt` histories for validation. The test evaluation uses this validation item in the input sequence, while the validation evaluation uses the preceding training history. Validation can be empty for users with short histories. The script prints training, validation, and test metrics at each evaluation interval; checkpoint selection uses validation NDCG@10.

## Experimental results

The following results compare three methods across three Amazon Reviews categories. **Retrieval + Text** denotes the combined model, **Profiles Text** denotes the text-profile variant, and **SASRec** is the baseline. Higher is better for all metrics.

| Dataset | Method | NDCG@10 | HR@10 | NDCG@5 | HR@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Beauty | Retrieval + Text | **0.1213** | **0.2132** | **0.0992** | **0.1467** |
| Beauty | Profiles Text | 0.1153 | 0.2024 | 0.0942 | 0.1391 |
| Beauty | SASRec | 0.1049 | 0.1879 | 0.0844 | 0.1285 |
| Video Games | Retrieval + Text | **0.1526** | **0.2640** | **0.1264** | **0.1860** |
| Video Games | Profiles Text | 0.1293 | 0.2333 | 0.1059 | 0.1583 |
| Video Games | SASRec | 0.1048 | 0.1854 | 0.0859 | 0.1258 |
| Movies & TV | Retrieval + Text | **0.1315** | **0.2201** | **0.1120** | **0.1598** |
| Movies & TV | Profiles Text | 0.1265 | 0.2143 | 0.1048 | 0.1501 |
| Movies & TV | SASRec | 0.1253 | 0.2051 | 0.1077 | 0.1532 |

The combined model achieves the highest NDCG@10 and HR@10 in all three categories. For **Movies & TV**, the SASRec baseline outperforms the Profiles Text variant on NDCG@5 and HR@5.

The evaluation code uses 1,000 sampled negative item IDs per positive target, so reported results represent sampled-candidate rather than full-catalog ranking.

## Source files

| File | Purpose |
| --- | --- |
| `data_processing_profile_generation.py` | Review preprocessing, interaction split, and LLM profile generation |
| `data_post_processing.py` | Held-out test selection and item remapping |
| `main.py` | Neighbor-memory construction, training, evaluation, and checkpointing |
| `model.py` | SASRec backbone and profile/retrieval fusion modules |
| `utils.py` | Data loading, sampling, metrics, and text embedding |

## Acknowledgments

This implementation builds upon the [SASRec PyTorch implementation by pmixer](https://github.com/pmixer/SASRec.pytorch). The project uses Qwen2.5-7B for preference-profile generation and `sentence-transformers/all-MiniLM-L6-v2` for profile embeddings.

The experiments use the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/), introduced in:

> Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952.
