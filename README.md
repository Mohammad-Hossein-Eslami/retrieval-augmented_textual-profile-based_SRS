# retrieval-augmented_textual-profile-based_SRS


## Execution Guide

The code should be executed in the following order:

### 1. `data_processing_profile_generation.py`

This script preprocesses the raw Amazon review dataset, creates user/item mappings, generates the SASRec interaction file, and generates textual user profiles using Qwen-2.5-7B.

**Inputs:**

* Raw review file
* Item metadata file
* Local Qwen-2.5-7B model

**Main outputs:**

* `sasrec_interactions_<dataset>.txt`
* `user_profiles_<dataset>_<START>_<END>.jsonl`
* `user2id.json`
* `item2id.json`

Run:

```bash
python data_processing_profile_generation.py
```

The dataset-specific input and output paths should be set in the script before execution.

### 2. `post_processing.py`

This script processes `sasrec_interactions_<dataset>.txt` and prepares the data for SASRec. It creates the training and test user-item pairs and remaps the item IDs.

**Input:**

* `sasrec_interactions_<dataset>.txt`

**Outputs:**

* `train.txt`
* `test.txt`
* `item_mapping.csv`

Run:

```bash
python data_post_processing.py
```

`train.txt` and `test.txt` must be placed in the `data/` directory before running `main.py`.

### 3. `main.py`

This is the main training script. It uses `utils.py` and `model.py` to train the proposed SASRec model with textual user embeddings and the retrieval-augmented information.

**Inputs:**

* `data/train.txt`
* `data/test.txt`
* `data/user_profiles.jsonl`
* `utils.py`
* `model.py`

Run:

```bash
python main.py --dataset <dataset> --train_dir <train_dir>
```

The main training parameters, including batch size, learning rate, sequence length, hidden size, number of layers, and number of epochs, can be specified through command-line arguments.

**Outputs:**
A `<dataset>_<train_dir>` directory containing `args.txt`, `log.txt`, and the trained model checkpoints (`.pth`). Also evaluation results are shown every 20 epochs.

`utils.py` and `model.py` are supporting files and should not be executed separately.
