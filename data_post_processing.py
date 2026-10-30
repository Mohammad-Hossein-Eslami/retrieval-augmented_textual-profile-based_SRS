import pandas as pd
import numpy as np

INPUT_FILE = "sasrec_interactions.txt"

TRAIN_OUT = "train.txt"
TEST_OUT = "test.txt"

SEED = 42



df = pd.read_csv(INPUT_FILE, sep="\t")

df = df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

user_groups = {}

for uid, g in df.groupby("user_idx"):

    g = g.sort_values("timestamp").reset_index(drop=True)

    user_groups[uid] = g

initial_train_items = set(df[df["split"] == "train"]["item_idx"].tolist())

print("Initial train items:", len(initial_train_items))


train_pairs = []
test_pairs = []

shift_counter = 0
skipped_users = 0

for uid, g in user_groups.items():

    items = g["item_idx"].tolist()

    if len(items) < 3:
        skipped_users += 1
        continue

    test_pos = len(items) - 1
    val_pos = len(items) - 2

    while test_pos >= 2:

        candidate_test_item = items[test_pos]

        if candidate_test_item in initial_train_items:
            break

        shift_counter += 1

        test_pos -= 1
        val_pos -= 1

    if test_pos < 2:
        skipped_users += 1
        continue

    train_end = val_pos

    for i in range(train_end):

        train_pairs.append((uid, items[i]))

    train_pairs.append((uid, items[val_pos]))

    test_pairs.append((uid, items[test_pos]))

all_items = sorted(set(i for _, i in train_pairs) | set(i for _, i in test_pairs))

rng = np.random.default_rng(SEED)

shuffled_items = rng.permutation(all_items)

item_remap = {
    old_item: new_item
    for new_item, old_item in enumerate(
        shuffled_items,
        start=0
    )
}


train_pairs = [(u, item_remap[i]) for u, i in train_pairs]

test_pairs = [(u, item_remap[i]) for u, i in test_pairs]

with open(TRAIN_OUT, "w") as f:
    for u, i in train_pairs:
        f.write(f"{u} {i}\n")


with open(TEST_OUT, "w") as f:
    for u, i in test_pairs:
        f.write(f"{u} {i}\n")

mapping_df = pd.DataFrame({
    "old_item_id": list(item_remap.keys()),
    "new_item_id": list(item_remap.values())
})

mapping_df.to_csv("item_mapping.csv", index=False)


print("\nDone")

print("Train pairs:", len(train_pairs))
print("Test pairs :", len(test_pairs))

print("Users skipped:", skipped_users)

print("Shifts performed:", shift_counter)

print("Unique remapped items:", len(item_remap))
