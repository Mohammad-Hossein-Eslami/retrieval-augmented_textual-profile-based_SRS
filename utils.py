import sys
import copy
from typing import Dict, List
import pandas as pd
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import json


def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue):
    def sample(uid):

        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)         
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User_train = defaultdict(list)
    User_test = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    f = open('data/train.txt', 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u) + 1
        i = int(i) + 1
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)

        User_train[u].append(i)

    f = open('data/test.txt', 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u) + 1
        i = int(i) + 1
        User_test[u].append(i)

    for user in User_train:
        nfeedback = len(User_train[user])
        if nfeedback < 4:                         
            user_train[user] = User_train[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User_train[user][:-1]
            user_valid[user] = []
            user_valid[user].append(int(User_train[user][-1]))

    for user in User_test:
        user_test[user] = []
        user_test[user].append(int(User_test[user][0]))

    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, seed=12345):
    rng = np.random.RandomState(seed)
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)


    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    users = test.keys()

    for u in users:

        if len(train[u]) < 1: continue 

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.update(valid[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(1000):
            t = rng.randint(1, itemnum + 1)
            while t in rated: t = rng.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] 

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    if valid_user == 0:
        return 0.0, 0.0

    return NDCG / valid_user, HT / valid_user



def evaluate_train(model, dataset, args, seed=12345):

    rng = np.random.RandomState(seed)
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = rng.choice(np.arange(1, usernum + 1), size=10000, replace=False)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if u not in train:
            continue

        # Need at least one input item and one target item.
        if len(train[u]) < 2:
            continue

        target = train[u][-1]
        history = train[u][:-1]

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        for i in reversed(history):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(history)
        rated.add(0)

        item_idx = [target]

        for _ in range(1000):
            t = rng.randint(1, itemnum + 1)
            while t in rated:
                t = rng.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    if valid_user == 0:
        return 0.0, 0.0

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, seed=12345):
    rng = np.random.RandomState(seed)
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(1000):
            t = rng.randint(1, itemnum + 1)
            while t in rated: t = rng.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    if valid_user == 0:
        return 0.0, 0.0

    return NDCG / valid_user, HT / valid_user


@torch.no_grad()
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts

@torch.no_grad()
def embed_texts_frozen(
    texts: List[str],
    tokenizer,
    model,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=64,
    max_length=256,
) -> torch.Tensor:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        e = mean_pool(out.last_hidden_state, enc["attention_mask"])
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def get_user_text_by_uid(final_profiles_df: pd.DataFrame) -> Dict[int, str]:
    out = {}
    for _, r in final_profiles_df.iterrows():
        uid = int(r["user_idx"])
        txt = r.get("final_profile_text", None)
        if not isinstance(txt, str) or not txt.strip():
            txt = r.get("base_profile_text", "")
        out[uid] = str(txt)
    return out


def create_user_text_matrix(user_profile_df):
    user_texts = {}

    for idx, row in user_profile_df.iterrows():
        user_idx = row['user_idx']
        user_text = row['profile']
    
        user_texts[user_idx] = user_text

    user_texts = dict(sorted(user_texts.items()))
    user_texts_array = list(user_texts.values())

    return user_texts_array

def user_embedding():

    user_profile_df = pd.read_json("data/user_profiles.jsonl", lines=True)

    user_texts = create_user_text_matrix(user_profile_df)

    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_encoder = AutoModel.from_pretrained(MODEL_NAME)
    base_encoder = base_encoder.to(DEVICE)
    base_encoder.eval()

    for p in base_encoder.parameters():
       p.requires_grad = False


    user_emb = embed_texts_frozen(user_texts, tokenizer, base_encoder, device=DEVICE, batch_size=64, max_length=256)
    user_emb = user_emb.float()
    return user_emb, user_texts


