from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from multiprocessing import Process, Queue

from .base import BaseSequenceEncoder

from pathlib import Path




class DataHandler():
    def __init__(self):
        pass

    def data_partition(self, fname):

        train_path = fname / 'train_interaction.txt'
        valid_path = fname / 'valid_interaction.txt'
        test_path = fname / 'test_interaction.txt'

        usernum = 0
        itemnum = 0
        User = defaultdict(list)
        user_train = {}
        user_valid = {}
        user_test = {}
        # assume user/item index starting from 1
        f = open(train_path, 'r')
        for line in f:
            u, i = line.rstrip().split('\t')
            u = int(u) + 1
            i = int(i) + 1
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
        for user in User:
            user_train[user] = User[user][:]

        f = open(valid_path, 'r')
        for line in f:
            u, i = line.rstrip().split('\t')
            u = int(u) + 1
            i = int(i) + 1
            user_valid[u] = [i]

        f = open(test_path, 'r')
        for line in f:
            u, i = line.rstrip().split('\t')
            u = int(u) + 1
            i = int(i) + 1
            user_test[u] = [i]

        return [user_train, user_valid, user_test, usernum, itemnum]


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            p = Process(
                target=self.sample_function,
                args=(
                    User,
                    usernum,
                    itemnum,
                    batch_size,
                    maxlen,
                    self.result_queue,
                    np.random.randint(2_000_000_000),
                )
            )

            p.daemon = True
            p.start()

            self.processors.append(p)

    def sample_function(self, user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
        def sample(uid):

            # uid = np.random.randint(1, usernum + 1)
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
                neg[idx] = self.random_neq(1, itemnum + 1, ts)          # Don't need "if nxt != 0"
                nxt = i
                idx -= 1
                if idx == -1: break

            return (uid, seq, pos, neg)

        np.random.seed(SEED)
        uids = np.arange(1, usernum+1, dtype=np.int32)
        counter = 0
        while True:
            if counter % usernum == 0:
                np.random.shuffle(uids)
            one_batch = []
            for i in range(batch_size):
                one_batch.append(sample(uids[counter % usernum]))
                counter += 1
            result_queue.put(tuple(zip(*one_batch)))

    def random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        return outputs



class SASRecEncoder(BaseSequenceEncoder):
    """
    SASRec-style causal Transformer encoder.

    The model can only attend to the current and previous positions.
    """

    def __init__(
        self,
        interaction_path,
        item_dim: int,
        hidden_dim: int,
        batch_size: int,
        num_layers: int = 2,
        num_heads: int = 2,
        max_len: int = 200,
        dropout: float = 0.2,
        model_name = 'sasrec',
        device = None
    ):

        super().__init__()

        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.interaction_path = interaction_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device


        self.item_projection = (
            nn.Linear(self.item_dim, self.hidden_dim)
            if self.item_dim != self.hidden_dim
            else nn.Identity()
        )


        self.data_handler = DataHandler()

        print(interaction_path)
        dataset = self.data_handler.data_partition(interaction_path)
        [self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum] = dataset
        self.sampler = WarpSampler(self.user_train, self.usernum, self.itemnum, batch_size=self.batch_size, maxlen=self.max_len, n_workers=3)



        self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.pos_emb = nn.Embedding(num_embeddings=self.max_len + 1, embedding_dim=self.hidden_dim, padding_idx=0)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-8)

        self.bce_criterion = torch.nn.BCEWithLogitsLoss()


        for _ in range(self.num_layers):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.hidden_dim,
                                                            self.num_heads,
                                                            self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_dim, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param)
            except:
                pass

        with torch.no_grad():
            self.pos_emb.weight[0].zero_()


    def log2feats(self, log_seqs, item_emb): # TODO: fp64 and int64 as default in python, trim?

        seq_indices = torch.as_tensor(log_seqs, dtype=torch.long, device=item_emb.device)

        seqs = item_emb[seq_indices]

        seqs = seqs * (self.hidden_dim ** 0.5)

        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        
        poss *= (log_seqs != 0)
        pos_indices = torch.as_tensor(poss, dtype=torch.long, device=item_emb.device)

        seqs = seqs + self.pos_emb(pos_indices)

        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if False: #self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, item_emb, u, seq, pos, neg, device) -> torch.Tensor:

        item_emb = self.item_projection(item_emb)

        zero_row = torch.zeros(1, self.hidden_dim, device=item_emb.device, dtype=item_emb.dtype)
        item_emb = torch.cat([zero_row, item_emb], dim=0)

        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

        log_feats = self.log2feats(seq, item_emb)

        pos_indices = torch.as_tensor(pos, dtype=torch.long, device=item_emb.device)

        neg_indices = torch.as_tensor(neg, dtype=torch.long, device=item_emb.device)

        pos_embs = item_emb[pos_indices]
        neg_embs = item_emb[neg_indices]


        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)

        indices = np.where(pos != 0)
        loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])

        return log_feats, loss



    def _prepare_eval_sequence(self, user_id, split):
        """Right-aligned one-based item IDs; zero is SASRec padding."""
        import numpy as np

        history = list(self.user_train.get(user_id, []))
        if split == "test":
            history += list(self.user_valid.get(user_id, []))
        elif split != "valid":
            raise ValueError(f"Unknown split: {split!r}")

        seq = np.zeros(self.max_len, dtype=np.int64)
        history = history[-self.max_len:]
        if history:
            seq[-len(history):] = history
        return seq

    def _sample_eval_candidates(self, user_id, split, num_items, rng,
                                num_negatives=100):
        """Original SASRec sampled-ranking protocol (1 positive + negatives)."""
        import numpy as np

        targets = self.user_valid if split == "valid" else self.user_test
        positive = targets.get(user_id, [])
        if len(positive) != 1:
            raise ValueError(f"Expected one {split} target for user {user_id}.")

        # Match the supplied SASRec evaluator: exclude TRAIN items, not the
        # target item, validation items, or prior sampled negatives.
        rated = set(self.user_train.get(user_id, []))
        rated.add(0)
        if len(rated) - 1 >= num_items:
            raise ValueError(f"User {user_id} has no available negative items.")

        candidates = [positive[0]]
        for _ in range(num_negatives):
            item = int(rng.integers(1, num_items + 1))
            while item in rated:
                item = int(rng.integers(1, num_items + 1))
            candidates.append(item)
        return np.asarray(candidates, dtype=np.int64)

    @torch.no_grad()
    def predict(self, text_item_embeddings, graph_item_embeddings,
                sequences, candidate_items, already_projected=False):
        """Return (text_scores, graph_scores), each [B, C], higher is better.

        Item matrices have 0-based rows, but sequences and candidates contain
        1-based SASRec IDs (0 reserved for padding). Uses the SAME scoring
        mechanism as the original training forward(), not untrained heads.
        """
        import torch

        device = text_item_embeddings.device
        if graph_item_embeddings.device != device:
            raise ValueError("Text and graph embeddings must be on the same device.")
        if text_item_embeddings.shape != graph_item_embeddings.shape:
            raise ValueError("Text and graph item matrices must match in shape.")
        if not hasattr(self, 'item_projection'):
            raise RuntimeError(
                "SASRecEncoder.item_projection is missing. Use the same learned "
                "768->256 projection as your training forward(), and do not "
                "add a new randomly initialized prediction projection."
            )

        seq = torch.as_tensor(sequences, dtype=torch.long, device=device)
        candidates = torch.as_tensor(candidate_items, dtype=torch.long, device=device)
        if seq.ndim != 2:
            raise ValueError(f"Expected sequences [B, L], received {seq.shape}")
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0).expand(seq.shape[0], -1)
        if candidates.ndim != 2 or candidates.shape[0] != seq.shape[0]:
            raise ValueError("Expected candidate_items [C] or [B, C].")
        num_items = text_item_embeddings.shape[0]
        if torch.any(seq < 0) or torch.any(seq > num_items):
            raise ValueError("Sequence item IDs are outside 0..num_items.")
        if torch.any(candidates < 1) or torch.any(candidates > num_items):
            raise ValueError("Candidate item IDs must be in 1..num_items.")

        def score_view(raw_items):
            projected = raw_items if already_projected else self.item_projection(raw_items)  # [N, H]
            padded = torch.cat((projected.new_zeros(1, projected.shape[1]), projected), dim=0)
            hidden = self.log2feats(seq.detach().cpu().numpy(), padded)       # [B, L, H]
            final_hidden = hidden[:, -1, :]           # original SASRec final position
            selected = padded[candidates]             # [B, C, H]
            return (final_hidden.unsqueeze(1) * selected).sum(dim=-1)

        return score_view(text_item_embeddings), score_view(graph_item_embeddings)

    @torch.no_grad()
    def evaluate(self, text_item_embeddings, graph_item_embeddings,
                 split="valid", seed=42,
                 num_negatives=100, fusion_weight=0.5, fusion_weights=None):
        """SASRec's sampled-ranking HR@10 and NDCG@10 for both views + mean.

        IMPORTANT: The caller must set the model to eval() mode. This method
        does not regenerate or sample GNN graph edges.
        """
        import numpy as np


        if fusion_weights is not None:
            fusion_weights = [float(alpha) for alpha in fusion_weights]
            if not fusion_weights or any(not (0.0 <= alpha <= 1.0) for alpha in fusion_weights):
                raise ValueError("fusion_weights must be a non-empty list of values in [0,1]")

        if split not in ("valid", "test"):
            raise ValueError("split must be 'valid' or 'test'.")
        if not (0 <= fusion_weight <= 1):
            raise ValueError("fusion_weight must lie in [0, 1].")
        rng = np.random.default_rng(seed)
        num_items = text_item_embeddings.shape[0]
        targets = self.user_valid if split == "valid" else self.user_test

        users = np.arange(1, self.usernum + 1, dtype=np.int64)

        totals = {view: {"hr@10": 0.0, "ndcg@10": 0.0}
                  for view in ("text", "graph", "fused")}


        sweep_totals = (
            {alpha: {"hr@10": 0.0, "ndcg@10": 0.0} for alpha in fusion_weights}
            if fusion_weights is not None else None
        )

        # Project full item matrices ONCE per evaluation, not once per user.
        projected_text = self.item_projection(text_item_embeddings)
        projected_graph = self.item_projection(graph_item_embeddings)

        n_evaluated = 0
        for uid_raw in users:
            uid = int(uid_raw)
            if not self.user_train.get(uid) or len(targets.get(uid, [])) != 1:
                continue
            if split == "test" and len(self.user_valid.get(uid, [])) != 1:
                continue
            seq = self._prepare_eval_sequence(uid, split)
            candidates = self._sample_eval_candidates(uid, split, num_items, rng, num_negatives=num_negatives)
            # Both views see EXACTLY the same sequence and candidates.
            text_scores, graph_scores = self.predict(
                projected_text,
                projected_graph,
                sequences=seq[None, :],
                candidate_items=candidates,
                already_projected=True,
            )
            text = text_scores.squeeze(0).detach().cpu().numpy()
            graph = graph_scores.squeeze(0).detach().cpu().numpy()
            scores = {
                "text": text,
                "graph": graph,
                "fused": fusion_weight * text + (1.0 - fusion_weight) * graph,
            }
            for view, values in scores.items():
                # Position zero is the held-out positive (same as original SASRec).
                rank = int((-values).argsort().argsort()[0])
                if rank < 10:
                    totals[view]["hr@10"] += 1.0
                    totals[view]["ndcg@10"] += 1.0 / np.log2(rank + 2)
            n_evaluated += 1

            if sweep_totals is not None:
                for alpha in fusion_weights:
                    values = alpha * text + (1.0 - alpha) * graph
                    rank = int((-values).argsort().argsort()[0])
                    if rank < 10:
                        sweep_totals[alpha]["hr@10"] += 1.0
                        sweep_totals[alpha]["ndcg@10"] += 1.0 / np.log2(rank + 2)

        if n_evaluated == 0:
            raise ValueError(f"No eligible users found for {split} evaluation.")

        metrics = {"evaluated_users": n_evaluated}
        for view, view_totals in totals.items():
            for metric, total in view_totals.items():
                metrics[f"{view}_{metric}"] = total / n_evaluated


        if sweep_totals is not None:
            metrics["fusion_sweep"] = {
                alpha: {name: value / n_evaluated for name, value in totals.items()}
                for alpha, totals in sweep_totals.items()
            }
        return metrics

