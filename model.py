import numpy as np
import torch
import torch.nn.functional as F
import random


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



    
class UserEmbedding(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UserEmbedding, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = torch.nn.Linear(input_dim, output_dim)

    def forward(self, user_embedding):
        out = self.layer(user_embedding)
        return out


class NeighborItemContextRAG(torch.nn.Module):
    def __init__(self, dim, n_heads=1, dropout=0.10, max_scale=0.30):
        super().__init__()

        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_norm = torch.nn.LayerNorm(dim, eps=1e-8)

        self.scale_raw = torch.nn.Parameter(torch.tensor(2.0))
        self.max_scale = max_scale

        self.debug_stats = {}

    def forward(self, log_feats, memory_item_embs, memory_weights, valid_mask):
        memory_mask = memory_weights.le(0.0)  

        all_empty = memory_mask.all(dim=1)
        if all_empty.any():
            memory_mask[all_empty, 0] = False

        weighted_memory = memory_item_embs * memory_weights.unsqueeze(-1)

        context, attn_weights = self.cross_attn(
            query=log_feats,
            key=weighted_memory,
            value=weighted_memory,
            key_padding_mask=memory_mask,
            need_weights=False,
        )

        context = self.out_norm(context)
        context = F.normalize(context, dim=-1)

        scale = self.max_scale * torch.sigmoid(self.scale_raw)

        out = log_feats + scale * context * valid_mask
        out = F.normalize(out, dim=-1)

        # for debugging only
        # with torch.no_grad():
        #     self.debug_stats = {

        #         "item_rag_context_norm": context.norm(dim=-1).mean().item(),

        #         "item_rag_context_seq_cos": F.cosine_similarity(
        #             F.normalize(context, dim=-1),
        #             F.normalize(log_feats, dim=-1),
        #             dim=-1,
        #         ).mean().item(),

        #         "item_rag_out_cos": F.cosine_similarity(
        #             F.normalize(log_feats, dim=-1),
        #             F.normalize(out, dim=-1),
        #             dim=-1,
        #         ).mean().item(),

        #         "item_rag_delta_norm": (
        #             (out - F.normalize(log_feats, dim=-1))
        #             .norm(dim=-1)
        #             .mean()
        #             .item()
        #         ),
    
        #     }

        return out



class NeighborItemLogitRAG(torch.nn.Module):
    def __init__(self, temperature=0.07, max_alpha=0.50):
        super().__init__()

        self.temperature = temperature
        self.max_alpha = max_alpha

        self.alpha_raw = torch.nn.Parameter(torch.tensor(-2.0))

        self.debug_stats = {}

    def forward(self, candidate_embs, memory_item_embs, memory_weights):
        candidate_embs = F.normalize(candidate_embs, dim=-1)
        memory_item_embs = F.normalize(memory_item_embs, dim=-1)

        memory_mask = memory_weights.le(0.0)

        all_empty = memory_mask.all(dim=1)
        if all_empty.any():
            memory_mask = memory_mask.clone()
            memory_weights = memory_weights.clone()
            memory_mask[all_empty, 0] = False
            memory_weights[all_empty, 0] = 1.0

        # Similarity between every candidate item and every neighbour-memory item
        sim = torch.einsum("bcd,bmd->bcm", candidate_embs, memory_item_embs)

        log_weights = torch.log(memory_weights.clamp_min(1e-8)).unsqueeze(1)

        attn_logits = sim / self.temperature + log_weights
        attn_logits = attn_logits.masked_fill(memory_mask.unsqueeze(1), -1e9)

        attn = F.softmax(attn_logits, dim=-1)

        # Candidate gets high RAG score if it is close to important neighbour items.
        rag_score = (attn * sim).sum(dim=-1)  # [B, C]

        alpha = self.max_alpha * torch.sigmoid(self.alpha_raw)

        #for debugging only
        # with torch.no_grad():
        #     self.debug_stats = {
        #         "logit_rag_alpha": alpha.item(),
        #         "logit_rag_score_mean": rag_score.mean().item(),
        #         "logit_rag_score_max": rag_score.max().item(),
        #         "logit_rag_score_min": rag_score.min().item(),
        #     }

        return alpha * rag_score


class SimpleInputItemRAG(torch.nn.Module):
    def __init__(self, dim, max_scale=0.30):
        super().__init__()

        self.scale_raw = torch.nn.Parameter(torch.tensor(-2.0))
        self.max_scale = max_scale

        self.debug_stats = {}

    def forward(self, seq_item_embs, memory_item_embs, memory_weights, valid_mask):

        seq_item_embs = F.normalize(seq_item_embs, dim=-1)
        memory_item_embs = F.normalize(memory_item_embs, dim=-1)

        memory_mask = memory_weights.le(0.0)

        weights = memory_weights.clone()
        weights = weights.masked_fill(memory_mask, 0.0)

        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        norm_weights = weights / weight_sum  # [B, M]

        # One weighted neighbour-item vector per user.
        neighbor_item_context = torch.sum(
            norm_weights.unsqueeze(-1) * memory_item_embs,
            dim=1,
        )

        neighbor_item_context = F.normalize(neighbor_item_context, dim=-1)

        scale = self.max_scale * torch.sigmoid(self.scale_raw)

        # Add the same neighbour-item context to every real sequence item.
        out = seq_item_embs + scale * neighbor_item_context[:, None, :] * valid_mask
        out = F.normalize(out, dim=-1)
        out = out * valid_mask


        # for debugging only
        # with torch.no_grad():
        #     self.debug_stats = {
        #         "simple_input_rag_scale": scale.item(),
        #         "simple_input_rag_context_norm": neighbor_item_context.norm(dim=-1).mean().item(),
        #         "simple_input_rag_seq_context_cos": F.cosine_similarity(
        #             seq_item_embs,
        #             neighbor_item_context[:, None, :].expand_as(seq_item_embs),
        #             dim=-1,
        #         ).mean().item(),
        #         "simple_input_rag_out_cos": F.cosine_similarity(
        #             seq_item_embs,
        #             out,
        #             dim=-1,
        #         ).mean().item(),
        #         "simple_input_rag_nonempty_ratio": (weights.sum(dim=1) > 0).float().mean().item(),
        #     }

        return out


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args, user_embs_dim, full_user_embs):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.user_embs_dim = user_embs_dim

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() 
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()



        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.register_buffer("text_user_embs", full_user_embs)

        self.user_embedding_layer = UserEmbedding(user_embs_dim, args.hidden_units)

        self.user_layernorm = torch.nn.LayerNorm(args.hidden_units)

        self.user_gate = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_units * 2, args.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_units, args.hidden_units),
            torch.nn.Sigmoid()
        )

   
        self.item_rag = NeighborItemContextRAG(
            dim=args.hidden_units,
            n_heads=1,
            dropout=0.10,
            max_scale=0.30,
        )


        self.logit_rag = NeighborItemLogitRAG(
            temperature=0.07,
            max_alpha=0.50,
        )

        self.input_item_rag = SimpleInputItemRAG(
            dim=args.hidden_units,
            max_scale=0.30,
        )



        self.debug_stats = {}


        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def get_projected_user_embs(self, user_ids_t, use_rag=False):
        base_user_embs = self.text_user_embs[user_ids_t]

        user_embs = self.user_embedding_layer(base_user_embs.float())
        user_embs = self.user_layernorm(user_embs)
        user_embs = F.normalize(user_embs, dim=-1)

        return user_embs


    def set_neighbor_item_memory(self, item_ids, item_weights):
        self.register_buffer("neighbor_item_ids", item_ids)
        self.register_buffer("neighbor_item_weights", item_weights)
    
    @torch.no_grad()
    def build_user_bank(self):
        user_ids = torch.arange(self.user_num, device=self.dev)

        base_user_embs = self.text_user_embs[user_ids]
    
        user_embs = base_user_embs 
        user_embs = self.user_embedding_layer(user_embs.float())
        user_embs = self.user_layernorm(user_embs)
        user_embs = F.normalize(user_embs, dim=-1)

        return user_embs

    def log2feats(self, log_seqs, user_ids_t=None):
    
        log_seqs_t = torch.LongTensor(log_seqs).to(self.dev)
        valid_mask = log_seqs_t.ne(0).unsqueeze(-1).float()

        seqs = self.item_emb(log_seqs_t)

        if user_ids_t is not None and hasattr(self, "neighbor_item_ids"):
            mem_ids = self.neighbor_item_ids[user_ids_t]
            mem_weights = self.neighbor_item_weights[user_ids_t]
            

            mem_embs = self.item_emb(mem_ids)

            seqs = self.input_item_rag(
                seq_item_embs=seqs,
                memory_item_embs=mem_embs,
                memory_weights=mem_weights,
                valid_mask=valid_mask,
            )
        else:
            seqs = F.normalize(seqs, dim=-1)
            seqs = seqs * valid_mask

        seqs *= self.item_emb.embedding_dim ** 0.5


        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])

        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] 
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
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

        log_feats = self.last_layernorm(seqs) 

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):    


        user_ids_t = torch.LongTensor(user_ids).to(self.dev)

        user_embs = self.get_projected_user_embs(user_ids_t)


        log_feats = self.log2feats(log_seqs, user_ids_t)
        log_feats = F.normalize(log_feats, dim=-1) 

        gate_seq = F.normalize(log_feats, dim=-1)
        gate_user = F.normalize(user_embs, dim=-1)


        final_gate = self.user_gate(
            torch.cat([gate_seq, gate_user[:, None, :].expand_as(log_feats)], dim=-1)
        )


        if random.random() < 0.001:
            print("gate mean:", final_gate.mean().item())

        valid_mask = torch.LongTensor(log_seqs).to(self.dev).ne(0).unsqueeze(-1).float()


        log_feats = log_feats + final_gate * user_embs[:, None, :] * valid_mask

        pre_fusion = log_feats

        if self.neighbor_item_ids is not None:
            mem_ids = self.neighbor_item_ids[user_ids_t]      
            mem_weights = self.neighbor_item_weights[user_ids_t]  

            mem_embs = self.item_emb(mem_ids)                     
            mem_embs = F.normalize(mem_embs, dim=-1)

            log_feats = self.item_rag(
                log_feats=log_feats,
                memory_item_embs=mem_embs,
                memory_weights=mem_weights,
                valid_mask=valid_mask,
            )

        # for debugging only
        # with torch.no_grad():
        #     seq_mask = torch.LongTensor(log_seqs).to(self.dev).ne(0)
        #     lengths = seq_mask.sum(dim=1).clamp(min=1)
        #     last_idx = lengths - 1
        #     batch_idx = torch.arange(log_feats.size(0), device=self.dev)

        #     seq_last = pre_fusion[batch_idx, last_idx]
        #     fused_last = log_feats[batch_idx, last_idx]
        #     user_last = user_embs

        #     self.debug_stats = {
        #         "gate_mean": final_gate.mean().item(),
        #         "gate_max": final_gate.max().item(),
        #         "gate_std": final_gate.std().item(),

        #         "seq_norm": seq_last.norm(dim=-1).mean().item(),
        #         "user_norm": user_last.norm(dim=-1).mean().item(),
        #         "fused_norm": fused_last.norm(dim=-1).mean().item(),

        #         "seq_user_cos": F.cosine_similarity(seq_last, user_last, dim=-1).mean().item(),
        #         "fused_user_cos": F.cosine_similarity(fused_last, user_last, dim=-1).mean().item(),
        #         "fused_seq_cos": F.cosine_similarity(fused_last, seq_last, dim=-1).mean().item(),
        #     }

        
        pos_ids = torch.LongTensor(pos_seqs).to(self.dev)
        neg_ids = torch.LongTensor(neg_seqs).to(self.dev)

        pos_embs = self.item_emb(pos_ids)
        neg_embs = self.item_emb(neg_ids)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)


        if hasattr(self, "neighbor_item_ids"):
            mem_ids = self.neighbor_item_ids[user_ids_t]
            mem_weights = self.neighbor_item_weights[user_ids_t]

            mem_embs = self.item_emb(mem_ids)

            both_candidate_embs = torch.cat([pos_embs, neg_embs], dim=1) 

            both_rag_logits = self.logit_rag(
                candidate_embs=both_candidate_embs,
                memory_item_embs=mem_embs,
                memory_weights=mem_weights,
            )

            T = pos_embs.size(1)
            pos_rag_logits = both_rag_logits[:, :T]
            neg_rag_logits = both_rag_logits[:, T:]

            pos_logits = pos_logits + pos_rag_logits
            neg_logits = neg_logits + neg_rag_logits


        return pos_logits, neg_logits 

    def predict(self, user_ids, log_seqs, item_indices): 
        
        user_ids = user_ids - 1
        user_ids_t = torch.LongTensor(user_ids).to(self.dev)

        user_embs = self.get_projected_user_embs(user_ids_t)

        log_feats = self.log2feats(log_seqs, user_ids_t)
        log_feats = F.normalize(log_feats, dim=-1)

        gate_seq = F.normalize(log_feats.detach(), dim=-1)
        gate_user = F.normalize(user_embs, dim=-1)

        final_gate = self.user_gate(
            torch.cat([gate_seq, gate_user[:, None, :].expand_as(log_feats)], dim=-1)
        )

        valid_mask = torch.LongTensor(log_seqs).to(self.dev).ne(0).unsqueeze(-1).float()

        log_feats = log_feats + final_gate * user_embs[:, None, :] * valid_mask

        if self.neighbor_item_ids is not None:
            mem_ids = self.neighbor_item_ids[user_ids_t]
            mem_weights = self.neighbor_item_weights[user_ids_t]

            mem_embs = self.item_emb(mem_ids)
            mem_embs = F.normalize(mem_embs, dim=-1)

            log_feats = self.item_rag(
                log_feats=log_feats,
                memory_item_embs=mem_embs,
                memory_weights=mem_weights,
                valid_mask=valid_mask,
            )


        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) 

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)


        if hasattr(self, "neighbor_item_ids"):
            mem_ids = self.neighbor_item_ids[user_ids_t]
            mem_weights = self.neighbor_item_weights[user_ids_t]

            mem_embs = self.item_emb(mem_ids)

        
            if item_embs.dim() == 2:
                candidate_embs = item_embs.unsqueeze(0).expand(final_feat.size(0), -1, -1)
            else:
                candidate_embs = item_embs

            rag_logits = self.logit_rag(
                candidate_embs=candidate_embs,
                memory_item_embs=mem_embs,
                memory_weights=mem_weights,
            )

            logits = logits + rag_logits


        return logits

