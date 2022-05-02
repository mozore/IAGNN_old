import math

import torch
import torch.nn as nn
from torch import sparse

from src.utils.utils import batched_index_select


class LocalGraph(nn.Module):
    def __init__(self, hidden_size):
        super(LocalGraph, self).__init__()
        self.hidden_size = hidden_size
        self.la1 = nn.Linear(self.hidden_size, 1, bias=False)
        self.la2 = nn.Linear(self.hidden_size, 1, bias=False)
        self.lb1 = nn.Linear(self.hidden_size, 1, bias=False)
        self.lb2 = nn.Linear(self.hidden_size, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, A, hidden, hs):
        batch_size = hidden.shape[0]
        N = hidden.shape[1]
        # gat node
        h_1 = hidden.repeat_interleave(N, dim=1)
        h_2 = hidden.repeat(1, N, 1)
        # gat
        att1 = self.leakyrelu(self.la1(h_1 * h_2) + self.lb1(hs.unsqueeze(1) * h_2)).view(batch_size, N, N)
        att2 = self.leakyrelu(self.la2(h_1 * h_2) + self.lb2(hs.unsqueeze(1) * h_2)).view(batch_size, N, N)
        zero_vec = -9e15 * torch.ones_like(A)
        alpha = torch.where(A.eq(1), att1, zero_vec)
        alpha = torch.where(A.eq(2), att2, alpha)
        alpha = torch.softmax(alpha, dim=2)
        h_local = torch.matmul(alpha, hidden)
        return h_local


class IntentGraph(nn.Module):
    def __init__(self, hidden_size, alpha):
        super(IntentGraph, self).__init__()
        self.hidden_size = hidden_size
        self.lq = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.lk = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.la = nn.Linear(self.hidden_size, 1, bias=False)
        self.lb = nn.Linear(self.hidden_size, 1, bias=False)
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(0.2)

    def GNNCell(self, item_emb, n_items, intent_emb, n_intents, idx_item, idx_intent):
        # # item to intent sparse attention
        att1 = self.leakyrelu(self.la(intent_emb[idx_intent[0, :]] * item_emb[idx_intent[1, :]])).squeeze()
        att1 = sparse.FloatTensor(idx_intent, att1, torch.Size([n_intents, n_items]))
        att1 = sparse.softmax(att1, dim=1)
        intent_emb = torch.sparse.mm(att1, item_emb)
        # intent to item sparse attention
        att2 = self.leakyrelu(self.lb(item_emb[idx_item[0, :]] * intent_emb[idx_item[1, :]])).squeeze()
        att2 = sparse.FloatTensor(idx_item, att2, torch.Size([n_items, n_intents]))
        att2 = sparse.softmax(att2, dim=1)
        item_nei_emb = torch.sparse.mm(att2, intent_emb)
        # combine self information
        item_emb = self.alpha * item_emb + (1 - self.alpha) * item_nei_emb
        return item_emb

    def forward(self, item_emb, n_items, intent_emb, n_intents):
        # compute similarity
        sim = torch.matmul(self.lq(item_emb), self.lk(intent_emb).t()) / math.sqrt(self.hidden_size)
        sim = torch.softmax(sim, dim=1)
        zeros = torch.zeros_like(sim)
        topk, indices = sim.topk(3, 1)
        sim = zeros.scatter(1, indices, topk)

        idx_item = (sim != 0).nonzero(as_tuple=False).t()
        idx_intent = idx_item[[1, 0]]
        item_emb = self.GNNCell(item_emb, n_items, intent_emb, n_intents, idx_item, idx_intent)
        return item_emb


class Readout(nn.Module):
    def __init__(self, hidden_size):
        super(Readout, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.l3 = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, sessions, masks, item_emb, pos_emb):
        masks = masks.float().unsqueeze(2)
        # add pos_emb
        len = sessions.shape[1]
        session_pos_emb = pos_emb.weight[: len].unsqueeze(0)
        hi = sessions + session_pos_emb
        # soft attention
        hs = torch.sum(hi * masks, 1) / torch.sum(masks, 1)
        q1 = self.l1(hi)
        q2 = self.l2(hs).unsqueeze(1)
        alpha = self.l3(torch.sigmoid(q1 + q2))
        hf = torch.sum(alpha * sessions * masks, 1)
        # add pos for session emb
        pos_n = pos_emb.weight[-1] * torch.sum(session_pos_emb * masks, 1) / torch.sum(masks, 1)
        hf = hf + pos_n
        b = item_emb.weight[1:]  # n_items x hidden_size

        scores = torch.matmul(hf, b.t())
        return scores


class IAGNN(nn.Module):
    def __init__(self, batch_size, n_items, n_intents, hidden_size, max_len, alpha):
        super(IAGNN, self).__init__()
        self.batch_size = batch_size
        self.n_items = n_items
        self.n_intents = n_intents
        self.hidden_size = hidden_size
        self.max_len = max_len
        # Aggregator
        self.local_gnn = LocalGraph(self.hidden_size)
        self.intent_gnn = IntentGraph(self.hidden_size, alpha)
        # item_emb & intent_emb & pos_emb
        self.item_emb = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.intent_emb = nn.Embedding(self.n_intents, self.hidden_size)
        self.pos_emb = nn.Embedding(self.max_len + 1, self.hidden_size)
        # readout
        self.readout = Readout(self.hidden_size)
        # resetParam
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, items, A, inputs, masks, alias_inputs):
        # l2 Normalization
        item_norm = torch.norm(self.item_emb.weight, p=2, dim=1, keepdim=True)
        self.item_emb.weight.data = self.item_emb.weight.data / item_norm
        intent_norm = torch.norm(self.intent_emb.weight, p=2, dim=1, keepdim=True)
        self.intent_emb.weight.data = self.intent_emb.weight.data / intent_norm
        # intent
        agg_emb = self.intent_gnn(self.item_emb.weight, self.n_items, self.intent_emb.weight, self.n_intents)
        h_intent = agg_emb[items]
        # local
        hi = self.item_emb(items)
        hs = torch.sum(self.item_emb(inputs) * masks.unsqueeze(2).float(), 1) / torch.sum(masks, 1, keepdim=True)
        h_local = self.local_gnn(A, hi, hs)
        # combine
        output = h_local + h_intent + h_local * h_intent
        # readout
        sessions = batched_index_select(output, 1, alias_inputs)
        scores = self.readout(sessions, masks, self.item_emb, self.pos_emb)
        return scores
