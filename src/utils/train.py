import sys
import time
import datetime

import torch
from tqdm import tqdm
from torch import optim, nn
from src.utils.utils import trans_to_cuda


def forward(model, data):
    alias_inputs, A, items, masks, labels, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    A = trans_to_cuda(A).float()
    masks = trans_to_cuda(masks).long()
    labels = trans_to_cuda(labels).long()
    inputs = trans_to_cuda(inputs).long()
    scores = model(items, A, inputs, masks, alias_inputs)
    return labels, scores


def evaluate(model, test_loader, cutoff=20):
    model.eval()
    num_samples = 0
    mrr = 0
    hit = 0
    for data in test_loader:
        labels, scores = forward(model, data)
        num_samples += scores.shape[0]
        topk = scores.topk(cutoff)[1]
        # targets = targets.numpy()
        labels = labels.unsqueeze(-1)
        hit_ranks = torch.where(topk == labels - 1)[1] + 1
        hit += hit_ranks.numel()
        mrr += hit_ranks.float().reciprocal().sum().item()
    return mrr / num_samples, hit / num_samples


class TrainRunner:
    def __init__(self, model, train_loader, test_loader, lr=1e-3, l2=1e-5, lr_dc=0.1, lr_dc_step=3, patience=3,
                 cutoff=20):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        params = model.parameters()
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=l2)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_dc_step, gamma=lr_dc)
        self.patience = patience
        self.cutoff = cutoff
        self.loss_function = nn.CrossEntropyLoss()

    def train(self, epochs):
        max_mrr = 0
        max_hit = 0
        max_mrr_epoch = 0
        max_hit_epoch = 0
        bad_counter = 0
        start = time.time()

        for epoch in tqdm(range(epochs), file=sys.stdout,
                          desc='------------------------------------------------------------'):
            print('')
            total_loss = 0
            self.model.train()
            for data in self.train_loader:
                self.optimizer.zero_grad()
                labels, scores = forward(self.model, data)
                loss = self.loss_function(scores, labels - 1)
                loss.backward()
                self.optimizer.step()
                total_loss += loss
            print(f'Epoch {epoch + 1}: Loss = {total_loss:.3f}')
            self.scheduler.step()
            print('start predicting: ', datetime.datetime.now())
            mrr, hit = evaluate(self.model, self.test_loader, self.cutoff)
            print(f'\tMRR@{self.cutoff} = {mrr * 100:.3f}%, Hit@{self.cutoff} = {hit * 100:.3f}%')

            if mrr <= max_mrr and hit <= max_hit:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            else:
                bad_counter = 0
            if max_mrr < mrr:
                max_mrr = mrr
                max_mrr_epoch = epoch + 1
            if max_hit < hit:
                max_hit = hit
                max_hit_epoch = epoch + 1
            print(
                f'\tBest: MRR@{self.cutoff} = {mrr * 100:.3f}%(Epoch:{max_mrr_epoch})'
                f', Hit@{self.cutoff} = {hit * 100:.3f}%(Epoch:{max_hit_epoch})')
        print('------------------------------------------------------------')
        end = time.time()
        print("\tRun time: %f s" % (end - start))
