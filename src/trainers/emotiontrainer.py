from __future__ import print_function

import copy
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm import tqdm
from tabulate import tabulate
from src.evaluate import eval_iemocap, eval_iemocap_ce
from src.trainers.basetrainer import TrainerBase
from transformers import AlbertTokenizer


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.2, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device(features.device)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-20
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask*log_prob).sum(1)/(mask.sum(1)+1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class IemocapTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(IemocapTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.args = args
        self.text_max_len = args['text_max_len']
        self.tokenizer = AlbertTokenizer.from_pretrained(f'albert-{args["text_model_size"]}-v2')
        self.eval_func = eval_iemocap if args['loss'] == 'bce' else eval_iemocap_ce
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []

        annotations = dataloaders['train'].dataset.get_annotations()

        if self.args['loss'] == 'bce':
            self.headers = [
                ['phase (acc)', *annotations, 'average'],
                ['phase (recall)', *annotations, 'average'],
                ['phase (precision)', *annotations, 'average'],
                ['phase (f1)', *annotations, 'average'],
                ['phase (auc)', *annotations, 'average']
            ]

            n = len(annotations) + 1
            self.prev_train_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n]
            self.prev_valid_stats = copy.deepcopy(self.prev_train_stats)
            self.prev_test_stats = copy.deepcopy(self.prev_train_stats)
            self.best_valid_stats = copy.deepcopy(self.prev_train_stats)
        else:
            self.header = ['Phase', 'Acc', 'Recall', 'Precision', 'F1']
            self.best_valid_stats = [0, 0, 0, 0]

        self.best_epoch = -1

        self.criterion_con = SupConLoss(temperature=0.2)

    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            train_stats, train_thresholds = self.train_one_epoch(epoch)

            valid_stats, valid_thresholds = self.eval_one_epoch()
            test_stats, _ = self.eval_one_epoch('test', valid_thresholds)

            print('Train thresholds: ', train_thresholds)
            print('Valid thresholds: ', valid_thresholds)

            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)

            if self.args['loss'] == 'ce':
                train_stats_str = [f'{s:.4f}' for s in train_stats]
                valid_stats_str = [f'{s:.4f}' for s in valid_stats]
                test_stats_str = [f'{s:.4f}' for s in test_stats]
                print(tabulate([
                    ['Train', *train_stats_str],
                    ['Valid', *valid_stats_str],
                    ['Test', *test_stats_str]
                ], headers=self.header))
                if valid_stats[-1] > self.best_valid_stats[-1]:
                    self.best_valid_stats = valid_stats
                    self.best_epoch = epoch
                    self.earlyStop = self.args['early_stop']
                else:
                    self.earlyStop -= 1
            else:
                for i in range(len(self.headers)):
                    for j in range(len(valid_stats[i])):
                        is_pivot = (i == 3 and j == (len(valid_stats[i]) - 1)) # auc average
                        if valid_stats[i][j] > self.best_valid_stats[i][j]:
                            self.best_valid_stats[i][j] = valid_stats[i][j]
                            if is_pivot:
                                self.earlyStop = self.args['early_stop']
                                self.best_epoch = epoch
                                self.best_model = copy.deepcopy(self.model.state_dict())
                        elif is_pivot:
                            self.earlyStop -= 1

                    train_stats_str = self.make_stat(self.prev_train_stats[i], train_stats[i])
                    valid_stats_str = self.make_stat(self.prev_valid_stats[i], valid_stats[i])
                    test_stats_str = self.make_stat(self.prev_test_stats[i], test_stats[i])

                    self.prev_train_stats[i] = train_stats[i]
                    self.prev_valid_stats[i] = valid_stats[i]
                    self.prev_test_stats[i] = test_stats[i]

                    print(tabulate([
                        ['Train', *train_stats_str],
                        ['Valid', *valid_stats_str],
                        ['Test', *test_stats_str]
                    ], headers=self.headers[i]))

            if self.earlyStop == 0:
                break

        print('=== Best performance ===')
        if self.args['loss'] == 'ce':
            print(tabulate([
                [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
            ], headers=self.header))
        else:
            for i in range(len(self.headers)):
                print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][i]]], headers=self.headers[i]))

        self.save_stats()
        self.save_model()
        print('Results and model are saved!')

    def valid(self):
        valid_stats = self.eval_one_epoch()
        for i in range(len(self.headers)):
            print(tabulate([['Valid', *valid_stats[i]]], headers=self.headers[i]))
            print()

    def test(self):
        test_stats = self.eval_one_epoch('test')
        for i in range(len(self.headers)):
            print(tabulate([['Test', *test_stats[i]]], headers=self.headers[i]))
            print()
        for stat in test_stats:
            for n in stat:
                print(f'{n:.4f},', end='')
        print()

    def train_one_epoch(self, epoch):
        self.model.train()
        if self.args['model'] == 'mme2e':
            self.model.mtcnn.eval()

        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc='Train')

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            text = text.to(device=self.device)
            specgrams = specgrams.to(device=self.device)

            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)

            Y = Y.to(device=self.device) 
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                con_loss = torch.tensor(0.0).to(self.device)
                logits, features = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                loss = self.criterion(logits, Y)
                # if epoch>3:                
                # for i in range(Y.shape[1]):
                #     con_loss += self.criterion_con(torch.unsqueeze(features[:,i,:],dim=1),Y[:,i])
                # loss = task_loss + 0.01*con_loss
                loss.backward()
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                self.optimizer.step()
            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())
            pbar.set_description("train loss:{:.4f}".format(epoch_loss / data_size))
            if self.scheduler is not None:
                self.scheduler.step()
        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)
        return self.eval_func(total_logits, total_Y, self.args['dataset'])

    def eval_one_epoch(self, phase='valid', thresholds=None):
        self.model.eval()
        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            text = text.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)          
            Y = Y.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits, features = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                loss = self.criterion(logits, Y)
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)

            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())

            pbar.set_description(f"{phase} loss:{epoch_loss/data_size:.4f}")

        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)
        return self.eval_func(total_logits, total_Y, self.args['dataset'], thresholds)
