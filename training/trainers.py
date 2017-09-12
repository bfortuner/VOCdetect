import time
import torch
import numpy as np
import logging
from torch.autograd import Variable

import constants as c
from predictions import pred_utils
from metrics import metric
from metrics import metric_utils
from . import utils as trn_utils



class QuickTrainer():
    def __init__(self, metrics):
        self.metrics = metrics
        self.logger = None

    def train(self, model, optim, lr_adjuster, criterion, trn_loader,
              val_loader, n_classes, threshold, n_epochs):
        start_epoch = 1
        end_epoch = start_epoch + n_epochs

        for epoch in range(start_epoch, end_epoch):
            current_lr = lr_adjuster.get_learning_rate(optim)

            ### Train ###
            trn_start_time = time.time()
            trn_metrics = trn_utils.train_model(model, trn_loader, threshold,
                    optim, criterion, lr_adjuster, epoch, n_epochs,
                    self.metrics)
            trn_msg = trn_utils.log_trn_msg(self.logger, trn_start_time,
                                            trn_metrics, current_lr, epoch)
            print(trn_msg)

            ### Test ###
            val_start_time = time.time()
            val_metrics = trn_utils.test_model(model, val_loader, threshold,
                    n_classes, criterion, self.metrics)
            val_msg = trn_utils.log_val_msg(self.logger, val_start_time,
                                            val_metrics, current_lr)
            print(val_msg)

            ### Adjust Lr ###
            if lr_adjuster.iteration_type == 'epoch':
                lr_adjuster.adjust(optim, epoch+1)


class Trainer():
    def __init__(self, trn_criterion, tst_criterion, optimizer, lr_adjuster):
        self.trn_criterion = trn_criterion
        self.tst_criterion = tst_criterion
        self.optimizer = optimizer
        self.lr_adjuster = lr_adjuster

    def train(self, model, loader, thresholds, epoch, metrics):
        model.train()

        loss_data = 0
        n_classes = loader.dataset.targets.shape[1]
        probs = np.empty((0, n_classes))
        labels = np.empty((0, n_classes))
        metric_totals = {m.name:0 for m in metrics}
        cur_iter = int((epoch-1) * len(loader))+1

        for inputs, targets, _ in loader:
            if len(targets.size()) == 1:
                targets = targets.float().view(-1, 1)
            inputs = Variable(inputs.cuda(async=True))
            targets = Variable(targets.cuda(async=True))

            ## Forward Pass
            output = model(inputs)

            ## Clear Gradients
            model.zero_grad()

            # Loss
            loss = self.trn_criterion(output, targets)

            ## Backprop
            loss.backward()
            self.optimizer.step()

            ### Adjust Lr ###
            if self.lr_adjuster.iteration_type == 'mini_batch':
                self.lr_adjuster.adjust(self.optimizer, cur_iter)
            cur_iter += 1

            loss_data += loss.data[0]
            probs = np.vstack([probs, output.data.cpu().numpy()])
            labels = np.vstack([labels, targets.data.cpu().numpy()])


        loss_data /= len(loader)
        preds = pred_utils.get_predictions(probs, thresholds)

        for m in metrics:
            score = m.evaluate(loss_data, preds, probs, labels)
            metric_totals[m.name] = score

        return metric_totals

    def test(self, model, loader, thresholds, metrics):
        model.eval()

        loss = 0
        n_classes = loader.dataset.targets.shape[1]
        probs = np.empty((0, n_classes))
        labels = np.empty((0, n_classes))
        metric_totals = {m.name:0 for m in metrics}

        for inputs, targets, _ in loader:
            if len(targets.size()) == 1:
                targets = targets.float().view(-1,1)
            inputs = Variable(inputs.cuda(async=True), volatile=True)
            targets = Variable(targets.cuda(async=True), volatile=True)

            output = model(inputs)

            loss += self.tst_criterion(output, targets).data[0]
            probs = np.vstack([probs, output.data.cpu().numpy()])
            labels = np.vstack([labels, targets.data.cpu().numpy()])

        loss /= len(loader)
        preds = pred_utils.get_predictions(probs, thresholds)

        for m in metrics:
            score = m.evaluate(loss, preds, probs, labels)
            metric_totals[m.name] = score

        return metric_totals


class MultiTargetTrainer(Trainer):
    def __init__(self, trn_criterion, tst_criterion, optimizer, lr_adjuster):
        super().__init__(trn_criterion, tst_criterion, optimizer, lr_adjuster)

    def train(self, model, loader, thresholds, epoch, metrics):
        model.train()
        n_batches = len(loader)
        cur_iter = int((epoch-1) * n_batches)+1
        metric_totals = {m.name:0 for m in metrics}

        for inputs, targets, aux_targets, _ in loader:
            if len(targets.size()) == 1:
                targets = targets.float().view(-1, 1)
            inputs = Variable(inputs.cuda(async=True))
            targets = Variable(targets.cuda(async=True))
            aux_targets = Variable(aux_targets.cuda(async=True))

            output = model(inputs)

            model.zero_grad()

            loss = self.trn_criterion(output, targets, aux_targets)
            loss_data = loss.data[0]
            labels = targets.data.cpu().numpy()
            probs = output.data.cpu().numpy()
            preds = pred_utils.get_predictions(probs, thresholds)

            for m in metrics:
                score = m.evaluate(loss_data, preds, probs, labels)
                metric_totals[m.name] += score

            loss.backward()
            self.optimizer.step()

            if self.lr_adjuster.iteration_type == 'mini_batch':
                self.lr_adjuster.adjust(self.optimizer, cur_iter)
            cur_iter += 1

        for m in metrics:
            metric_totals[m.name] /= n_batches

        return metric_totals


class MultiInputTrainer(Trainer):
    def __init__(self, trn_criterion, tst_criterion, optimizer, lr_adjuster):
        super().__init__(trn_criterion, tst_criterion, optimizer, lr_adjuster)

    def train(self, model, loader, thresholds, epoch, metrics):
        model.train()
        n_batches = len(loader)
        cur_iter = int((epoch-1) * n_batches)+1
        metric_totals = {m.name:0 for m in metrics}

        for inputs, targets, aux_inputs, _ in loader:
            if len(targets.size()) == 1:
                targets = targets.float().view(-1, 1)
            inputs = Variable(inputs.cuda(async=True))
            aux_inputs = Variable(aux_inputs.cuda(async=True))
            targets = Variable(targets.cuda(async=True))

            output = model(inputs, aux_inputs)

            model.zero_grad()

            loss = self.trn_criterion(output, targets)
            loss_data = loss.data[0]
            labels = targets.data.cpu().numpy()
            probs = output.data.cpu().numpy()
            preds = pred_utils.get_predictions(probs, thresholds)

            for m in metrics:
                score = m.evaluate(loss_data, preds, probs, labels)
                metric_totals[m.name] += score

            loss.backward()
            self.optimizer.step()

            if self.lr_adjuster.iteration_type == 'mini_batch':
                self.lr_adjuster.adjust(self.optimizer, cur_iter)
            cur_iter += 1

        for m in metrics:
            metric_totals[m.name] /= n_batches

        return metric_totals

    def test(self, model, loader, thresholds, metrics):
        model.eval()

        loss = 0
        probs = []
        labels = []
        metric_totals = {m.name:0 for m in metrics}

        for inputs, targets, aux_inputs, _ in loader:
            if len(targets.size()) == 1:
                targets = targets.float().view(-1,1)
            inputs = Variable(inputs.cuda(async=True), volatile=True)
            aux_inputs = Variable(aux_inputs.cuda(async=True), volatile=True)
            targets = Variable(targets.cuda(async=True), volatile=True)

            output = model(inputs, aux_inputs)

            loss += self.tst_criterion(output, targets).data[0]
            probs = np.vstack([probs, output.data.cpu().numpy()])
            labels = np.vstack([labels, targets.data.cpu().numpy()])

        loss /= len(loader)
        preds = pred_utils.get_predictions(probs, thresholds)
        for m in metrics:
            score = m.evaluate(loss, preds, probs, labels)
            metric_totals[m.name] = score

        return metric_totals


class ImageTargetTrainer(Trainer):
    def __init__(self, trn_criterion, tst_criterion, optimizer, 
                    lr_adjuster, n_classes, n_batches_per_step=1):
        super().__init__(trn_criterion, tst_criterion, optimizer, lr_adjuster)
        self.n_batches_per_step = n_batches_per_step

    def train(self, model, loader, thresholds, epoch, n_epochs,
              metrics):
        model.train()
        n_batches = len(loader)
        cur_iter = int((epoch-1) * n_batches)+1
        metric_totals = {m.name:0 for m in metrics}

        for inputs, targets, _, _ in loader:
            inputs = Variable(inputs.cuda(async=True))
            targets = Variable(targets.cuda(async=True))

            output = model(inputs)

            loss = self.trn_criterion(output, targets)
            loss_data = loss.data[0]
            labels = targets.data.cpu().numpy()
            probs = output.data.cpu().numpy()
            preds = pred_utils.get_predictions(probs, thresholds)

            for m in metrics:
                score = m.evaluate(loss_data, preds, probs, labels)
                metric_totals[m.name] += score

            ## Backprop (Calculate gradient)
            loss.backward()

            ## Update gradient
            if cur_iter % self.n_batches_per_step == 0:
                self.optimizer.step()
                model.zero_grad()

            if self.lr_adjuster.iteration_type == 'mini_batch':
                self.lr_adjuster.adjust(self.optimizer, cur_iter)
            cur_iter += 1

        for m in metrics:
            metric_totals[m.name] /= n_batches

        return metric_totals

    def test(self, model, loader, thresholds, metrics):
        model.eval()
        n_batches = len(loader)
        metric_totals = {m.name:0 for m in metrics}

        for inputs, targets, _, _ in loader:
            inputs = Variable(inputs.cuda(async=True), volatile=True)
            targets = Variable(targets.cuda(async=True), volatile=True)

            output = model(inputs)

            loss = self.tst_criterion(output, targets)
            loss_data = loss.data[0]
            labels = targets.data.cpu().numpy()
            probs = output.data.cpu().numpy()
            preds = pred_utils.get_predictions(probs, thresholds)

            for m in metrics:
                score = m.evaluate(loss_data, preds, probs, labels)
                metric_totals[m.name] += score

        for m in metrics:
            metric_totals[m.name] /= n_batches

        return metric_totals


class SSDTrainer(Trainer):
    def __init__(self, trn_criterion, tst_criterion, optimizer, 
                 lr_adjuster, tst_transform, idx_to_labels, detect_fn):
        super().__init__(trn_criterion, tst_criterion, optimizer, lr_adjuster)
        self.tst_transform = tst_transform
        self.idx_to_labels = idx_to_labels
        self.detect_fn = detect_fn
        
    def train(self, model, loader, thresholds, epoch, metrics):
        model.train()

        loss_data = 0
        # probs = np.empty((0, n_classes))
        # labels = np.empty((0, n_classes))
        metric_totals = {m.name:0 for m in metrics}
        cur_iter = int((epoch-1) * len(loader))+1

        for inputs, targets, _, _ in loader:
            # print(inputs.size())
            inputs = Variable(inputs.cuda(async=True))
            targets = [Variable(anno.cuda(async=True), 
                    volatile=True) for anno in targets]

            out = model(inputs)
            # print(out[0].size(), out[1].size(), out[2].size())
            self.optimizer.zero_grad()
            loss_l, loss_c = self.trn_criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            self.optimizer.step()

            if self.lr_adjuster.iteration_type == 'mini_batch':
                self.lr_adjuster.adjust(self.optimizer, cur_iter)
            cur_iter += 1

            loss_data += loss.data[0]
            # probs = np.vstack([probs, output.data.cpu().numpy()])
            # labels = np.vstack([labels, targets.data.cpu().numpy()])
            if cur_iter % 10 == 0:
                print('iter ' + repr(cur_iter) + ' || Loss: %.4f' % (
                    loss.data[0]), end='\n')

        loss_data /= len(loader)
        # preds = pred_utils.get_predictions(probs, thresholds)

        for m in metrics:
            score = m.evaluate(loss_data, preds=None, probs=None, labels=None)
            metric_totals[m.name] = score

        return metric_totals

    def test(self, model, loader, thresholds, metrics):
        model.eval()

        loss_data = 0
        n_imgs = len(loader.dataset)
        # probs = np.empty((0, n_classes))
        # labels = np.empty((0, n_classes))
        metric_totals = {m.name:0 for m in metrics}

        for i in range(n_imgs):
            img = loader.dataset.get_image(i)
            bboxes = loader.dataset.get_bboxes(i)
            x = torch.from_numpy(self.tst_transform(img)[0]).permute(2, 0, 1)
            x = Variable(x.cuda().unsqueeze(0))
            print(x.size())

            out = model(x)
            detections = self.detect_fn(out).data
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                img.shape[1], img.shape[0]])
            pred_num = 0
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= thresholds:
                    score = detections[0, i, j, 0]
                    label_name = self.idx_to_labels[i-1]
                    pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    pred_num += 1
                    print(str(pred_num)+' label: '+label_name+' score: ' +
                          str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                    j += 1

            # probs = np.vstack([probs, output.data.cpu().numpy()])
            # labels = np.vstack([labels, targets.data.cpu().numpy()])

        # loss_data /= len(loader)
        # # preds = pred_utils.get_predictions(probs, thresholds)

        # for m in metrics:
        #     score = m.evaluate(loss_data, preds=None, probs=None, labels=None)
        #     metric_totals[m.name] = score

        return metric_totals