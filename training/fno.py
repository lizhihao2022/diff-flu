import torch
import logging
import os
from tqdm import tqdm
from utils.metrics import AverageRecord, Metrics
from neuralop.training.patching import MultigridPatching2D
import sys


class FNOTrainer:
    def __init__(self, model, device, epochs, eval_freq=5, patience=-1,
                 mg_patching_levels=0, mg_patching_padding=0, mg_patching_stitching=True,
                 verbose=False, wandb_log=False, logger=False, use_distributed=False, saving_path=None):
        self.device = device
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.patience = patience
        self.wandb = wandb_log
        self.verbose = verbose
        self.saving_path = saving_path
        if verbose:
            self.logger = logging.info if logger else print

        if mg_patching_levels > 0:
            self.mg_n_patches = 2 ** mg_patching_levels
            if verbose:
                self.logger("Multi-grid patching: {} patches".format(self.mg_n_patches))
        else:
            self.mg_n_patches = 1
            mg_patching_padding = 0
            if verbose:
                self.logger("Multi-grid patching: disabled")
        
        self.mg_patching_levels = mg_patching_levels
        self.patcher = MultigridPatching2D(
            model, levels=mg_patching_levels, padding_fraction=mg_patching_padding,
            use_distributed=use_distributed, stitching=mg_patching_stitching
        )
    
    def process(self, model, train_loader, valid_loader, test_loader, optimizer, criterion, eval_metrics, regularizer=None, scheduler=None):
        if self.verbose:
            self.logger("Start training")
            self.logger("Train dataset size: {}".format(len(train_loader.dataset)))
            self.logger("Valid dataset size: {}".format(len(valid_loader.dataset)))
            self.logger("Test dataset size: {}".format(len(test_loader.dataset)))

        best_epoch = 0
        best_metrics = None
        counter = 0
        
        for epoch in range(self.epochs):
            train_loss = self.train(model, train_loader, optimizer, criterion, scheduler, regularizer)
            if self.verbose:
                self.logger("Epoch {} | average train loss: {:.4f} | lr: {:.6f}".format(epoch, train_loss, optimizer.param_groups[0]["lr"]))
                sys.stdout.flush()
            del train_loss
            
            if (epoch + 1) % self.eval_freq == 0:
                valid_loss, valid_metrics = self.evaluate(model, valid_loader, criterion, eval_metrics, split="valid")
                if self.verbose:
                    self.logger("Epoch {} | valid loss: {:.4f} | valid metrics: {}".format(epoch, valid_loss, valid_metrics))
                valid_metrics = valid_metrics.to_dict()
                
                if not best_metrics or valid_metrics[eval_metrics[0]] < best_metrics[eval_metrics[0]]:
                    best_epoch = epoch
                    best_metrics = valid_metrics
                    torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
                    model.cuda()
                    if self.verbose:
                        self.logger("Epoch {} | save best models in {}".format(epoch, self.saving_path))
                elif self.patience != -1:
                    counter += 1
                    if counter >= self.patience:
                        if self.verbose:
                            self.logger("Early stop at epoch {}".format(epoch))
                        break
        if self.verbose:
            self.logger("Optimization Finished!")
        else:
            return
        
        # load best model
        if not best_metrics:
            torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
        else:
            model.load_state_dict(torch.load(os.path.join(self.saving_path, "best_model.pth")))
            if self.verbose:
                self.logger("Load best models at epoch {} from {}".format(best_epoch, self.saving_path))        
        model.cuda()
        
        _, valid_metrics = self.evaluate(model, valid_loader, criterion, eval_metrics, split="valid")
        self.logger("Valid metrics: {}".format(valid_metrics))
        _, test_metrics = self.evaluate(model, test_loader, criterion, eval_metrics, split="test")
        self.logger("Test metrics: {}".format(test_metrics))
        
    def train(self, model, train_loader, optimizer, criterion, scheduler, regularizer):
        train_loss = AverageRecord()
        model.cuda()
        model.train()
        with tqdm(total=len(train_loader)) as bar:
            for i, data in enumerate(train_loader):
                X, y_true = data[0], data[1]
                X, y_true = self.patcher.patch(X, y_true)
                X = X.to('cuda')
                # compute loss
                y_pred = model(X)
                y_pred = y_pred.to('cpu')
                y_pred, y_true = self.patcher.unpatch(y_pred, y_true)
                loss = criterion(y_pred, y_true)
                # compute gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # record loss and update progress bar
                train_loss.update(loss.item())
                bar.update(1)
                bar.set_postfix_str("train loss: {:.4f}".format(loss.item()))
        del X, y_true, y_pred
        return train_loss.avg
    
    def evaluate(self, model, eval_loader, criterion, metric_list, split="valid"):
        eval_metric = Metrics(metric_list, split)
        eval_loss = AverageRecord()
        model.eval()
        with torch.no_grad():
            for data in eval_loader:
                X, y_true = data[0], data[1]
                X, y_true = self.patcher.patch(X, y_true)
                X = X.to('cuda')
                # compute loss
                y_pred = model(X)
                y_pred = y_pred.to('cpu')
                y_pred, y_true = self.patcher.unpatch(y_pred, y_true)
                loss = criterion(y_pred, y_true)
                eval_loss.update(loss.item(), X.size(0))
                # compute metrics
                eval_metric.update(y_pred, y_true)
        del X, y_true, y_pred
        return eval_loss.avg, eval_metric
