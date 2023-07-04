import torch
import logging
import os
from tqdm import tqdm
from utils.metrics import AverageRecord, Metrics


class BaseTrainer:
    def __init__(self, model, epochs, device, eval_freq=5, verbose=False, wandb_log=False, logger=False, use_distributed=False, saving_path=None):
        self.device = device
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.wandb = wandb_log
        self.verbose = verbose
        self.saving_path = saving_path
        if verbose:
            self.logger = logging.info if logger else print
    
    def process(self, model, train_loader, valid_loader, optimizer, scheduler, regularizer, criterion):
        if self.verbose:
            self.logger("Start training")
            self.logger("Train dataset size: {}".format(len(train_loader.dataset)))
            self.logger("Valid dataset size: {}".format(len(valid_loader.dataset)))
        
        best_epoch = 0
        best_metrics = None
        counter = 0
        
        for epoch in range(self.epochs):
            train_loss = self.train(model, train_loader, optimizer, scheduler, regularizer, criterion)
            if self.verbose:
                self.logger("Epoch {} | average train loss: {:.4f} | lr: {:.6f}".format(epoch, train_loss, optimizer.param_groups[0]["lr"]))
            
            if (epoch + 1) % self.eval_freq == 0:
                valid_loss, valid_metrics = self.evaluate(model, valid_loader, criterion, split="valid")
                if self.verbose:
                    self.logger("Epoch {} | valid loss: {:.4f} | valid metrics: {}".format(epoch, valid_loss, valid_metrics))
                
                if not best_metrics or valid_metrics < best_metrics:
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
        
        _, valid_metrics = self.evaluate(model, valid_loader, criterion, split="valid")
        self.logger("Valid metrics: {}".format(valid_metrics))
        _, test_metrics = self.evaluate(model, valid_loader, criterion, split="test")
        self.logger("Test metrics: {}".format(test_metrics))
        
    def train(self, model, train_loader, optimizer, scheduler, regularizer, criterion):
        train_loss = AverageRecord()
        model.cuda()
        model.train()
        with tqdm(total=len(train_loader)) as bar:
            for i, data in enumerate(train_loader):
                X = data[0].to('cuda')
                y_true = data[1]
                # compute loss
                y_pred = model(X)
                y_pred = y_pred.to('cpu')
                loss = criterion(y_pred, y_true)
                # compute gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # record loss and update progress bar
                train_loss.update(loss.item())
                bar.update(1)
                bar.set_postfix_str("train loss: {:.4f}".format(loss.item()))

        return train_loss.avg
    
    def evaluate(self, model, eval_loader, criterion, split="valid"):
        eval_metric = Metrics(split)
        eval_loss = AverageRecord()
        model.eval()
        
        for data in eval_loader:
            X = data[0].to('cuda')
            # compute loss
            y_pred = model(X)
            y_pred = y_pred.to('cpu')
            y_true = data[1]
            loss = criterion(y_pred, y_true)
            # compute metrics
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            eval_metric.update(y_pred, y_true)
            # record loss and update progress bar
            eval_loss.update(loss.item())
        
        return eval_loss.avg, eval_metric
