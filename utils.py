# Code is sourced from https://github.com/Bjarten/early-stopping-pytorch

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, save_model=False, verbose=False, delta=0, path='checkpoint.pt', more_is_better=False,
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            save_model (bool): If True, save model when validation loss decrease.
                            Default: False
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.save_model = save_model
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.more_is_better = more_is_better
        self.trace_func = trace_func
        self.yellow = '\033[33m'
        self.reset = '\033[0m'

    def __call__(self, val_loss, model, optimizer, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif (self.more_is_better and score > self.best_score + self.delta) or (
                not self.more_is_better and score < self.best_score + self.delta):
            self.counter += 1
            self.trace_func(
                    f'{self.yellow}EarlyStopping ({self.counter}/{self.patience}) - best: {self.val_loss_min:.4f}{self.reset}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """Saves model when validation loss decrease."""
        if self.save_model:
            if self.verbose:
                self.trace_func(
                        f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save({
                    'epoch'               : epoch,
                    'model_state_dict'    : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss'                : val_loss,
            }, self.path)
            self.val_loss_min = val_loss
