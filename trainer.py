import os
import sys
from pprint import pprint

import torch
import torch.nn as nn

from loguru import logger
from tqdm import tqdm

from src.utils import AverageMeter, EarlyStopping


class Trainer():
    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer,
        log_writer,
        acc_metric,
        lr_scheduler,
        resume=False,
        save_dir=None,
        max_epoch=None,
        **kwargs):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_writer = log_writer
        self.acc_metric = acc_metric
        self.loss_averager = AverageMeter()
        self.resume = resume
        self.save_dir = save_dir
        self.max_epoch = max_epoch
        self.args = kwargs
        self.logger = self._logger()
        self.early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0) if self.args['early_stopping'] else None

        if not self.resume:
            self.step = 0
            self.epoch = 0
            self.best_acc = 0.0
            self.best_epoch = 0
        else:
            self.load_checkpoint()

    def print_config(self, input_size=(40, 1024, 1, 30)):   # (BCHW) for CNN, (B, seq_len, dim_size) for RNN
        self.logger.info('Current experiment configuration:')
        self.logger.info(pprint(self.args))
        # self.logger.info('-'*30)
        # self.logger.info('Model info:')
        # self.logger.info(paddle.summary(self.model, input_size))


    def _logger(self):
        logger.remove()
        f_handler = logger.add(sink=os.path.join(self.args['log_dir'], self.args['log_filename']), level='INFO', format="{time:MM-DD HH:mm:ss}: {message}")
        c_handler = logger.add(sys.stderr, level='INFO', format='{message}')
        return logger

    def train_epoch(self, train_loader):
        self.model.train()
        progressbar = tqdm(enumerate(train_loader), desc='train', total=len(train_loader))
        for step, (data, labels) in progressbar:
            # preprocess
            N, C, H, W = data.shape
            data = data.reshape(N*C, 1, H, W)
            labels = labels.reshape(-1)
            # forward part
            logits = self.model(data)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # record part
            loss_val = loss.item()
            self.loss_averager.update(loss_val)
            self.log_writer.add_scalar('train/loss_per_step', loss_val, self.step)
            
            batch_acc = self.acc_metric(logits, labels)
            avg_acc = self.acc_metric.compute()
            self.log_writer.add_scalar('train/acc_per_step', batch_acc, self.step)

            progressbar.set_postfix({
                'batch_loss': "{:.4f}".format(loss_val),
                'avg_loss': "{:.4f}".format(self.loss_averager.avg),
                'avg_acc': "{:.4f}".format(avg_acc)
            })
            self.step += 1

        logs = {}
        logs['acc'] = self.acc_metric.compute()
        logs['loss'] = self.loss_averager.avg
        self.loss_averager.reset()
        self.acc_metric.reset()
        return logs
    
    def eval_epoch(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            progressbar = tqdm(enumerate(test_loader), desc='eval', total=len(test_loader))
            for step, (data, labels) in progressbar:
                # preprocess
                N, C, H, W = data.shape
                data = data.reshape(N*C, 1, H, W)
                labels = labels.reshape(-1)
                # forward part
                logits = self.model(data)
                loss = self.loss_fn(logits, labels)

                # record part
                loss_val = loss.item()
                self.loss_averager.update(loss_val)
                
                correct = self.acc_metric.compute(logits, labels)
                batch_acc = self.acc_metric.update(correct)
                avg_acc = self.acc_metric.accumulate()

                progressbar.set_postfix({
                    'batch_loss': "{:.4f}".format(loss_val),
                    'avg_loss': "{:.4f}".format(self.loss_averager.avg),
                    'avg_acc': "{:.4f}".format(avg_acc)
                })

        logs = {}
        logs['acc'] = self.acc_metric.accumulate()
        logs['loss'] = self.loss_averager.avg
        self.loss_averager.reset()
        self.acc_metric.reset()
        return logs
    
    def fit(self, train_loader, test_loader):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch
            self.logger.info(f'Epoch: {self.epoch} / {self.max_epoch}:')
            self.logger.info(f"current LR: {[group['lr'] for group in self.optimizer.param_groups]}")

            train_logs = self.train_epoch(train_loader)
            eval_logs = self.eval_epoch(test_loader)
            self.lr_scheduler.step()

            self.log_writer.add_scalar('train/loss_per_epoch', train_logs['loss'], epoch)
            self.log_writer.add_scalar('train/acc_per_epoch', train_logs['acc'], epoch)

            self.log_writer.add_scalar('eval/loss_per_epoch', eval_logs['loss'], epoch)
            self.log_writer.add_scalar('eval/acc_per_epoch', eval_logs['acc'], epoch)
            self.log_writer.flush()
            
            self.save_checkpoint(eval_logs['acc'])
            self.logger.info(f"Epoch: {epoch} - train_loss: {train_logs['loss']:.4f} - train_acc: {train_logs['acc']:.4f} - val_loss: {eval_logs['loss']:.4f} - val acc: {eval_logs['acc']:.4f}, best acc is {self.best_acc:.4f} achieved at epoch {self.best_epoch}")
            
            if self.args['early_stopping']:
                self.early_stopping(eval_logs)
                if self.early_stopping.is_stop_traing:
                    break
            self.logger.info('')

    def save_checkpoint(self, acc):
        mid = 'latest'
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = self.epoch
            mid = 'best'

        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'acc': acc,
            'best_epoch': self.best_epoch,
            'best_acc': self.best_acc,
        }
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, mid, 'checkpoint.ptparams'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.save_dir, mid, 'checkpoint.ptopt'))
        torch.save(checkpoint, os.path.join(self.save_dir, mid, 'kwargs.pt'))
        self.logger.info('save checkpoint at {}'.format(os.path.abspath(self.save_dir)))

    def load_checkpoint(self, type_='latest'):
        """
        Args:
            type_: str, one of in (best, latest)
        """
        self.logger.info('==> Resume training...')
        self.logger.info('load checkpoint at {}\n'.format(os.path.abspath(os.path.join(self.save_dir, type_))))
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, type_, 'checkpoint.ptparams')))
        self.optimizer.load_state_dict(torch.load(os.path.join(self.save_dir, type_, 'checkpoint.ptopt')))
        checkpoint = torch.load(os.path.join(self.save_dir, type_, 'kwargs.pt'))

        self.epoch = checkpoint['epoch'] + 1
        self.step = checkpoint['step'] + 1
        self.best_acc = checkpoint['best_acc']
        self.best_epoch = checkpoint['best_epoch']


    def evaluate(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            progressbar = tqdm(enumerate(test_loader), desc='eval', total=len(test_loader))
            for step, (data, labels) in progressbar:
                # preprocess
                N, C, H, W = data.shape
                data = data.reshape(N*C, 1, H, W)
                labels = labels.reshape(-1)
                # forward part
                logits = self.model(data)
                loss = self.loss_fn(logits, labels)

                # record part
                loss_val = loss.numpy().item()
                self.loss_averager.update(loss_val)
                
                correct = self.acc_metric.compute(logits, labels)
                batch_acc = self.acc_metric.update(correct)
                avg_acc = self.acc_metric.accumulate()

                progressbar.set_postfix({
                    'batch_loss': "{:.4f}".format(loss_val),
                    'avg_loss': "{:.4f}".format(self.loss_averager.avg),
                    'avg_acc': "{:.4f}".format(avg_acc)
                })

        logs = {}
        logs['acc'] = self.acc_metric.accumulate()
        logs['loss'] = self.loss_averager.avg
        self.loss_averager.reset()
        self.acc_metric.reset()
        return logs