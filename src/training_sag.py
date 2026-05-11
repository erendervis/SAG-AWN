
import os
import time
import torch
import pandas as pd
from torch import optim, nn
from tqdm import tqdm
from util.early_stop import EarlyStopping
from util.logger import AverageMeter


class TrainerSAG:
    def __init__(self, model, train_loader, val_loader, cfg, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.iter = 0
        self.epochs_stats = None
        self.best_monitor = 0.0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.lr_list = []

    def _snr_weights(self, snr_norm):
        """
        Düşük SNR → yüksek ağırlık
        snr_norm: 0..1 arası normalize SNR
        SNR = -20dB → weight=5, SNR=+18dB → weight=1
        """
        weights = 5.0 - 4.0 * snr_norm  # lineer: 5→1
        return weights.to(self.cfg.device)

    def loop(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.cfg.device)
        self.early_stopping = EarlyStopping(self.logger, patience=self.cfg.patience)

        backbone_params = [p for n, p in self.model.named_parameters() if 'sag' not in n]
        sag_params      = [p for n, p in self.model.named_parameters() if 'sag' in n]
        self.optimizer  = optim.Adam([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': sag_params,      'lr': 1e-3},
        ])

        for self.iter in range(self.cfg.epochs):
            self._train_epoch()
            self._val_epoch()
            if self.early_stopping.early_stop:
                self.logger.info('Early stopping')
                break

        self.epochs_stats = pd.DataFrame({
            'epoch':      range(self.iter + 1),
            'lr_list':    self.lr_list,
            'train_loss': self.train_loss_list,
            'val_loss':   self.val_loss_list,
            'train_acc':  self.train_acc_list,
            'val_acc':    self.val_acc_list,
        })

    def _train_epoch(self):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter  = AverageMeter()

        for sig, lab, snr in tqdm(self.train_loader,
                                   desc=f'Train {self.iter}/{self.cfg.epochs}',
                                   mininterval=0.3):
            sig = sig.to(self.cfg.device)
            lab = lab.to(self.cfg.device)
            snr = snr.to(self.cfg.device)

            self.optimizer.zero_grad()
            logit, _ = self.model(sig, snr)

            # Weighted loss — düşük SNR'e daha fazla ağırlık
            loss_per_sample = self.criterion(logit, lab)
            weights = self._snr_weights(snr)
            loss = (loss_per_sample * weights).mean()

            loss.backward()
            self.optimizer.step()

            acc = (logit.argmax(1) == lab).float().mean().item()
            loss_meter.update(loss.item())
            acc_meter.update(acc)

        self.lr_list.append(self.optimizer.param_groups[0]['lr'])
        self.train_loss_list.append(loss_meter.avg)
        self.train_acc_list.append(acc_meter.avg)
        self.logger.info(f'Epoch {self.iter} | Train Loss: {loss_meter.avg:.4f} | Train Acc: {acc_meter.avg:.4f}')

    def _val_epoch(self):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter  = AverageMeter()

        with torch.no_grad():
            for sig, lab, snr in tqdm(self.val_loader,
                                       desc=f'Val {self.iter}/{self.cfg.epochs}',
                                       mininterval=0.3):
                sig = sig.to(self.cfg.device)
                lab = lab.to(self.cfg.device)
                snr = snr.to(self.cfg.device)

                logit, _ = self.model(sig, snr)
                loss_per_sample = self.criterion(logit, lab)
                weights = self._snr_weights(snr)
                loss = (loss_per_sample * weights).mean()

                acc  = (logit.argmax(1) == lab).float().mean().item()
                loss_meter.update(loss.item())
                acc_meter.update(acc)

        self.val_loss_list.append(loss_meter.avg)
        self.val_acc_list.append(acc_meter.avg)
        self.logger.info(f'Epoch {self.iter} | Val Loss: {loss_meter.avg:.4f} | Val Acc: {acc_meter.avg:.4f}')

        if acc_meter.avg >= self.best_monitor:
            self.best_monitor = acc_meter.avg
            save_name = '2016.10a_AWN_SAG.pkl'
            torch.save(self.model.state_dict(),
                       os.path.join(self.cfg.model_dir, save_name))
            self.logger.info(f'✅ Model kaydedildi (val_acc: {acc_meter.avg:.4f})')

        self.early_stopping(loss_meter.avg, self.model)
