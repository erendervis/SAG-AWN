
import os
import numpy as np
import torch

from data_loader.data_loader_sag import Load_Dataset_SAG, Dataset_Split_SAG, Create_Data_Loader_SAG
from util.config import Config, merge_args2cfg
from util.logger import create_logger
from util.utils import fix_seed, log_exp_settings
from util.training_sag import TrainerSAG
from models.model_sag import AWN_SAG

fix_seed(2022)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = Config('2016.10a', train=True)
cfg.device = device

os.makedirs(cfg.log_dir,   exist_ok=True)
os.makedirs(cfg.model_dir, exist_ok=True)

logger = create_logger(os.path.join(cfg.log_dir, 'log_sag.txt'))
log_exp_settings(logger, cfg)

# ─── Dataset ───────────────────────────────────────────
Signals, Labels, SNRs, snrs, mods = Load_Dataset_SAG(logger)
train_set, val_set, test_set, test_idx = Dataset_Split_SAG(
    Signals, Labels, SNRs, snrs, mods, logger)
train_loader, val_loader = Create_Data_Loader_SAG(train_set, val_set, cfg, logger)

# ─── Model ─────────────────────────────────────────────
model = AWN_SAG(
    num_classes=len(mods),
    num_levels=1, in_channels=64,
    kernel_size=3, latent_dim=320,
    regu_details=0.01, regu_approx=0.01
).to(device)

# AWN checkpoint yükle
awn_ckpt   = torch.load('./checkpoint/2016.10a_AWN.pkl', map_location=device)
model_dict = model.state_dict()
pretrained = {k: v for k, v in awn_ckpt.items() if k in model_dict}
model_dict.update(pretrained)
model.load_state_dict(model_dict)
logger.info(f'AWN checkpoint yüklendi! ({len(pretrained)} katman)')

total_params = sum(p.numel() for p in model.parameters()) / 1e6
logger.info(f'Toplam parametre: {total_params:.2f}M')

# ─── Eğitim ────────────────────────────────────────────
trainer = TrainerSAG(model, train_loader, val_loader, cfg, logger)
trainer.loop()

logger.info(f'Eğitim tamamlandı! En iyi val_acc: {trainer.best_monitor:.4f}')
