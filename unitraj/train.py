import pytorch_lightning as pl
import torch
import wandb
torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    # ✅ 加载专家预训练权重（冻结专家，只训练Router）
    if cfg.method.model_name == 'MOE':
        # 加载 AutoBot 权重
        if hasattr(cfg, 'ckpt_path_autobot') and cfg.ckpt_path_autobot:
            autobot_ckpt = torch.load(cfg.ckpt_path_autobot, map_location='cpu')
            model.experts[0].load_state_dict(autobot_ckpt['state_dict'], strict=False)
            print(f"✅ Loaded AutoBot weights from {cfg.ckpt_path_autobot}")
            
        # 加载 Wayformer 权重  
        if hasattr(cfg, 'ckpt_path_wayformer') and cfg.ckpt_path_wayformer:
            wayformer_ckpt = torch.load(cfg.ckpt_path_wayformer, map_location='cpu')
            model.experts[1].load_state_dict(wayformer_ckpt['state_dict'], strict=False)
            print(f"✅ Loaded Wayformer weights from {cfg.ckpt_path_wayformer}")
            
        # 加载其他专家权重（如果有 MTR、SMART）
        # if hasattr(cfg, 'ckpt_path_mtr') and cfg.ckpt_path_mtr:
        #     mtr_ckpt = torch.load(cfg.ckpt_path_mtr, map_location='cpu')
        #     model.experts[2].load_state_dict(mtr_ckpt['state_dict'], strict=False)
            
        # ✅ 冻结专家参数，只训练 Router
        for expert in model.experts:
            for param in expert.parameters():
                param.requires_grad = False
        print("�� All expert parameters frozen, only Router will be trained")

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',  # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}-{val/minFDE6:.2f}-{val/minADE6:.2f}',
        save_top_k=1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        dirpath=f'./unitraj_ckpt/{cfg.exp_name}'
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name, id=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        # accumulate_grad_batches=cfg.method.Trainer.accumulate_grad_batches,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "auto", # 
        callbacks=call_backs
    )


    if cfg.ckpt_path is None and not cfg.debug:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = os.path.join('/home/zzs/zzs/', cfg.exp_name, '**', '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path,ckpt_path_autobot=cfg.ckpt_path_autobot,ckpt_path_wayformer=cfg.ckpt_path_wayformer)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    wandb.init(mode="offline")
    train()
