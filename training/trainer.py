"""Model trainers for standard and contrastive learning."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Dict, Any
import random
import numpy as np

from .config import RopeConfig, ContrastiveConfig
from .data_loader import DataLoader, BidirectionalTranslationDataset
from .utils import (
    LabelSmoothedCrossEntropyLoss,
    WarmupInverseSqrtScheduler,
    evaluate,
)


class Trainer:
    """Trainer for bidirectional translation model."""

    def __init__(
        self,
        model: nn.Module,
        config: RopeConfig,
        tokenizer_payload: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Transformer model
            config: Configuration
            tokenizer_payload: Optional tokenizer payload to save with checkpoints
        """
        self.model = model
        self.config = config
        self.tokenizer_payload = tokenizer_payload

        # Setup optimizer and loss
        self.criterion = LabelSmoothedCrossEntropyLoss(
            config.label_smoothing,
            ignore_index=0
        )
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr_base,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.weight_decay,
        )
        self.scheduler = WarmupInverseSqrtScheduler(
            self.optimizer,
            config.warmup_steps,
            config.lr_base
        )

        os.makedirs(config.save_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self.best_val_bleu = 0.0

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            epoch: Epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)

        for src_batch, tgt_batch in pbar:
            src_batch = src_batch.to(self.config.device)
            tgt_batch = tgt_batch.to(self.config.device)

            logits = self.model(src_batch, tgt_batch)
            targets = tgt_batch[:, 1:]
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{self.scheduler.get_lr():.6f}"
            )

        return total_loss / len(dataloader)

    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_bleu: float,
        is_best: bool = False,
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            val_bleu: Validation BLEU
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_bleu": val_bleu,
            "config": self.config,
            "tokenizer": self.tokenizer_payload,
        }

        if is_best:
            ckpt_path = os.path.join(self.config.save_dir, "best_model.pt")
        else:
            ckpt_path = os.path.join(self.config.save_dir, f"checkpoint_epoch_{epoch}.pt")

        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def train(
        self,
        train_dataset: BidirectionalTranslationDataset,
        valid_dataloader: DataLoader,
        sp_model,
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_dataset: Training dataset
            valid_dataloader: Validation dataloader
            sp_model: SentencePiece model
            
        Returns:
            Dictionary with training statistics
        """
        for epoch in range(1, self.config.num_epochs + 1):
            # Build dynamic train loader with curriculum learning
            from .data_loader import build_train_loader
            train_loader, vi_slice_len = build_train_loader(train_dataset, self.config, epoch)

            # Train epoch
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_bleu = evaluate(
                self.model,
                valid_dataloader,
                self.criterion,
                sp_model,
                self.config,
                calculate_bleu=True,
                max_bleu_samples=300,
            )

            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print(f"  vi→zh coverage: {vi_slice_len}/{len(train_dataset.vi2zh_indices)} "
                  f"(~{100 * self.config.vi2zh_epoch_ratio:.1f}% per epoch)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Valid Loss: {val_loss:.4f}")
            print(f"  Valid BLEU: {val_bleu:.2f}")
            print(f"  Learning Rate: {self.scheduler.get_lr():.6f}")

            # Save checkpoint
            if epoch % self.config.save_every == 0:
                ckpt_path = self.save_checkpoint(epoch, train_loss, val_loss, val_bleu)
                print(f"  ✓ Saved checkpoint to {ckpt_path}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_bleu = val_bleu
                best_path = self.save_checkpoint(epoch, train_loss, val_loss, val_bleu, is_best=True)
                print(f"  ✓ New best model! Saved to {best_path}")

        print("\nTraining finished.")
        print(f"Best valid loss: {self.best_val_loss:.4f} | Best BLEU: {self.best_val_bleu:.2f}")

        return {
            "best_val_loss": self.best_val_loss,
            "best_val_bleu": self.best_val_bleu,
            "save_dir": self.config.save_dir,
        }

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint


class ContrastiveTrainer:
    """Trainer for contrastive learning fine-tuning."""

    def __init__(
        self,
        model: nn.Module,
        projection: nn.Module,
        config: RopeConfig,
        cl_config: ContrastiveConfig,
        tokenizer_payload: Optional[Dict[str, Any]] = None,
    ):
        """Initialize contrastive trainer."""
        self.model = model
        self.projection = projection
        self.config = config
        self.cl_config = cl_config
        self.tokenizer_payload = tokenizer_payload

        self.criterion = LabelSmoothedCrossEntropyLoss(0.01, ignore_index=0)
        self.optimizer = optim.AdamW(
            list(model.parameters()) + list(projection.parameters()),
            lr=cl_config.lr_base,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01,
        )
        self.scheduler = WarmupInverseSqrtScheduler(
            self.optimizer,
            cl_config.warmup_steps,
            cl_config.lr_base
        )

        os.makedirs(cl_config.save_dir, exist_ok=True)
        self.best_cl_loss = float("inf")

    def train_epoch(
        self,
        dataloader: DataLoader,
        compute_crosslingual_loss_fn,
        global_step: int,
        epoch: int,
    ) -> tuple:
        """Train one epoch with contrastive learning."""
        self.model.train()
        self.projection.train()

        total_ce_loss = 0.0
        total_cl_loss = 0.0
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Contrastive Epoch {epoch}", leave=True)

        for batch in pbar:
            src = batch['src'].to(self.config.device)
            tgt = batch['tgt'].to(self.config.device)
            ids_zh_vi = batch['ids_zh_vi'].to(self.config.device)
            ids_vi_vi = batch['ids_vi_vi'].to(self.config.device)
            ids_vi_zh = batch['ids_vi_zh'].to(self.config.device)
            ids_zh_zh = batch['ids_zh_zh'].to(self.config.device)

            self.optimizer.zero_grad()

            # Cross-entropy loss
            logits = self.model(src, tgt)
            targets = tgt[:, 1:]
            ce_loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            # Contrastive loss with warmup
            warmup_ratio = min(1.0, global_step / max(1, self.cl_config.cross_warmup_steps))
            lambda_cross = self.cl_config.cross_lambda_max * warmup_ratio

            cl_loss = compute_crosslingual_loss_fn(
                self.model,
                self.projection,
                ids_zh_vi,
                ids_vi_vi,
                ids_vi_zh,
                ids_zh_zh,
            )

            # Total loss
            total_batch_loss = ce_loss + lambda_cross * cl_loss

            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.projection.parameters()),
                self.config.grad_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            total_ce_loss += ce_loss.item()
            total_cl_loss += cl_loss.item()
            total_loss += total_batch_loss.item()
            global_step += 1

            pbar.set_postfix({
                'ce': f'{ce_loss.item():.4f}',
                'cl': f'{cl_loss.item():.4f}',
                'λ': f'{lambda_cross:.3f}',
                'lr': f'{self.scheduler.get_lr():.6f}'
            })

        return (
            total_ce_loss / len(dataloader),
            total_cl_loss / len(dataloader),
            total_loss / len(dataloader),
            global_step,
        )

    def save_checkpoint(
        self,
        epoch: int,
        ce_loss: float,
        cl_loss: float,
        total_loss: float,
        is_best: bool = False,
    ) -> str:
        """Save contrastive training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'projection_state_dict': self.projection.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ce_loss': ce_loss,
            'cl_loss': cl_loss,
            'total_loss': total_loss,
            'config': self.config,
            'cl_config': self.cl_config,
            'tokenizer': self.tokenizer_payload,
        }

        if is_best:
            ckpt_path = os.path.join(self.cl_config.save_dir, "plan_cl_finetune_best.pt")
        else:
            ckpt_path = os.path.join(
                self.cl_config.save_dir,
                f"plan_cl_finetune_epoch_{epoch}.pt"
            )

        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def train(
        self,
        dataloader: DataLoader,
        compute_crosslingual_loss_fn,
    ) -> Dict[str, Any]:
        """Full contrastive training loop."""
        global_step = 0

        for epoch in range(1, self.cl_config.num_epochs + 1):
            ce_loss, cl_loss, total_loss, global_step = self.train_epoch(
                dataloader,
                compute_crosslingual_loss_fn,
                global_step,
                epoch,
            )

            print(f"\nContrastive Epoch {epoch}/{self.cl_config.num_epochs}")
            print(f"  CE Loss: {ce_loss:.4f}")
            print(f"  CL Loss: {cl_loss:.4f}")
            print(f"  Total Loss: {total_loss:.4f}")
            print(f"  Learning Rate: {self.scheduler.get_lr():.6f}")

            # Save checkpoint
            if epoch % self.cl_config.save_every == 0:
                ckpt_path = self.save_checkpoint(epoch, ce_loss, cl_loss, total_loss)
                print(f"  ✓ Saved checkpoint to {ckpt_path}")

            # Save best model
            if total_loss < self.best_cl_loss:
                self.best_cl_loss = total_loss
                best_path = self.save_checkpoint(epoch, ce_loss, cl_loss, total_loss, is_best=True)
                print(f"  ✓ New best model! Saved to {best_path}")

        print("\nContrastive training finished!")
        print(f"Best contrastive loss: {self.best_cl_loss:.4f}")
        print(f"Checkpoints saved in: {self.cl_config.save_dir}")

        return {
            "best_cl_loss": self.best_cl_loss,
            "save_dir": self.cl_config.save_dir,
        }
