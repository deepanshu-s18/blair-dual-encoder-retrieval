"""
src/trainer.py
==============
Training loop for BLaIR dual encoder.
Saves best_model after EVERY epoch so Mac restart never loses progress.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from src.loss import infonce_loss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        args,
        output_dir: str,
        device: Optional[str] = None,
    ):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model      = model.to(self.device)
        self.loader     = train_loader
        self.args       = args
        self.output_dir = output_dir
        self.epochs     = args.epochs
        self.temperature = args.temperature

        # AMP disabled for MPS compatibility
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        total_steps  = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        self.optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
        )

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_steps - warmup_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_steps],
        )

        print(f"\n[Trainer] Device           : {self.device}")
        print(f"[Trainer] Total steps      : {total_steps:,}")
        print(f"[Trainer] Warmup steps     : {warmup_steps:,}")
        print(f"[Trainer] Learning rate    : {args.lr}")
        print(f"[Trainer] Temperature τ    : {args.temperature}")
        print(f"[Trainer] Batch size       : {args.batch_size}")
        print(f"[Trainer] Negative mode    : {args.neg_mode}")

    def _training_step(self, batch):
        self.optimizer.zero_grad()

        q_ids  = batch['query_input_ids'].to(self.device)
        q_mask = batch['query_attn_mask'].to(self.device)
        p_ids  = batch['pos_input_ids'].to(self.device)
        p_mask = batch['pos_attn_mask'].to(self.device)

        q_emb = self.model._encode_query(q_ids, q_mask)
        p_emb = self.model._encode_doc(p_ids, p_mask)
        loss  = infonce_loss(q_emb, p_emb, temperature=self.temperature)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train(self):
        history = {
            'train_loss':  [],
            'epoch_times': [],
        }

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Training for {self.epochs} epochs")
        print(f"{'='*60}")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_start  = time.time()
            running_loss = 0.0
            step_count   = 0

            for step, batch in enumerate(self.loader, 1):
                step_loss     = self._training_step(batch)
                running_loss += step_loss
                step_count   += 1

                if step % 100 == 0:
                    avg = running_loss / step_count
                    elapsed = time.time() - epoch_start
                    print(
                        f"  Epoch {epoch+1}/{self.epochs} | "
                        f"Step {step}/{len(self.loader)} | "
                        f"Loss: {avg:.4f} | "
                        f"Time: {elapsed:.0f}s"
                    )

            epoch_time = time.time() - epoch_start
            avg_loss   = running_loss / max(step_count, 1)

            history['train_loss'].append(avg_loss)
            history['epoch_times'].append(epoch_time)

            print(
                f"\n  Epoch {epoch+1}/{self.epochs} COMPLETE | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # ── SAVE AFTER EVERY EPOCH ─────────────────────────────
            best_path = os.path.join(self.output_dir, 'best_model')
            Path(best_path).mkdir(parents=True, exist_ok=True)
            self.model.save(best_path)

            hist_path = os.path.join(self.output_dir, 'training_history.json')
            with open(hist_path, 'w') as f:
                json.dump(history, f, indent=2)

            print(f"  [Checkpoint] Saved → {best_path}")
            # ──────────────────────────────────────────────────────

        return history
