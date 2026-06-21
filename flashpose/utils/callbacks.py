"""Training callbacks for logging, checkpointing, and scheduling."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List


class Callback:
    """Base callback class."""

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any, metrics: Dict) -> None:
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        pass

    def on_batch_start(self, trainer: Any, batch_idx: int) -> None:
        pass

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        pass


class CallbackManager:
    """Manages a collection of training callbacks."""

    def __init__(self):
        self.callbacks: List[Callback] = []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer: Any, metrics: Dict) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer, metrics)

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics)

    def on_batch_start(self, trainer: Any, batch_idx: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_start(trainer, batch_idx)

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, loss)


class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""

    def __init__(self, save_dir: str | Path, save_best: bool = True, save_last: bool = True):
        self.save_dir = Path(save_dir)
        self.save_best = save_best
        self.save_last = save_last
        self.best_metric = 0.0

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        val_metric = metrics.get("val_metric", 0.0)
        if val_metric > self.best_metric:
            self.best_metric = val_metric


class LRLogger(Callback):
    """Log learning rate changes during training."""

    def __init__(self):
        self.lr_history: List[float] = []

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        if hasattr(trainer, "optimizer"):
            lr = trainer.optimizer.param_groups[0]["lr"]
            self.lr_history.append(lr)


class EarlyStopping(Callback):
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = 0.0
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        val_metric = metrics.get("val_metric", 0.0)
        if val_metric > self.best_metric + self.min_delta:
            self.best_metric = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n  Early stopping triggered at epoch {epoch+1}")


class Timer(Callback):
    """Track training time per epoch and total."""

    def __init__(self):
        self.epoch_start_time = 0.0
        self.train_start_time = 0.0
        self.epoch_times: List[float] = []

    def on_train_start(self, trainer: Any) -> None:
        self.train_start_time = time.time()

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        self.epoch_start_time = time.time()

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed)

    def on_train_end(self, trainer: Any, metrics: Dict) -> None:
        total = time.time() - self.train_start_time
        avg_epoch = sum(self.epoch_times) / max(len(self.epoch_times), 1)
        print(f"\n  Total training time: {total:.1f}s ({avg_epoch:.1f}s/epoch)")
