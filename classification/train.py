"""
Usage:
python -m classification.train \
  --data_dir /path/to/dataCatVSDog \
  --epochs 10 \
  --batch_size 64 \
  --lr 1e-4 \
  --save_dir ./runs \
  --run_name resnet18_pretrained \
  --saveEvery 10
"""

import json
import os

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import get_args, set_seed
from .data_preprocessing import build_dataloaders
from .model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_backbone_trainable(model, trainable: bool):
    for name, param in model.backbone.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def compute_accuracy(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False, log_interval=50):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = targets.float().unsqueeze(1).to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += compute_accuracy(logits.detach(), labels) * batch_size

        if is_train and (batch_idx % log_interval == 0):
            print(
                f"Batch {batch_idx}/{len(loader)} - "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_acc):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }
    torch.save(state, path)


def append_metrics(metrics_path, payload):
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main():
    args = get_args()
    set_seed(args.seed)

    device = get_device()
    print(f"[Device]: Device")
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(args)
    print(f"Class names: {class_names}")

    model = build_model(args).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, trainable=False)

    run_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    latest_ckpt = os.path.join(run_dir, "checkpoint_latest.pt")
    best_ckpt = os.path.join(run_dir, "checkpoint_best.pt")

    start_epoch = 1
    best_val_acc = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_backbone_epochs > 0:
            if epoch <= args.freeze_backbone_epochs:
                set_backbone_trainable(model, trainable=False)
            elif epoch == args.freeze_backbone_epochs + 1:
                set_backbone_trainable(model, trainable=True)

        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            log_interval=args.log_interval,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            log_interval=args.log_interval,
        )

        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        epoch_metrics = {
            "type": "epoch",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_metrics(metrics_path, epoch_metrics)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_acc=best_val_acc,
            )

        save_checkpoint(
            latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_acc=best_val_acc,
        )

        if epoch % args.saveEvery == 0:
            periodic_ckpt = os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(
                periodic_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_acc=best_val_acc,
            )

    test_loss, test_acc = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=scaler,
        use_amp=use_amp,
        log_interval=args.log_interval,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    append_metrics(
        metrics_path,
        {
            "type": "test",
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
    )


if __name__ == "__main__":
    main()
