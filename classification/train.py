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

Expected data layout:
<data_dir>/data/train/{cats,dogs} and <data_dir>/data/test/{cats,dogs}
"""

import json
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

if __package__ is None or __package__ == "":
    # Allow running as: python /path/to/classification/train.py
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from classification.config import get_args, set_seed
    from classification.data_preprocessing import build_dataloaders
    from classification.model import build_model
else:
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
    if not hasattr(model, "backbone"):
        raise AttributeError("Model does not have a 'backbone' attribute for freezing.")

    for param in model.backbone.parameters():
        param.requires_grad = trainable

    if hasattr(model.backbone, "fc"):
        for param in model.backbone.fc.parameters():
            param.requires_grad = True


def compute_accuracy(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == labels).float().mean().item()


def make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    use_amp=False,
    log_interval=50,
    epoch=None,
    total_epochs=None,
    split_name="train",
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    use_tqdm = tqdm is not None and sys.stdout.isatty()
    if use_tqdm:
        desc = f"{split_name} {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else split_name
        iterator = tqdm(loader, total=len(loader), desc=desc, leave=False, dynamic_ncols=True)
    else:
        iterator = loader

    for batch_idx, (images, targets) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        labels = targets.float().unsqueeze(1).to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast_context(use_amp):
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

        if use_tqdm:
            running_loss = total_loss / max(1, total_samples)
            running_acc = total_correct / max(1, total_samples)
            iterator.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")
        elif is_train and (batch_idx % log_interval == 0):
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


def save_model_only(path, model):
    torch.save(model.state_dict(), path)


def main():
    args = get_args()
    set_seed(args.seed)

    device = get_device()
    use_amp = args.amp and device.type == "cuda"
    print(f"[Device]: {device}")
    print(f"[AMP enabled]: {use_amp}", flush=True)
    scaler = make_grad_scaler(use_amp)

    print("Building dataloaders...", flush=True)
    train_loader, val_loader, test_loader, class_names = build_dataloaders(args)
    print(f"Class names: {class_names}", flush=True)
    print(
        f"Num batches | train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}",
        flush=True,
    )

    print("Building model...", flush=True)
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
    latest_model_only = os.path.join(run_dir, "model_only_latest.pth")
    best_model_only = os.path.join(run_dir, "model_only_best.pth")
    run_meta = os.path.join(run_dir, "run_meta.json")

    with open(run_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "class_names": class_names,
            },
            f,
            indent=2,
        )

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
        print(f"Starting epoch {epoch}/{args.epochs}", flush=True)
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
            epoch=epoch,
            total_epochs=args.epochs,
            split_name="train",
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
            epoch=epoch,
            total_epochs=args.epochs,
            split_name="val",
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
            save_model_only(best_model_only, model)

        save_checkpoint(
            latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_acc=best_val_acc,
        )
        save_model_only(latest_model_only, model)

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
        split_name="test",
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
