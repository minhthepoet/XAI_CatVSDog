import os
import random
import sys

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from VAE.config import get_args, make_exp_dirs
    from VAE.data_preprocessing import build_dataloader
    from VAE.model import ConditionalVAE
else:
    from .config import get_args, make_exp_dirs
    from .data_preprocessing import build_dataloader
    from .model import ConditionalVAE


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str):
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    global_step: int,
    args,
    stats_path: str,
):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "stats_path": stats_path,
    }
    torch.save(payload, path)


def main():
    args = get_args()
    exp_root = make_exp_dirs(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    use_amp = bool(args.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("[Train] Step 0: Setup done", flush=True)
    print(f"[Train] Device={device} AMP={use_amp}", flush=True)

    print("[Train] Step 1: Build dataloader", flush=True)
    train_loader = build_dataloader(args, exp_root)
    dataset = train_loader.dataset
    stats_path = str(dataset.stats_path) if getattr(dataset, "normalize_acts", False) else ""
    print("[Train] Step 1 done", flush=True)

    print("[Train] Step 2: Build model + optimizer", flush=True)
    model = ConditionalVAE(
        latent_dim=args.latent_dim,
        target_hw=args.target_hw,
        y_channels=512,
        beta=args.beta,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("[Train] Step 2 done", flush=True)

    start_epoch = 1
    global_step = 0

    print("[Train] Step 3: Resume checkpoint (optional)", flush=True)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if use_amp and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)
    print("[Train] Step 3 done", flush=True)

    print("[Train] Step 4: Training", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"[Train] Epoch {epoch}/{args.epochs} start", flush=True)
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        seen = 0

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")
        for x_img, y_merged, _sample_id in pbar:
            x_img = x_img.to(device, non_blocking=True)
            y_merged = y_merged.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
            with autocast_ctx:
                y_hat, mu, logvar = model(x_img, y_merged)
                recon = model.recon_mse_loss(y_hat, y_merged)
                kl = model.kl_loss(mu, logvar)
                loss = recon + args.beta * kl

            if use_amp:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            bs = x_img.size(0)
            global_step += 1
            seen += bs
            running_loss += loss.item() * bs
            running_recon += recon.item() * bs
            running_kl += kl.item() * bs

            avg_loss = running_loss / max(1, seen)
            avg_recon = running_recon / max(1, seen)
            avg_kl = running_kl / max(1, seen)
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                recon=f"{avg_recon:.4f}",
                kl=f"{avg_kl:.4f}",
                lr=f"{lr:.2e}",
            )

            if global_step % args.log_every == 0:
                print(
                    f"[step {global_step}] loss={loss.item():.4f} recon={recon.item():.4f} "
                    f"kl={kl.item():.4f} lr={lr:.2e}",
                    flush=True,
                )

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(exp_root, "checkpoints", f"epoch_{epoch:04d}.pt")
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler if use_amp else None,
                epoch=epoch,
                global_step=global_step,
                args=args,
                stats_path=stats_path,
            )
            print(f"Saved checkpoint: {ckpt_path}", flush=True)
        print(f"[Train] Epoch {epoch}/{args.epochs} done", flush=True)


if __name__ == "__main__":
    main()
