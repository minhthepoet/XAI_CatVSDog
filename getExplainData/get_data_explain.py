import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROGRESS_EVERY = 10


def load_model_module(model_path: Path):
    spec = importlib.util.spec_from_file_location("dynamic_model_module", str(model_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model module from: {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model_from_module(module):
    if hasattr(module, "DogCatResNetClassifier"):
        cls = getattr(module, "DogCatResNetClassifier")
        try:
            return cls(dropout=0.3)
        except TypeError:
            return cls()

    if hasattr(module, "build_model"):
        builder = getattr(module, "build_model")
        dummy_args = SimpleNamespace(dropout=0.3)
        return builder(dummy_args)

    raise RuntimeError("Model module must expose DogCatResNetClassifier or build_model(args).")


def load_checkpoint(model, ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]

    result = model.load_state_dict(state_dict, strict=False)
    print(
        f"checkpoint_loaded missing_keys={len(result.missing_keys)} "
        f"unexpected_keys={len(result.unexpected_keys)}"
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    tensor = torch.frombuffer(image.tobytes(), dtype=torch.uint8)
    tensor = tensor.view(height, width, 3).permute(2, 0, 1).contiguous()
    tensor = tensor.to(dtype=torch.float32) / 255.0

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def build_color_mask(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    tensor = torch.frombuffer(image.tobytes(), dtype=torch.uint8)
    tensor = tensor.view(height, width, 3).permute(2, 0, 1).contiguous()
    return (tensor.sum(dim=0, keepdim=True) > 3).to(dtype=torch.float32)


def process_image(img_path: Path, out_png: Path, out_act_pt: Path, out_grad_pt: Path, model, device, layer4_module):
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"[WARN] Failed to open image: {img_path} ({exc})")
        return False

    x = preprocess_image(image).unsqueeze(0).to(device)
    # Keep only pixels that are not pure black in the original masked image.
    color_mask = build_color_mask(image).unsqueeze(0).to(device)
    x = x * color_mask
    activation_holder = {}

    def hook_fn(_module, _inputs, output):
        activation_holder["act"] = output
        output.retain_grad()

    hook_handle = layer4_module.register_forward_hook(hook_fn)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits_2d = logits if logits.ndim == 2 else logits.view(logits.shape[0], -1)
        pred = logits_2d.argmax(dim=1).item()
        target = logits_2d[0, pred]
        target.backward(retain_graph=False)
    except Exception as exc:
        print(f"[WARN] Forward/backward failed for {img_path} ({exc})")
        hook_handle.remove()
        return False
    finally:
        hook_handle.remove()

    activation = activation_holder.get("act")
    if activation is None or activation.grad is None:
        print(f"[WARN] Missing activation gradient for {img_path}")
        return False

    act = activation.detach()
    grad = activation.grad.detach()
    mask_small = F.interpolate(color_mask, size=act.shape[-2:], mode="nearest")
    act = (act * mask_small).to(dtype=torch.float32).cpu()
    grad = (grad * mask_small).to(dtype=torch.float32).cpu()
    try:
        image.save(out_png, format="PNG")
        torch.save(act, out_act_pt)
        torch.save(grad, out_grad_pt)
    except Exception as exc:
        print(f"[WARN] Failed to save outputs for {img_path} ({exc})")
        return False

    return True


def scan_images(data_dir: Path):
    files = []
    for category in ("cat", "dog"):
        cat_dir = data_dir / category
        if cat_dir.exists():
            files.extend((category, p) for p in sorted(cat_dir.glob("*.png")))
    return files


def main():
    parser = argparse.ArgumentParser(description="Generate data_explain gradients from raw_data_explain images.")
    parser.add_argument("--data_dir", required=True, help="Path to raw_data_explain root (contains cat/ and dog/).")
    parser.add_argument("--out_dir", required=True, help="Base output directory.")
    parser.add_argument("--model_path", required=True, help="Path to Python file defining the model architecture.")
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint file.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir) / "data_explain"
    out_cat = out_root / "cat"
    out_dog = out_root / "dog"
    out_cat.mkdir(parents=True, exist_ok=True)
    out_dog.mkdir(parents=True, exist_ok=True)

    module = load_model_module(Path(args.model_path))
    model = build_model_from_module(module)
    load_checkpoint(model, Path(args.ckpt_path))

    if not hasattr(model, "backbone") or not hasattr(model.backbone, "layer4"):
        raise RuntimeError("Model must have model.backbone.layer4 for gradient extraction.")
    layer4_module = model.backbone.layer4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    items = scan_images(data_dir)
    total_images = len(items)
    total_saved = 0
    total_skipped = 0

    for idx, (category, img_path) in enumerate(items, start=1):
        out_dir = out_cat if category == "cat" else out_dog
        out_png = out_dir / img_path.name
        out_act_pt = out_dir / f"{img_path.stem}_layer4_act.pt"
        out_grad_pt = out_dir / f"{img_path.stem}_layer4_grad.pt"

        ok = process_image(
            img_path=img_path,
            out_png=out_png,
            out_act_pt=out_act_pt,
            out_grad_pt=out_grad_pt,
            model=model,
            device=device,
            layer4_module=layer4_module,
        )
        if ok:
            total_saved += 1
        else:
            total_skipped += 1

        if idx % PROGRESS_EVERY == 0:
            print(f"progress processed={idx}/{total_images} saved={total_saved} skipped={total_skipped}")

    print(
        f"done total_images={total_images} total_saved={total_saved} total_skipped={total_skipped} "
        f"output_dir={out_root}"
    )


if __name__ == "__main__":
    main()
