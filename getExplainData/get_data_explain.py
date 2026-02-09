import argparse
import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE_STAGES = ("stem", "layer1", "layer2", "layer3", "layer4")
PROGRESS_EVERY = 10
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


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
        return builder(SimpleNamespace(dropout=0.3))

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
    return IMAGE_TRANSFORM(image).to(dtype=torch.float32)


def build_color_mask(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    arr = np.array(image, dtype=np.uint8)
    mask = (arr.sum(axis=2) > 3).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)


def scan_images(data_dir: Path):
    files = []
    for category in ("cat", "dog"):
        category_dir = data_dir / category
        if category_dir.exists():
            files.extend((category, p) for p in sorted(category_dir.glob("*.png")))
    return files


def process_image(img_path: Path, out_dir: Path, model, device):
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"[WARN] Failed to open image: {img_path} ({exc})")
        return False

    x = preprocess_image(image)
    color_mask = build_color_mask(image)

    x = x.unsqueeze(0).to(device)
    color_mask = color_mask.unsqueeze(0).to(device)
    x = x * color_mask

    if not hasattr(model, "backbone"):
        print(f"[WARN] Missing model.backbone for {img_path}")
        return False

    acts = {}
    hooks = []

    for stage_name in BACKBONE_STAGES:
        stage_module = getattr(model.backbone, stage_name, None)
        if stage_module is None:
            continue

        def make_hook(layer_name):
            def hook_fn(_module, _inputs, output):
                acts[layer_name] = output
                output.retain_grad()

            return hook_fn

        hooks.append(stage_module.register_forward_hook(make_hook(stage_name)))

    if not hooks:
        print(f"[WARN] No backbone stages found for hooks: {img_path}")
        return False

    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits_2d = logits if logits.ndim == 2 else logits.view(logits.shape[0], -1)
        pred = logits_2d.argmax(dim=1).item()
        target = logits_2d[0, pred]
        target.backward(retain_graph=False)
    except Exception as exc:
        print(f"[WARN] Forward/backward failed for {img_path} ({exc})")
        return False
    finally:
        for handle in hooks:
            handle.remove()

    acts_out = {}
    grads_out = {}

    for stage_name in BACKBONE_STAGES:
        activation = acts.get(stage_name)
        if activation is None:
            continue

        grad = activation.grad
        if grad is None:
            print(f"[WARN] Missing gradient for {img_path} layer={stage_name}")
            continue

        mask_small = F.interpolate(color_mask, size=activation.shape[-2:], mode="nearest")
        acts_out[stage_name] = (activation.detach() * mask_small).to(dtype=torch.float32).cpu()
        grads_out[stage_name] = (grad.detach() * mask_small).to(dtype=torch.float32).cpu()

    if not acts_out:
        print(f"[WARN] No valid captured layers for {img_path}")
        return False

    out_png = out_dir / img_path.name
    out_acts = out_dir / f"{img_path.stem}__acts.pt"
    out_grads = out_dir / f"{img_path.stem}__grads.pt"

    try:
        shutil.copy2(img_path, out_png)
        torch.save(acts_out, out_acts)
        torch.save(grads_out, out_grads)
    except Exception as exc:
        print(f"[WARN] Failed to save outputs for {img_path} ({exc})")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate multi-layer activations and gradients from raw_data_explain.")
    parser.add_argument("--data_dir", required=True, help="Path to raw_data_explain root (contains cat/ and dog/).")
    parser.add_argument("--out_dir", required=True, help="Base output directory.")
    parser.add_argument("--model_path", required=True, help="Path to Python file defining model architecture.")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    items = scan_images(data_dir)
    total_images = len(items)
    total_saved = 0
    total_skipped = 0

    for idx, (category, img_path) in enumerate(items, start=1):
        out_dir = out_cat if category == "cat" else out_dog
        ok = process_image(img_path=img_path, out_dir=out_dir, model=model, device=device)
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
