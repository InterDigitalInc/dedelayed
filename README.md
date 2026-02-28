# dedelayed

## Overview
`dedelayed` is a research codebase for delay-aware semantic segmentation that mixes a remote temporal context encoder with a lightweight local decoder. The remote server ingests delayed sequences of compressed frames to extract context, while the local device performs real-time segmentation on the current high-quality frame and fuses in the remote features to compensate for latency. Everything needed to run and extend the models lives in this repository: the PyTorch modules (`dedelayed/models`), feature-exchange layers (`dedelayed/layers/splitvid_v10.py`), dataset helpers, and the reference training script under `scripts/`.

## Highlights
- **Hybrid remote/local models** – `dedelayed/models/dedelayed_v1` stitches EfficientViT-L1 + EfficientViT-B0 or EfficientViT-L1 + MSTransformer2D pairs, exposing both the remote and local halves as standalone modules when you need to deploy them on different devices.
- **Temporal EfficientViT backbone** – `dedelayed/models/backbones/evit_vd.py` wraps EfficientViT with a 3D transformer, learnable delay embeddings, and an optional multi-task head so the remote encoder understands motion over a window of `X_REMOTE_LEN=4` frames.
- **Custom fusion layers** – `dedelayed/layers/splitvid_v10.py` implements reusable ND convolutions, grouped norms, residual wrappers, and the `PrepoolBlock`/`PostpoolBlock` pair used to pool remote context into `stage2_backbone` features before transmitting them to the local model.
- **Normalization & palette utilities** – `dedelayed/data/normalization.py` offers consistent conversions between `[0,1]`, `[-1,1]`, ImageNet, and CLIP statistics, while `dedelayed/datasets/cityscapes.py` decodes class palettes for qualitative inspection.
- **End-to-end training script** – `scripts/train.py` handles Hugging Face dataset loading, aggressive spatial/photometric augmentation, optimizer & scheduler setup (Adan + cosine schedule), validation using `torchmetrics.JaccardIndex`, and checkpoint exports.

## Repository Layout
- `dedelayed/registry.py` – lightweight decorator-based registry so models can be looked up by string name.
- `dedelayed/data/` – normalization helpers and other data utilities.
- `dedelayed/datasets/` – dataset-specific helpers (currently Cityscapes color decoding).
- `dedelayed/layers/` – building blocks for analysis/synthesis convs, multi-head attention, MBConv variants, pooling MLPs, etc.
- `dedelayed/models/backbones/` – temporal EfficientViT (`evit_vd.py`) and multi-scale transformer (`mstransformer2d.py`) implementations.
- `dedelayed/models/dedelayed_v1/` – composite remote/local models that pass the remote `stage2_backbone` tensor through a lightweight bottleneck before fusing it in the local decoder.
- `scripts/train.py` – PyTorch training + evaluation entry point with dataset loading, augmentation, logging, and checkpoint management.

## Installation
Requirements: Python 3.8–3.11 and a PyTorch build (CUDA recommended). Install the package and its runtime dependencies directly from the repo root:

```bash
git clone https://github.com/InterDigitalInc/dedelayed.git
cd dedelayed
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional utilities listed in `pyproject.toml` under `[dependency-groups.secondary]` (e.g., `av`, `torchmetrics`, `matplotlib`, `onnx`) can be installed on demand:

```bash
pip install av fastprogress hsluv matplotlib onnx torchmetrics
```

## Quick Start (Inference)
Instantiate any registered model via `dedelayed.registry.MODELS`. Remote modules expect tensors shaped `[B, 3, F, H, W]` (F=4 by default) and local modules expect `[B, 3, H, W]`.

```python
import torch
from dedelayed.registry import MODELS

remote = MODELS["dedelayed_v1_efficientvitl1_efficientvitb0_remote"](
    normalization_src="01",
    temporal_depth=4,
    temporal_width=96,
)
local = MODELS["dedelayed_v1_efficientvitl1_efficientvitb0_local"](
    normalization_src="01",
)
model = MODELS["dedelayed_v1_efficientvitl1_efficientvitb0"](
    remote_model=remote,
    local_model=local,
)

x_remote = torch.rand(2, 3, 4, 720, 1248)  # delayed remote clip
x_local = torch.rand(2, 3, 480, 832)        # current local frame
out = model(x_local, x_remote, past_ticks=5)
segmentation = out["seg_logits"].argmax(dim=1)
```

Switch to `dedelayed_v1_efficientvitl1_mstransformer2d` if you want an MSTransformer2D local decoder instead of EfficientViT-B0; both share the same remote interface and feature-fusion contract.

## Training Pipeline
`scripts/train.py` is a fully worked training script you can adapt to your dataset.

1. **Configure datasets.** Set `meta["hp"]["dataset"]` to Hugging Face dataset builders or local Arrow files. Each split must expose key/value pairs for every frame in a 16-frame clip (see next section).
2. **Tune hyperparameters.** Update `meta["hp"]["config"]` for image sizes, fps, delays (`min_delay`/`max_delay`), optimizer ranges, and worker counts. The defaults freeze both EfficientViT feature extractors and only train the fusion modules/head.
3. **Adjust model kwargs.** `meta["hp"]["model"]` lets you override remote/local kwargs (temporal depth/width, send bottleneck width, etc.).
4. **Run training.**
   ```bash
   python scripts/train.py
   ```
   The script prints which parameters are frozen vs. trainable, streams progress bars via `tqdm`, runs `evaluate()` every epoch (mIoU on validation data with configurable compression/delay), and writes checkpoints plus training curves to `checkpoints/*.pth`.

The script compiles the model (`torch.compile(..., mode="max-autotune")`), uses Adan with cosine-like `raised_cosine_scheduler`, and clips gradients to 5.0. Validation defaults to `past_ticks=5` with WEBP-compressed remote frames to emulate lossy upstream video.

## Dataset Expectations
The reference dataloader assumes each Hugging Face sample looks roughly like this:

- `near_lossless_{i}` – current-frame RGB images (PIL) for `i ∈ [0,15]`, used for both the delayed remote frames and the on-device local frame during training.
- `original_{i}` – uncompressed RGB images for evaluation; the script can optionally re-encode them via `DEFAULT_EVAL_COMPRESSION` before feeding the remote tower.
- `label_{i}` / `label_hq_{i}` – semantic masks stored as single-channel PIL images using Cityscapes IDs (ignore index 255). `dedelayed/datasets/cityscapes.py` contains helpers to visualize these masks.

Temporal windows are sampled by `sample_temporal_indices` so that `X_REMOTE_LEN=4` past frames plus `past_ticks` (0–5 by default) precede the local frame. Make sure every record exposes at least 16 sequential frames so the sampling logic remains valid.

## Model Zoo & Extension
The registry currently exposes the following entries:

- `dedelayed_v1_efficientvitl1_efficientvitb0_remote`
- `dedelayed_v1_efficientvitl1_efficientvitb0_local`
- `dedelayed_v1_efficientvitl1_efficientvitb0`
- `dedelayed_v1_efficientvitl1_mstransformer2d_remote`
- `dedelayed_v1_efficientvitl1_mstransformer2d_local`
- `dedelayed_v1_efficientvitl1_mstransformer2d`

Creating a new variant is as simple as decorating a subclass of `torch.nn.Module` with `@register_model("your_name")` (see `dedelayed/registry.py`). You can reuse the building blocks in `dedelayed/layers/splitvid_v10.py` or swap in different encoders, as long as the remote branch returns dictionaries with `{"send": {"features": {"stage2_backbone": Tensor}}}` for compatibility with the provided local decoders.

## Utilities
- **Normalization** – `dedelayed/data/normalization.renormalize` takes any tensor in `[0,1]`, `[-1,1]`, ImageNet, or CLIP space and converts it to the destination stats while respecting arbitrary channel dimensions.
- **Cityscapes palettes** – `dedelayed/datasets/cityscapes.decode_cityscapes_{tensor,pil}` maps 0–19 class IDs + ignore label (rolled to 0) into RGB tensors, useful for debugging predictions.

## License
This project is distributed under the BSD license included in `LICENSE`. The notice explicitly states that no patent rights are granted; review the file before redistributing or modifying the software.
