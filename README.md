# Dedelayed


## Works

### [CVPR 2026] Dedelayed: Deleting Remote Inference Delay via On-Device Correction [[paper](https://arxiv.org/pdf/2510.13714)] [[archival code snapshot](https://github.com/InterDigitalInc/dedelayed/tree/papers/2026-cvpr-dedelayed)]




## Installation

```bash
git clone https://github.com/InterDigitalInc/dedelayed.git
cd dedelayed

# Using uv:
uv sync

# Using pip:
python -m venv .venv
source .venv/bin/activate
pip install -e .
```




## Quick Start (Demo)

```python
uv run python dedelayed/apps/dedelayed_v1/scripts/demo.py samples/input/source_video.mp4 --model_name=dedelayed_v1_efficientvitl1_mstransformer2d_bdd100k
```




## Quick Start (Inference)

```python
import torch
from dedelayed.zoo import get_model

device = "cuda"
model_name = "dedelayed_v1_efficientvitl1_mstransformer2d_bdd100k"  # CVPR 2026
model = get_model(model_name, pretrained=True).eval().to(device=device)

x_remote = torch.rand((1, 3, 4, 720, 1248), device=device)
x_local = torch.rand((1, 3, 480, 832), device=device)
past_ticks = torch.full((1,), 5.0, device=device)

with torch.inference_mode():
    out_remote = model.remote_model(
        x_remote,
        past_ticks=past_ticks,
        x_local_size=x_local.shape[-2:],
    )
    out_local = model.local_model(
        x_local,
        downlink_features=out_remote["downlink_features"],
    )
    pred_mask = out_local["seg_logits"].argmax(dim=1)
```




## Training

Configure datasets:

Set `meta["hp"]["dataset"]` to dataset path.

Then, run training:

```bash
python scripts/train.py
```




## Dataset format

The reference dataloader assumes each Hugging Face sample looks roughly like this:

For frames `i` in `[0, 15]`:

- `original_{i}` – uncompressed RGB images for evaluation.
- `near_lossless_{i}` – current-frame RGB images.
- `label_{i}` / `label_hq_{i}` – semantic masks stored as single-channel PIL images using Cityscapes IDs (ignore index 255).




## License

This project is distributed under the BSD license included in `LICENSE`. The notice explicitly states that no patent rights are granted; review the file before redistributing or modifying the software.




## Citation

```bibtex
@inproceedings{jacobellis2026dedelayed,
  title     = {Dedelayed: Deleting Remote Inference Delay via On-Device Correction},
  author    = {Jacobellis, Dan and Ulhaq, Mateen and Racap{\'e}, Fabien and Choi, Hyomin and Yadwadkar, Neeraja J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  note      = {To appear}
}
```
