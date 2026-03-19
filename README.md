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




## Quick Start (Inference)

```python
import torch
from dedelayed.registry import MODELS

# Choose a model:
# model_name = "dedelayed_v1_efficientvitl1_mstransformer2d"  # used in CVPR 2026 paper
# model_name = "dedelayed_v1_efficientvitl1_efficientvitb0"  # finetuned on pre-trained MIT checkpoints

remote_model = MODELS[f"{model_name}_remote"]()
local_model = MODELS[f"{model_name}_local"]()
model = MODELS[f"{model_name}"](
    remote_model=remote_model,
    local_model=local_model,
)

x_remote = torch.rand(1, 3, 4, 720, 1248)
x_local = torch.rand(1, 3, 480, 832)
out = model(x_local, x_remote, past_ticks=5)
segmentation = out["seg_logits"].argmax(dim=1)
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

