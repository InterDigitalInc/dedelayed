import PIL.Image
import torch
from torch import Tensor
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image

# WARN: In our dataset, the labels are actually stored with an offset of +1,
# i.e., {"road": 1, "sidewalk": 2, ..., "bicycle": 19, "unlabeled": 255}.
LABELS = [
    "unlabeled",
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

COLORS = [
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

COLORS_TENSOR = torch.tensor(COLORS, dtype=torch.uint8)

PALETTE = {label: color for label, color in zip(LABELS, COLORS)}


def decode_cityscapes_tensor(mask: Tensor) -> Tensor:
    assert mask.dtype == torch.uint8
    assert mask.ndim == 2
    # NOTE: The + 1 rolls the ignore_index label (255) to 0.
    y = mask + 1
    return COLORS_TENSOR.to(y.device)[y.long()].permute(2, 0, 1)


def decode_cityscapes_pil(img_mask: PIL.Image.Image) -> PIL.Image.Image:
    mask = pil_to_tensor(img_mask).squeeze(0)
    return to_pil_image(decode_cityscapes_tensor(mask))
