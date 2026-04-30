# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import argparse
import copy
import time
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from fractions import Fraction
from itertools import chain, islice
from pathlib import Path

import av
import av.container
import av.video.stream
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import ImageDraw, ImageFont
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image

import dedelayed.datasets.cityscapes
from dedelayed.datasets.cityscapes import decode_cityscapes_tensor
from dedelayed.models.dedelayed_v1.base import Dedelayed_v1_Fused, Dedelayed_v1_Remote
from dedelayed.models.dedelayed_v1.factory import build_fused_model
from dedelayed.zoo import get_model

device = "cuda"


@torch.inference_mode()
def preprocess_frame(frame_rgb: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    assert frame_rgb.ndim == 3 and frame_rgb.shape[0] == 3
    x = frame_rgb
    x = x.to(device).float()
    x = x.unsqueeze(0)
    x = F.interpolate(
        x, size=size, mode="bicubic", align_corners=False, antialias=True
    ).clip(0, 255)
    x = x / 255.0
    x = x.squeeze(0)
    return x  # (3, H, W)


def resize_seg_logits(
    seg_logits: torch.Tensor, target_size: Sequence[int]
) -> torch.Tensor:
    return TF.resize(
        seg_logits, size=list(target_size), interpolation=TF.InterpolationMode.BICUBIC
    ).squeeze(0)


@torch.inference_mode()
def draw_frame(
    orig_frame: torch.Tensor,
    seg_logits: torch.Tensor,
    *,
    confidence_threshold=0.0,
    mask_strength=0.65,
    ignore_index=255,
    weight_by_confidence=True,
) -> torch.Tensor:
    assert orig_frame.ndim == 3 and orig_frame.shape[0] == 3
    assert seg_logits.ndim == 3 and seg_logits.shape[0] == 19
    probs = seg_logits.softmax(dim=-3)
    probs_max, pred = probs.max(dim=-3)
    pred[probs_max < confidence_threshold] = ignore_index
    color_mask = decode_cityscapes_tensor(pred.byte())
    weight = mask_strength * (
        probs_max if weight_by_confidence else torch.ones_like(probs_max)
    )
    blended_frame = (
        torch.lerp(
            orig_frame.cpu().float(),
            color_mask.cpu().float(),
            weight=weight.cpu(),
        )
        .clip(0, 255)
        .round()
        .byte()
    )
    return blended_frame


def load_font(
    font_name: str, size: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (font_name, "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


@torch.inference_mode()
def draw_comparison_frame(
    orig_frame: torch.Tensor,
    seg_logits: dict[str, torch.Tensor],
    target_size: Sequence[int],
    *,
    past_ticks_true: int,
    fps_fraction: Fraction,
    **kwargs,
) -> PIL.Image.Image:
    font_name = "Cantarell-Regular"
    scale = 1.2
    font = load_font(font_name, size=int(64 * scale))
    font_small = load_font(font_name, size=int(48 * scale))

    delay_ms = round(1000 * past_ticks_true / float(fps_fraction))
    info_text = f"{delay_ms} ms round-trip delay  |  {round(float(fps_fraction))} fps"
    column_labels = [
        "Remote (conventional)",
        "Remote (predictive)",
        "Local",
        "DeDelayed",
    ]

    pad_color = 255
    info_color = (30, 120, 180)
    column_color = (0, 0, 0)

    frame = torch.cat(
        [
            draw_frame(orig_frame, seg_logits["remote"], **kwargs),
            draw_frame(orig_frame, seg_logits["remote_predictive"], **kwargs),
            draw_frame(orig_frame, seg_logits["local"], **kwargs),
            draw_frame(orig_frame, seg_logits["local_fused"], **kwargs),
        ],
        dim=-1,
    )

    frame = torch.nn.functional.pad(
        frame,
        pad=(0, 0, 200, 200),
        mode="constant",
        value=pad_color,
    )

    img = to_pil_image(frame)
    draw = ImageDraw.Draw(img)

    x = 4 * target_size[1] - 60
    y = 90
    draw.text(
        (x, y),
        info_text,
        font=font_small,
        fill=info_color,
        align="right",
        anchor="rt",
    )

    for i, column_text in enumerate(column_labels):
        x = i * target_size[1] + target_size[1] // 2
        y = 200 + target_size[0] + 120
        draw.text(
            (x, y),
            column_text,
            font=font,
            fill=column_color,
            align="center",
            anchor="ms",
        )

    return img


def read_video_frames(filename: str) -> Iterator[torch.Tensor]:
    container = av.open(filename)
    for frame in container.decode(video=0):
        pil_img = frame.to_image().convert("RGB")
        assert isinstance(pil_img, PIL.Image.Image)
        pil_img = pil_img.rotate(frame.rotation, expand=True)
        frame_rgb = pil_to_tensor(pil_img)
        yield frame_rgb


def open_video_writer(
    output_filename: str, *, output_size: tuple[int, int], fps_fraction: Fraction
) -> tuple[av.container.OutputContainer, av.video.stream.VideoStream]:
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    container = av.open(output_filename, mode="w")
    stream = container.add_stream(
        "libx264",
        rate=fps_fraction,
        options={
            "preset": "medium",
            "crf": "28",
            "x264-params": "log-level=error",
            "x265-params": "log-level=error",
            "keyint": "30",
        },
    )
    stream.width = output_size[1]
    stream.height = output_size[0]
    stream.pix_fmt = "yuv420p"
    return container, stream


def init_model(args):
    if args.model_checkpoint is not None:
        ckpt = torch.load(args.model_checkpoint, map_location="cpu")
        model_cfg = ckpt["meta"]["hp"]["model"]
        model = build_fused_model(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model_name = model_cfg["name"]
    else:
        model = get_model(args.model_name, pretrained=True)
        model_name = args.model_name

    model.localonly_model = model.local_model
    if args.localonly_model_checkpoint is not None:
        localonly_ckpt = torch.load(args.localonly_model_checkpoint, map_location="cpu")
        model.localonly_model = copy.deepcopy(model.local_model)
        model.localonly_model.image_model.load_state_dict(
            localonly_ckpt["model_state_dict"], strict=True
        )

    model.to(device)
    assert isinstance(model, Dedelayed_v1_Fused)

    if args.compile_enabled:
        compile_kwargs: dict = {"mode": "max-autotune"}
        model.compile(**compile_kwargs)
        model.remote_model.compile(**compile_kwargs)
        model.remote_model.encode_frames = torch.compile(
            model.remote_model.encode_frames, **compile_kwargs
        )
        model.local_model.compile(**compile_kwargs)
        model.localonly_model.compile(**compile_kwargs)

    return model, model_name


class RemoteStream:
    def __init__(
        self, remote_model: Dedelayed_v1_Remote, *, x_remote_size: tuple[int, int]
    ):
        self.remote_model = remote_model
        self.x_remote_size = x_remote_size

    @torch.inference_mode()
    def init(self):
        x_remote = torch.zeros(1, 3, *self.x_remote_size, device=device)
        self.z_propagated = self.remote_model.init_stream_state(x_remote)
        self.z_blended = self.remote_model.blend(self.z_propagated)

    @torch.inference_mode()
    def encode_step(self, frame_rgb: torch.Tensor) -> None:
        x_remote_latest = preprocess_frame(
            frame_rgb, size=self.x_remote_size
        ).unsqueeze(0)
        self.z_blended, self.z_propagated = self.remote_model.encode_step(
            x_remote_latest,
            self.z_propagated,
        )

    @torch.inference_mode()
    def readout(
        self,
        *,
        past_ticks: float,
        x_local_size: tuple[int, int],
        output_keys: Sequence[str] = ("downlink_features",),
    ) -> dict[str, torch.Tensor]:
        z_prealigned = self.remote_model.prealign(
            self.z_blended,
            torch.tensor([float(past_ticks)], device=device),
        )
        out = self.remote_model.head(
            z_prealigned,
            x_local_size=x_local_size,
            output_keys=output_keys,
        )
        return copy.deepcopy(out)


@torch.inference_mode()
def iter_streaming_outputs(
    model: Dedelayed_v1_Fused,
    frames_rgb: Iterable[torch.Tensor],
    *,
    past_ticks_true: int,
    past_ticks: int,
    x_remote_size: tuple[int, int],
    x_local_size: tuple[int, int],
) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    downlink = deque()
    recv = {}
    recv_predictive = {}
    remote_stream = RemoteStream(model.remote_model, x_remote_size=x_remote_size)
    remote_stream.init()
    seg_logits_blank: torch.Tensor | None = None
    downlink_features_shape = model.local_model.downlink_features_shape(x_local_size)
    downlink_features_zeros = torch.zeros((1, *downlink_features_shape), device=device)
    model.eval()
    start_time = time.perf_counter()

    for i, frame_rgb in enumerate(frames_rgb):
        remote_stream.encode_step(frame_rgb)
        out_remote = remote_stream.readout(
            past_ticks=0,
            x_local_size=x_local_size,
            output_keys=("downlink_features", "downlink_seg_logits"),
        )
        out_remote_predictive = remote_stream.readout(
            past_ticks=past_ticks,
            x_local_size=x_local_size,
            output_keys=("downlink_features", "downlink_seg_logits"),
        )
        downlink.append((out_remote, out_remote_predictive))

        if len(downlink) > past_ticks_true:
            recv, recv_predictive = downlink.popleft()

        x_local = preprocess_frame(frame_rgb, size=x_local_size).unsqueeze(0)
        downlink_features = recv_predictive.get(
            "downlink_features", downlink_features_zeros
        )

        out_local = copy.deepcopy(
            model.localonly_model(
                x_local,
            )
        )
        out_local_fused = copy.deepcopy(
            model.local_model(
                x_local,
                downlink_features=downlink_features,
            )
        )
        if seg_logits_blank is None:
            seg_logits_blank = torch.zeros_like(out_local["seg_logits"])

        seg_logits = {
            "remote": recv.get("downlink_seg_logits", seg_logits_blank),
            "remote_predictive": recv_predictive.get(
                "downlink_seg_logits", seg_logits_blank
            ),
            "local": out_local["seg_logits"],
            "local_fused": out_local_fused["seg_logits"],
        }

        elapsed = time.perf_counter() - start_time
        print(f"frame={i} fps={(i + 1) / elapsed:.2f}")
        yield frame_rgb, seg_logits


def parse_args(argv=None):
    arguments = [
        {"name": ["input_filename"]},
        {"name": ["--output_filename"]},
        {
            "name": ["--model_name"],
            "default": "dedelayed_v1_efficientvitl1_mstransformer2d_bdd100k",
        },
        {"name": ["--model_checkpoint"]},
        {"name": ["--localonly_model_checkpoint"]},
        {"name": ["--speedup"], "type": int, "default": 1},
        {"name": ["--past_ticks"], "type": int, "default": 5},
        {"name": ["--past_ticks_offset"], "type": int, "default": 0},
        {"name": ["--x_remote_size"], "type": int, "nargs": 2, "default": (704, 1248)},
        {"name": ["--x_local_size"], "type": int, "nargs": 2, "default": (480, 832)},
        {
            "name": ["--fps"],
            "dest": "fps_fraction",
            "type": Fraction,
            "default": Fraction(2997, 100),
        },
        {
            "name": ["--compile"],
            "dest": "compile_enabled",
            "action": argparse.BooleanOptionalAction,
            "default": True,
        },
    ]

    parser = argparse.ArgumentParser()
    for argument in arguments:
        names = argument["name"]
        hyphen_names = [
            name.replace("_", "-") for name in names if name.startswith("--")
        ]
        parser.add_argument(
            *names,
            *[name for name in hyphen_names if name not in names],
            **{k: v for k, v in argument.items() if k != "name"},
        )

    args = parser.parse_args(argv)
    args.input_filename = Path(args.input_filename).expanduser()
    if args.output_filename is not None:
        args.output_filename = Path(args.output_filename).expanduser()
    if args.model_checkpoint is not None:
        args.model_checkpoint = Path(args.model_checkpoint).expanduser()
    if args.localonly_model_checkpoint is not None:
        args.localonly_model_checkpoint = Path(
            args.localonly_model_checkpoint
        ).expanduser()
    args.past_ticks_true = args.past_ticks + args.past_ticks_offset
    args.x_remote_size = tuple(args.x_remote_size)
    args.x_local_size = tuple(args.x_local_size)
    return args


def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    args = parse_args()
    model, model_name = init_model(args)

    frames_rgb = read_video_frames(args.input_filename)
    frame_rgb_first = next(frames_rgb)
    frames_rgb = chain([frame_rgb_first], frames_rgb)
    target_size = frame_rgb_first.shape[-2:]  # (H, W)

    # Visual tweaks.
    dedelayed.datasets.cityscapes.COLORS_TENSOR = torch.tensor(
        dedelayed.datasets.cityscapes.COLORS, dtype=torch.int16
    )
    dedelayed.datasets.cityscapes.COLORS_TENSOR[12] = 2 * torch.tensor(
        [220, 20, 60], dtype=torch.int16
    )  # Boost person visibility.

    output_filename = args.output_filename
    if output_filename is None:
        output_filename = (
            f"samples/output/"
            f"{args.input_filename.stem}.s{args.speedup}x"
            f"_past{round(args.past_ticks * 1000 / 30)}ms"
            f"_pastoffset{round(args.past_ticks_offset * 1000 / 30)}ms"
            f".{model_name}.r{args.x_local_size[0]}_720.mkv"
        )

    container, stream = open_video_writer(
        output_filename=output_filename,
        output_size=(target_size[0] + 2 * 200, target_size[1] * 4),
        fps_fraction=args.fps_fraction,
    )

    for frame_rgb, seg_logits in iter_streaming_outputs(
        model,
        islice(frames_rgb, 0, None, args.speedup),
        past_ticks_true=args.past_ticks_true,
        past_ticks=args.past_ticks,
        x_remote_size=args.x_remote_size,
        x_local_size=args.x_local_size,
    ):
        seg_logits = {
            k: resize_seg_logits(v, target_size) for k, v in seg_logits.items()
        }
        # Adjust for visibility.
        frame_rgb = (255 * ((frame_rgb / 255) ** 2.0 + 0.2)).round().clip(0, 255)
        img = draw_comparison_frame(
            frame_rgb,
            seg_logits,
            target_size,
            past_ticks_true=args.past_ticks_true,
            fps_fraction=args.fps_fraction,
        )
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


if __name__ == "__main__":
    main()
