# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import io
import json
import os
import textwrap
from typing import Any

import numpy as np
import PIL.Image
import torch
import yaml
from omegaconf import open_dict
from torchvision.transforms.functional import to_pil_image


class Tracker:
    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        return

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        return

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        return

    def close(self) -> None:
        return


class MultiTracker(Tracker):
    def __init__(self, trackers: list[Tracker]) -> None:
        self.trackers = trackers

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.log_hyperparams(hparams)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step=step)

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        for tracker in self.trackers:
            tracker.log_image(name, image, step=step)

    def close(self) -> None:
        for tracker in self.trackers:
            tracker.close()


class ConsoleTracker(Tracker):
    def __init__(self) -> None:
        self.latest_metrics: dict[str, float] = {}
        self.last_step: int | None = None
        self.last_epoch: int | None = None

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        print("Hyperparameters:")
        print(textwrap.indent(yaml.safe_dump(hparams, sort_keys=False), "  "), end="")

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        self.last_step = step
        for key, value in metrics.items():
            scalar = _to_scalar(value)
            if key == "epoch":
                self.last_epoch = int(scalar)
                continue
            self.latest_metrics[key] = scalar

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        return

    def close(self) -> None:
        print("Run Summary:")
        summary_lines = [
            f"step: {self.last_step}",
            f"epoch: {self.last_epoch}",
            *[f"{key}: {value:.6g}" for key, value in self.latest_metrics.items()],
        ]
        print(textwrap.indent("\n".join(summary_lines), "  "))


class FileTracker(Tracker):
    def __init__(self, *, dir: str) -> None:
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        self._metrics_path = os.path.join(self.dir, "metrics.csv")

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        with open(os.path.join(self.dir, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump(hparams, f, indent=2)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        import csv

        epoch_value = metrics.get("epoch")
        if epoch_value is None:
            return
        epoch_int = None if epoch_value is None else int(_to_scalar(epoch_value))
        scalars = {k: _to_scalar(v) for k, v in metrics.items() if k != "epoch"}

        file_exists = os.path.exists(self._metrics_path)
        with open(self._metrics_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "epoch", "key", "value"])
            if not file_exists:
                writer.writeheader()
            for key, value in scalars.items():
                writer.writerow(
                    {"step": step, "epoch": epoch_int, "key": key, "value": value}
                )

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        pil = _as_pil_image(image)
        image_dir = os.path.join(self.dir, "images", name)
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"{step:012d}.png")
        pil.save(image_path, format="PNG")

    def close(self) -> None:
        return


class AimTracker(Tracker):
    def __init__(
        self,
        *,
        repo: str | None = None,
        experiment: str,
        **kwargs: Any,
    ) -> None:
        from aim import Run

        self.run = Run(repo=repo, experiment=experiment, **kwargs)

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        self.run["hparams"] = hparams

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        epoch_value = metrics.get("epoch")
        epoch_int = None if epoch_value is None else int(_to_scalar(epoch_value))
        scalars = {k: _to_scalar(v) for k, v in metrics.items()}
        for key, value in scalars.items():
            self.run.track(value, name=key, step=step, epoch=epoch_int)  # type: ignore[arg-type]

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        from aim import Image

        pil = _as_pil_image(image)
        self.run.track(Image(pil), name=name, step=step)

    def close(self) -> None:
        self.run.close()


class MLflowTracker(Tracker):
    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str,
        run_name: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        import mlflow

        self.mlflow = mlflow
        if tracking_uri is not None:
            self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)
        self.run = self.mlflow.start_run(
            run_name=run_name,
            parent_run_id=parent_run_id,
        )

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        flat = _flatten_dict(hparams, sep=".")
        _ = json.dumps(flat)
        for k, v in flat.items():
            self.mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        scalars = {k: _to_scalar(v) for k, v in metrics.items()}
        self.mlflow.log_metrics(scalars, step=step)

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        self.mlflow.log_image(_as_pil_image(image), key=name, step=step)

    def close(self) -> None:
        self.mlflow.end_run()


class NeptuneTracker(Tracker):
    def __init__(
        self,
        *,
        project: str,
        experiment_name: str | None = None,
        log_directory: str | None = None,
        **kwargs: Any,
    ) -> None:
        from neptune_scale import Run

        self.run = Run(
            project=project,
            experiment_name=experiment_name,
            log_directory=log_directory,
            **kwargs,
        )

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        self.run.log_configs({"hparams": hparams}, flatten=True, cast_unsupported=False)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        scalars = {k: _to_scalar(v) for k, v in metrics.items()}
        self.run.log_metrics(data=scalars, step=step)

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        from neptune_scale.types import File

        pil = _as_pil_image(image)
        with io.BytesIO() as buf:
            pil.save(buf, format="PNG")
            data = buf.getvalue()
        self.run.assign_files(
            {f"{name}/{step:012d}.png": File(source=data, mime_type="image/png")}
        )

    def close(self) -> None:
        self.run.close()


class PlutoTracker(Tracker):
    def __init__(
        self,
        *,
        dir: str | None = None,
        project: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        import pluto

        self.pluto = pluto
        self.run = pluto.init(
            dir=dir, project=project, name=name, config=config, **kwargs
        )

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        self.run.log({"hparams": json.dumps(hparams, sort_keys=True)}, step=0)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        scalars = {k: _to_scalar(v) for k, v in metrics.items()}
        self.run.log(scalars, step=step)

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        self.run.log({name: self.pluto.Image(_as_pil_image(image))}, step=step)

    def close(self) -> None:
        self.pluto.finish(self.run)


class TensorBoardTracker(Tracker):
    def __init__(self, *, log_dir: str) -> None:
        from torch.utils.tensorboard.writer import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        self.writer.add_text("hparams", json.dumps(hparams, indent=2, sort_keys=True))

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, _to_scalar(value), step)

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        pil = _as_pil_image(image)
        self.writer.add_image(name, np.asarray(pil), step, dataformats="HWC")

    def close(self) -> None:
        self.writer.close()


class TrackioTracker(Tracker):
    def __init__(
        self,
        *,
        project: str,
        name: str | None = None,
        group: str | None = None,
        **kwargs: Any,
    ) -> None:
        import trackio

        self.trackio = trackio
        self.run = self.trackio.init(
            project=project,
            name=name,
            group=group,
            **kwargs,
        )

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        self.run.config = hparams

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        scalars = {k: _to_scalar(v) for k, v in metrics.items()}
        self.trackio.log(scalars, step=step)

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        pil = _as_pil_image(image)
        self.trackio.log({name: self.trackio.Image(pil)}, step=step)

    def close(self) -> None:
        self.trackio.finish()


class WandbTracker(Tracker):
    def __init__(
        self,
        *,
        project: str | None = None,
        name: str | None = None,
        group: str | None = None,
        dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        import wandb

        self.wandb = wandb
        self.run = self.wandb.init(
            project=project, name=name, group=group, dir=dir, **kwargs
        )

    def log_hyperparams(self, hparams: dict[str, Any]) -> None:
        self.run.config.update(hparams, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        scalars = {k: _to_scalar(v) for k, v in metrics.items()}
        self.run.log(scalars, step=step)

    def log_image(self, name: str, image: Any, *, step: int) -> None:
        pil = _as_pil_image(image)
        self.run.log({name: self.wandb.Image(pil)}, step=step)

    def close(self) -> None:
        self.run.finish()


def _build_one_tracker(
    backend_cfg: dict,
    *,
    project: str,
    experiment: str,
    run_id: str,
    run_name: str | None,
    log_dir: str,
    hparams: dict[str, Any],
) -> Tracker:
    name = backend_cfg["name"]
    kwargs = backend_cfg.get("kwargs", {})

    if name == "file":
        tracker = FileTracker(
            dir=kwargs.get(
                "dir",
                os.path.join(log_dir, "file", project, experiment, run_id),
            )
        )
        tracker.log_hyperparams(hparams)
        return tracker

    if name == "console":
        tracker = ConsoleTracker()
        tracker.log_hyperparams(hparams)
        return tracker

    if name == "aim":
        tracker = AimTracker(
            **_with_shared_kwargs(
                kwargs,
                repo=os.path.join(log_dir, "aim", project),
                experiment=experiment,
            )
        )
        with open_dict(kwargs):
            kwargs["run_hash"] = tracker.run.hash
        tracker.log_hyperparams(hparams)
        return tracker

    if name == "mlflow":
        tracker = MLflowTracker(
            tracking_uri=kwargs.get(
                "tracking_uri", os.environ.get("MLFLOW_TRACKING_URI")
            ),
            experiment_name=kwargs.get("experiment_name", experiment),
            run_name=kwargs.get("run_name", run_name),
            parent_run_id=kwargs.get("parent_run_id"),
        )
        with open_dict(kwargs):
            kwargs["parent_run_id"] = (
                kwargs.get("parent_run_id") or tracker.run.info.run_id
            )
        tracker.log_hyperparams(hparams)
        return tracker

    if name == "neptune":
        tracker = NeptuneTracker(
            **_with_shared_kwargs(
                kwargs,
                project=kwargs.get("project") or f"{kwargs['workspace']}/{project}",
                experiment_name=experiment,
                log_directory=os.path.join(log_dir, "neptune"),
            )
        )
        tracker.log_hyperparams(hparams)
        return tracker

    if name == "pluto":
        tracker = PlutoTracker(
            **_with_shared_kwargs(
                kwargs,
                dir=os.path.join(log_dir, "pluto"),
                project=project,
                name=run_name,
            )
        )
        tracker.log_hyperparams(hparams)
        return tracker

    if name == "tensorboard":
        tracker = TensorBoardTracker(
            log_dir=kwargs.get(
                "log_dir",
                os.path.join(log_dir, "tensorboard", project, experiment, run_id),
            )
        )
        tracker.log_hyperparams(hparams)
        return tracker

    if name == "trackio":
        tracker = TrackioTracker(
            **_with_shared_kwargs(
                kwargs,
                project=project,
                name=run_name,
                group=experiment,
            )
        )
        with open_dict(kwargs):
            kwargs["name"] = tracker.run.name
            kwargs["resume"] = "must"
        tracker.log_hyperparams(hparams)
        return tracker

    if name == "wandb":
        tracker = WandbTracker(
            **_with_shared_kwargs(
                kwargs,
                project=project,
                name=run_name,
                group=experiment,
                dir=os.path.join(log_dir, "wandb"),
            )
        )
        with open_dict(kwargs):
            kwargs["id"] = tracker.run.id
            kwargs["resume"] = "must"
        tracker.log_hyperparams(hparams)
        return tracker

    raise ValueError(f"Unsupported tracker: {name}")


def build_tracker(
    tracker_cfg: dict, *, run_id: str, hparams: dict[str, Any]
) -> Tracker:
    return MultiTracker(
        [
            _build_one_tracker(
                backend_cfg,
                project=tracker_cfg["project"],
                experiment=tracker_cfg["experiment"],
                run_id=run_id,
                run_name=tracker_cfg.get("run_name"),
                log_dir=tracker_cfg["log_dir"],
                hparams=hparams,
            )
            for backend_cfg in tracker_cfg["backends"]
        ]
    )


def _with_shared_kwargs(
    kwargs: dict[str, Any], /, **shared_defaults: Any
) -> dict[str, Any]:
    return {**shared_defaults, **kwargs}


def _to_scalar(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.generic):
        return float(value.item())
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().cpu().item())
    raise TypeError(f"Expected scalar metric, got {type(value)}")


def _as_pil_image(x: Any) -> PIL.Image.Image:
    return x if isinstance(x, PIL.Image.Image) else to_pil_image(x)


def _flatten_dict(
    d: dict[str, Any], *, prefix: str = "", sep: str = "."
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out
