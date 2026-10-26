"""Held-out predictions and ID-aligned comparisons of complete training runs.

These are validation estimates after checkpoint selection on the same folds,
not an independent test estimate. No test images are read by this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

OOF_COLUMNS = ["image_filename", "holding_label", "fold_index", "holding_probability"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_ids(split: Mapping) -> tuple[set[str], set[str]]:
    groups = []
    for key in ("training_image_filenames", "validation_image_filenames"):
        names = split.get(key)
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"OOF requires distinct nonempty {key}")
        groups.append(set(names))
    if groups[0] & groups[1]:
        raise ValueError("OOF training and validation image IDs must be disjoint")
    return groups[0], groups[1]


def save_fold_oof(model, validation_loader, checkpoint_path, *, split, device) -> Path:
    """Reload best.pt, predict only this fold's held-out IDs, then persist privately."""
    import torch

    from .engine import _check_logits
    from .prediction import _check_probability

    _, validation_ids = _split_ids(split)
    expected_targets = split.get("validation_targets")
    if expected_targets is not None and set(expected_targets) != validation_ids:
        raise ValueError("OOF validation_targets must match the held-out image IDs")
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    metadata = checkpoint["metadata"]
    model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
    del checkpoint
    model.to(device)
    model.eval()
    rows = {}
    with torch.inference_mode():
        for batch in validation_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            filenames, labels = batch["image_filename"], batch["holding_label"]
            if (
                isinstance(filenames, str)
                or len(filenames) != len(images)
                or labels.ndim != 1
                or len(labels) != len(images)
                or not ((labels == 0) | (labels == 1)).all()
            ):
                raise ValueError("OOF requires one image ID and binary target per image")
            if any(name not in validation_ids or name in rows for name in filenames):
                raise ValueError("OOF loader contains a duplicate or non-held-out image ID")
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                _check_logits(logits, len(images))
            probabilities = logits.float().softmax(dim=1)[:, 1].cpu().tolist()
            for name, target, probability in zip(filenames, labels.cpu().tolist(), probabilities):
                if name in rows:
                    raise ValueError("OOF loader contains duplicate image IDs")
                if expected_targets is not None and target != expected_targets[name]:
                    raise ValueError("OOF loader target disagrees with the saved split")
                rows[name] = (name, target, metadata["fold"], _check_probability(probability, name))
    if set(rows) != validation_ids:
        raise ValueError("OOF loader must contain every held-out image exactly once")
    path = checkpoint_path.parent / "oof.csv"
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(OOF_COLUMNS)
        writer.writerows(rows[name] for name in split["validation_image_filenames"])
    (path.parent / "oof.json").write_text(
        json.dumps(
            {
                "format": "fan-classifier-oof-v1",
                "model_name": metadata["model_name"],
                "fold": metadata["fold"],
                "best_epoch": metadata["best_epoch"],
                "training_csv_sha256": metadata["training_csv_sha256"],
                "config": metadata["config"],
                "sample_count": len(rows),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "oof_sha256": file_sha256(path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _load_run_oof(run_directory: str | Path) -> tuple[dict, dict[str, dict], dict]:
    """Require a complete partition, completed folds, and matching artifact provenance."""
    from .prediction import _check_probability

    run_directory = Path(run_directory)
    manifest = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "fan-classifier-run-v1":
        raise ValueError("Unsupported run.json format")
    config = manifest["config"]
    fold_count = config["num_folds"]
    if type(fold_count) is not int or fold_count < 2:
        raise ValueError("OOF requires at least two folds")
    all_rows, population, fold_artifacts = {}, None, {}
    for fold in range(fold_count):
        directory = run_directory / f"fold_{fold}"
        training = json.loads((directory / "training.json").read_text(encoding="utf-8"))
        sidecar = json.loads((directory / "oof.json").read_text(encoding="utf-8"))
        split_metadata = json.loads((directory / "fold_manifest.json").read_text(encoding="utf-8"))
        split = split_metadata["split"]
        if (
            training.get("status") != "complete"
            or sidecar.get("format") != "fan-classifier-oof-v1"
            or split_metadata.get("fold") != fold
            or split_metadata.get("num_folds") != fold_count
            or split_metadata.get("seed") != config["seed"]
            or sidecar.get("best_epoch") != training.get("best_epoch")
        ):
            raise ValueError("OOF requires completed folds with matching best-checkpoint metadata")
        for source in (training, sidecar):
            if (
                source.get("fold") != fold
                or source.get("model_name") != config["model_name"]
                or source.get("training_csv_sha256") != manifest["training_csv_sha256"]
                or source.get("config") != config
            ):
                raise ValueError("OOF artifact metadata does not match run.json")
        train_ids, valid_ids = _split_ids(split)
        current_population = train_ids | valid_ids
        if population is not None and current_population != population:
            raise ValueError("OOF fold manifests refer to different image populations")
        population = current_population
        path = directory / "oof.csv"
        if file_sha256(path) != sidecar.get("oof_sha256"):
            raise ValueError("OOF CSV content hash does not match its metadata")
        checkpoint_path = directory / "best.pt"
        if file_sha256(checkpoint_path) != sidecar.get("checkpoint_sha256"):
            raise ValueError("OOF checkpoint content hash does not match its metadata")
        fold_artifacts[str(fold)] = {
            "best_epoch": sidecar["best_epoch"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sidecar["checkpoint_sha256"],
            "oof_path": str(path),
            "oof_sha256": sidecar["oof_sha256"],
        }
        fold_rows = {}
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != OOF_COLUMNS:
                raise ValueError(f"Expected OOF columns {OOF_COLUMNS}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Malformed OOF row")
                name = row["image_filename"]
                if name not in valid_ids or name in fold_rows or name in all_rows:
                    raise ValueError("OOF IDs must be unique and belong to their held-out fold")
                if row["holding_label"] not in {"0", "1"} or row["fold_index"] != str(fold):
                    raise ValueError("OOF target or fold is invalid")
                target = int(row["holding_label"])
                if split.get("validation_targets", {}).get(name, target) != target:
                    raise ValueError("OOF target disagrees with the saved split")
                fold_rows[name] = {
                    "holding_label": target,
                    "fold_index": fold,
                    "holding_probability": _check_probability(row["holding_probability"], name),
                }
        if set(fold_rows) != valid_ids or len(fold_rows) != sidecar.get("sample_count"):
            raise ValueError("OOF must cover every validation image exactly once")
        all_rows.update(fold_rows)
    if set(all_rows) != population or len(all_rows) != manifest.get("sample_count"):
        raise ValueError("OOF must cover the complete training population exactly once")
    return manifest, all_rows, fold_artifacts


def classification_metrics(rows: Mapping[str, Mapping]) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate empty OOF predictions")
    tp = fp = tn = fn = 0
    losses = []
    for row in rows.values():
        target, probability = row["holding_label"], row["holding_probability"]
        prediction = int(probability > 0.5)
        tp += target == prediction == 1
        tn += target == prediction == 0
        fp += target == 0 and prediction == 1
        fn += target == 1 and prediction == 0
        clipped = min(1 - 1e-15, max(1e-15, probability))
        losses.append(-math.log(clipped if target else 1 - clipped))
    return {
        "sample_count": len(rows),
        "accuracy": (tp + tn) / len(rows),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "log_loss": math.fsum(losses) / len(rows),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def evaluate_oof_runs(run_directories: Sequence[str | Path], output_directory: str | Path) -> Path:
    """Compare complete held-out runs and their equal-weight probability ensemble.

    Different folds are permitted, but explicitly reported. These comparisons
    alone cannot attribute improvements to one changed hyperparameter.
    Every original best.pt must remain present and match its recorded hash.
    Checkpoints are streamed for hashing, never deserialized for evaluation.
    """
    paths = [Path(path).resolve() for path in run_directories]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide distinct run directories")
    if Path(output_directory).exists():
        raise FileExistsError(f"Output directory already exists: {output_directory}")
    loaded = [_load_run_oof(path) for path in paths]
    reference = loaded[0][1]
    for _, rows, _ in loaded[1:]:
        if set(rows) != set(reference):
            raise ValueError("OOF comparison requires exactly matching image IDs")
        if any(rows[name]["holding_label"] != reference[name]["holding_label"] for name in rows):
            raise ValueError("OOF comparison requires matching targets")
    report = {
        "format": "fan-classifier-oof-evaluation-v1",
        "threshold": 0.5,
        "threshold_operator": ">",
        "evaluation_scope": "held-out validation after checkpoint selection; not independent test",
        "same_fold_assignments": all(
            all(rows[name]["fold_index"] == reference[name]["fold_index"] for name in rows)
            for _, rows, _ in loaded
        ),
        "members": [],
    }
    error_rows = []
    for index, (path, (manifest, rows, fold_artifacts)) in enumerate(zip(paths, loaded)):
        member = {
            "run_directory": str(path),
            "config": manifest["config"],
            "training_csv_sha256": manifest["training_csv_sha256"],
            "fold_artifacts": fold_artifacts,
            "metrics": classification_metrics(rows),
            "fold_metrics": {
                str(fold): classification_metrics(
                    {name: row for name, row in rows.items() if row["fold_index"] == fold}
                )
                for fold in range(manifest["config"]["num_folds"])
            },
        }
        report["members"].append(member)
        for name, row in rows.items():
            if int(row["holding_probability"] > 0.5) != row["holding_label"]:
                error_rows.append(
                    [
                        str(index),
                        name,
                        row["holding_label"],
                        row["fold_index"],
                        row["holding_probability"],
                    ]
                )
    ensemble_rows = {
        name: {
            "holding_label": reference[name]["holding_label"],
            "holding_probability": math.fsum(
                rows[name]["holding_probability"] for _, rows, _ in loaded
            )
            / len(loaded),
        }
        for name in reference
    }
    if len(loaded) > 1:
        report["ensemble"] = {
            "aggregation": "equal_weight_per_run_held_out_probability_mean",
            "metrics": classification_metrics(ensemble_rows),
        }
        for name, row in ensemble_rows.items():
            if int(row["holding_probability"] > 0.5) != row["holding_label"]:
                error_rows.append(
                    ["ensemble", name, row["holding_label"], "", row["holding_probability"]]
                )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    report_path = output_directory / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_directory / "errors.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["member_index", *OOF_COLUMNS])
        writer.writerows(error_rows)
    with (output_directory / "probabilities.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_filename", "holding_label", "holding_probability"])
        writer.writerows(
            (name, row["holding_label"], row["holding_probability"])
            for name, row in ensemble_rows.items()
        )
    return report_path
