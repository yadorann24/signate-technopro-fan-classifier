"""Explicit command-line entry points for private training and prediction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .config import resolve_training_config
from .registry import build_model, get_model_spec, list_model_specs


def _fold_indices(requested: list[int] | None, count: int) -> list[int]:
    folds = list(range(count)) if requested is None else requested
    if not folds or len(set(folds)) != len(folds) or any(f < 0 or f >= count for f in folds):
        raise ValueError(f"--folds must contain distinct indices in 0..{count - 1}")
    return sorted(folds)


def _seed_worker(worker_id: int) -> None:
    """Seed both library RNGs and Albumentations' independent worker RNG."""
    import random
    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    worker = torch.utils.data.get_worker_info()
    transform = getattr(worker.dataset, "transform", None) if worker else None
    if transform is not None:
        transform.set_random_seed(seed)


def _data_loader(records, transform, *, batch_size, num_workers, training, seed, device):
    import torch
    from torch.utils.data import DataLoader
    from .data import FanHoldingDataset

    return DataLoader(
        FanHoldingDataset(records, transform),
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=_seed_worker,
    )


def _environment() -> dict:
    packages = {}
    for package in (
        "torch",
        "torchvision",
        "timm",
        "albumentations",
        "numpy",
        "opencv-python-headless",
        "scikit-learn",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _training(args) -> None:
    config = resolve_training_config(
        args.model,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        num_folds=args.num_folds,
        pretrained=args.pretrained,
        normalization={"mode": args.normalization} if args.normalization else None,
    )
    from .data import read_training_records, split_records
    from .engine import fit_model, resolve_device, seed_everything
    from .transforms import build_transforms

    folds = _fold_indices(args.folds, config.num_folds)
    selected_device = resolve_device(args.device)
    records = read_training_records(args.train_csv, args.image_root)
    # Validate all requested splits and output paths before allocating a model.
    splits = {
        fold: split_records(records, fold_count=config.num_folds, fold_index=fold, seed=config.seed)
        for fold in folds
    }
    output_directory = Path(args.output_dir)
    manifest_path = output_directory / "run.json"
    run_identity = {
        "config": json.loads(json.dumps(config.to_dict())),
        "training_csv_sha256": hashlib.sha256(Path(args.train_csv).read_bytes()).hexdigest(),
        "image_root": str(Path(args.image_root).resolve()),
        "sample_count": len(records),
    }
    if output_directory.exists():
        if not manifest_path.is_file():
            raise ValueError("Existing output directory has no run.json; choose a new --output-dir")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("format") != "fan-classifier-run-v1":
            raise ValueError("Unsupported run.json format")
        if any(previous.get(key) != value for key, value in run_identity.items()):
            raise ValueError(
                "Existing run uses different settings or input data; choose a new --output-dir"
            )
        if previous.get("environment") != _environment():
            raise ValueError("Existing run uses a different environment; choose a new --output-dir")
    for fold in folds:
        if (output_directory / f"fold_{fold}").exists():
            raise FileExistsError(
                f"fold_{fold} already exists; choose a new run or untrained --folds"
            )
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=False)
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "format": "fan-classifier-run-v1",
                    **run_identity,
                    "environment": _environment(),
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")
    print(
        f"model={config.model_name} image_size={config.image_size} folds={folds} device={selected_device}",
        flush=True,
    )
    for fold in folds:
        train_records, validation_records = splits[fold]
        seed_everything(config.seed)
        model = build_model(config.model_name, pretrained=config.pretrained)
        training_transform = build_transforms(
            config.image_size,
            training=True,
            seed=config.seed,
            mean=config.normalization.mean,
            std=config.normalization.std,
            **asdict(config.augmentation),
        )
        validation_transform = build_transforms(
            config.image_size,
            training=False,
            mean=config.normalization.mean,
            std=config.normalization.std,
        )
        loader_settings = dict(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            device=selected_device,
        )
        train_loader = _data_loader(
            train_records, training_transform, training=True, **loader_settings
        )
        validation_loader = _data_loader(
            validation_records, validation_transform, training=False, **loader_settings
        )
        result = fit_model(
            model,
            train_loader,
            validation_loader,
            model_name=config.model_name,
            image_size=config.image_size,
            fold=fold,
            output_directory=output_directory / f"fold_{fold}",
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            scheduler_t_max=config.epochs,
            scheduler_eta_min=config.min_learning_rate,
            selection_metric=config.selection_metric,
            device=str(selected_device),
            seed=config.seed,
            num_folds=config.num_folds,
            config=config.to_dict(),
            training_csv_sha256=run_identity["training_csv_sha256"],
            fold_manifest={
                "training_image_filenames": [r.image_filename for r in train_records],
                "validation_image_filenames": [r.image_filename for r in validation_records],
                "validation_targets": {
                    r.image_filename: r.holding_label for r in validation_records
                },
            },
        )
        print(f"Saved {result.checkpoint_path}", flush=True)
        del model, train_loader, validation_loader
        if selected_device.type == "cuda":
            import torch

            torch.cuda.empty_cache()


def _prediction(args) -> None:
    from .data import read_inference_records
    from .engine import resolve_device
    from .prediction import predict_checkpoints
    from .transforms import build_transforms

    spec = get_model_spec(args.model)
    run_directory = Path(args.run_dir)
    manifest = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "fan-classifier-run-v1":
        raise ValueError("Unsupported run.json format")
    config = manifest.get("config")
    if not isinstance(config, dict) or get_model_spec(config.get("model_name")).alias != spec.alias:
        raise ValueError("--model does not match the trained run")
    training_config = resolve_training_config(spec.alias, restore_saved=True, **config)
    folds = _fold_indices(args.folds, training_config.num_folds)
    checkpoints = [run_directory / f"fold_{fold}" / "best.pt" for fold in folds]
    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained fold checkpoint: {path}")
        training_metadata = json.loads((path.parent / "training.json").read_text(encoding="utf-8"))
        if training_metadata.get("status") != "complete":
            raise ValueError(f"Fold training is incomplete: {path.parent}")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be nonnegative")
    selected_device = resolve_device(args.device)
    records = read_inference_records(args.test_csv, args.image_root)

    def loader_factory(image_size, normalization):
        return _data_loader(
            records,
            build_transforms(
                image_size, training=False, mean=normalization.mean, std=normalization.std
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            training=False,
            seed=training_config.seed,
            device=selected_device,
        )

    csv_digest = manifest.get("training_csv_sha256")
    if not isinstance(csv_digest, str) or len(csv_digest) != 64:
        raise ValueError("run.json has no valid training_csv_sha256")
    result = predict_checkpoints(
        checkpoints,
        loader_factory,
        output_directory=args.output_dir,
        device=str(selected_device),
        expected_folds=folds,
        expected_model=spec.alias,
        expected_training_csv_sha256=csv_digest,
        # Compare the original saved shape, including legacy configs with no
        # normalization field, rather than adding new defaults to their identity.
        expected_config=config,
    )
    print(f"Probabilities: {result.probability_path}\nSubmission: {result.submission_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fan-classifier",
        description="Model-name-driven training and prediction for the fan-holding task.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "list-models", help="List the six documented models without downloading weights"
    )
    describe = commands.add_parser(
        "describe-model", help="Show model identity and registered training defaults"
    )
    describe.add_argument("--model", required=True)
    prepare = commands.add_parser(
        "prepare-strata", help="Prepare private label/scene/brightness strata and saved folds"
    )
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--image-root", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    brightness = prepare.add_mutually_exclusive_group()
    brightness.add_argument(
        "--brightness-bins",
        type=int,
        help="Number of quantile bins fitted to training-image brightness (default: 3)",
    )
    brightness.add_argument("--brightness-thresholds", nargs="+", type=float)
    prepare.add_argument("--num-folds", type=int, help="Default: selected model's fold count")
    prepare.add_argument("--seed", type=int, help="Default: selected model's split seed")
    prepare.add_argument("--cluster-seed", default=42, type=int)
    prepare.add_argument("--pca-components", default=50, type=int)
    prepare.add_argument("--num-clusters", default=3, type=int)
    prepare.add_argument("--batch-size", default=16, type=int)
    prepare.add_argument(
        "--clusters-csv",
        type=Path,
        help="Reuse scene_cluster/cluster values aligned by exact training image IDs",
    )
    train = commands.add_parser(
        "train", help="Train selected folds using private images and stratification CSV"
    )
    train.add_argument("--model", required=True)
    train.add_argument("--train-csv", required=True, type=Path)
    train.add_argument("--image-root", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--config", type=Path, help="Optional JSON overrides")
    train.add_argument("--folds", nargs="+", type=int, help="Default: every configured fold")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--num-folds", type=int)
    train.add_argument("--device", default="auto")
    train.add_argument(
        "--normalization",
        choices=("pretrained", "imagenet"),
        help="New runs default to pinned timm mean/std; custom RGB values use --config",
    )
    train.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: initialize from public pretrained backbone weights",
    )
    predict = commands.add_parser("predict", help="Average trained folds and write predictions")
    predict.add_argument("--model", required=True)
    predict.add_argument("--run-dir", required=True, type=Path)
    predict.add_argument("--test-csv", required=True, type=Path)
    predict.add_argument("--image-root", required=True, type=Path)
    predict.add_argument("--output-dir", required=True, type=Path)
    predict.add_argument(
        "--folds", nargs="+", type=int, help="Default: all folds recorded in run.json"
    )
    predict.add_argument("--batch-size", default=4, type=int)
    predict.add_argument("--num-workers", default=0, type=int)
    predict.add_argument("--device", default="auto")
    ensemble = commands.add_parser(
        "ensemble", help="Align image IDs and average per-model probabilities"
    )
    ensemble.add_argument("--inputs", required=True, nargs="+", type=Path)
    ensemble.add_argument("--output-dir", required=True, type=Path)
    evaluate = commands.add_parser("evaluate-oof", help="Evaluate all held-out folds of one run")
    evaluate.add_argument("--run-dir", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    compare = commands.add_parser(
        "compare-oof", help="Compare held-out runs and their equal-weight probability ensemble"
    )
    compare.add_argument("--run-dirs", required=True, nargs="+", type=Path)
    compare.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-models":
            for spec in list_model_specs():
                print(
                    f"{spec.alias:22s} size={spec.image_size:3d} folds={spec.num_folds}  {spec.timm_name}"
                )
        elif args.command == "describe-model":
            print(json.dumps(asdict(get_model_spec(args.model)), ensure_ascii=False, indent=2))
        elif args.command == "prepare-strata":
            from .preparation import prepare_stratification

            result = prepare_stratification(
                args.image_root,
                args.output_dir,
                model_name=args.model,
                brightness_bins=args.brightness_bins,
                brightness_thresholds=args.brightness_thresholds,
                num_folds=args.num_folds,
                seed=args.seed,
                cluster_seed=args.cluster_seed,
                pca_components=args.pca_components,
                cluster_count=args.num_clusters,
                batch_size=args.batch_size,
                clusters_csv=args.clusters_csv,
            )
            print(f"Training CSV: {result.training_csv}\nSplit summary: {result.summary_path}")
        elif args.command == "train":
            _training(args)
        elif args.command == "predict":
            _prediction(args)
        elif args.command in {"evaluate-oof", "compare-oof"}:
            from .oof import evaluate_oof_runs

            directories = [args.run_dir] if args.command == "evaluate-oof" else args.run_dirs
            if args.command == "compare-oof" and len(directories) < 2:
                raise ValueError("compare-oof requires at least two complete runs")
            print(evaluate_oof_runs(directories, args.output_dir))
        elif args.command == "ensemble":
            from .prediction import ensemble_probability_files

            result = ensemble_probability_files(args.inputs, args.output_dir)
            print(f"Probabilities: {result.probability_path}\nSubmission: {result.submission_path}")
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
