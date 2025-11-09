from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import (
    TASKS,
    SoilDataset,
    branch_targets_from_dict,
    build_modal_arrays,
    discover_files,
    kennard_stone,
    load_libs,
    load_soil,
    load_vnir,
    load_xrf,
    normalise_targets,
)
from data.dataset import Standardizers  # type: ignore
from distillation.trainer import DistillationTrainer, evaluate_model
from evaluation.metrics import aggregate_metrics
from evaluation.plots import plot_scatter
from models.student import StudentNet
from models.teacher import train_single_modal_predictors, train_teacher_ensemble
from utils.log import get_logger, setup_logging
from utils.seed import set_seed


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_modalities(data_cfg: Dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, str]]:
    base = Path(data_cfg.get("root", "data_unzip")).resolve()
    unzip = Path(data_cfg.get("unzip_dir", base)).resolve()
    paths = discover_files(str(base), str(unzip))
    soil = load_soil(paths["soil"])
    vnir = load_vnir(paths["vnir"])
    xrf = load_xrf(paths["xrf"])
    _, wavelength, libs_matrix = load_libs(paths["libs"])

    arrays_cfg = {
        "sg_window": data_cfg["sg_window"],
        "sg_poly": data_cfg["sg_poly"],
        "sg_deriv": data_cfg["sg_deriv"],
        "libs_ranges": data_cfg["libs_ranges"],
        "libs_band": data_cfg["libs_band"],
    }
    Xv, Xx, Xl_by_task, Xl_full, targets = build_modal_arrays(
        soil,
        vnir,
        xrf,
        libs_matrix,
        wavelength,
        arrays_cfg,
    )
    return Xv, Xx, Xl_by_task, Xl_full, targets, paths


def make_standardizers(x_vnir: np.ndarray, x_xrf: np.ndarray, train_idx: np.ndarray) -> Standardizers:
    from sklearn.preprocessing import StandardScaler

    sc_v = StandardScaler().fit(x_vnir[train_idx])
    sc_x = StandardScaler().fit(x_xrf[train_idx])
    return Standardizers(sc_v, sc_x)


def transform_modalities(
    scalers: Standardizers,
    x_vnir: np.ndarray,
    x_xrf: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xv_train, Xx_train = scalers.transform(x_vnir[train_idx], x_xrf[train_idx])
    Xv_val, Xx_val = scalers.transform(x_vnir[val_idx], x_xrf[val_idx])
    return Xv_train, Xv_val, Xx_train, Xx_val


def run_train(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger = get_logger("train")

    seed_base = int(cfg.get("seed", 34))
    set_seed(seed_base)

    data_cfg = cfg["data"]
    Xv, Xx, Xl_by_task, Xl_full, targets, paths = load_modalities(data_cfg)
    logger.info("Data files: %s", json.dumps(paths, ensure_ascii=False, indent=2))

    n_samples = targets.shape[0]
    train_ratio = float(data_cfg.get("train_ratio", 0.7))
    n_train = max(1, min(n_samples - 1, int(round(n_samples * train_ratio))))
    split_seed = int(data_cfg.get("split_seed", seed_base))
    train_idx, val_idx = kennard_stone(targets, n_train=n_train, seed=split_seed)
    logger.info("Split: train=%d val=%d", len(train_idx), len(val_idx))

    scalers = make_standardizers(Xv, Xx, train_idx)
    Xv_train, Xv_val, Xx_train, Xx_val = transform_modalities(scalers, Xv, Xx, train_idx, val_idx)
    Xl_train = Xl_full[train_idx]
    Xl_val = Xl_full[val_idx]
    y_train = targets[train_idx]
    y_val = targets[val_idx]

    teacher_cfg = cfg["teacher"]
    teacher_report = train_teacher_ensemble(teacher_cfg, Xv, Xx, Xl_by_task, targets, train_idx, val_idx)
    branch_train_raw, branch_val_raw = train_single_modal_predictors(Xv, Xx, Xl_by_task, targets, train_idx, val_idx)
    teacher_metrics = aggregate_metrics(y_val, teacher_report.val_predictions, TASKS)
    logger.info("Teacher mean R2=%.4f RMSE=%.4f", teacher_metrics["mean"]["R2"], teacher_metrics["mean"]["RMSE"])

    target_mean, target_std, y_train_n, y_val_n = normalise_targets(y_train, y_val)
    teacher_train_n = (teacher_report.train_predictions - target_mean) / target_std
    teacher_val_n = (teacher_report.val_predictions - target_mean) / target_std

    branch_train = branch_targets_from_dict(branch_train_raw)
    branch_val = branch_targets_from_dict(branch_val_raw)
    for key in branch_train:
        branch_train[key] = ((branch_train[key] - target_mean) / target_std).astype(np.float32)
        branch_val[key] = ((branch_val[key] - target_mean) / target_std).astype(np.float32)

    train_dataset = SoilDataset(
        Xv_train,
        Xx_train,
        Xl_train,
        y_train_n.astype(np.float32),
        teacher_train_n.astype(np.float32),
        branch_train,
    )
    val_dataset = SoilDataset(
        Xv_val,
        Xx_val,
        Xl_val,
        y_val_n.astype(np.float32),
        teacher_val_n.astype(np.float32),
        branch_val,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    train_cfg = cfg["training"]
    artifacts_dir = Path(cfg.get("artifacts", {}).get("dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in cfg.get("seeds", [seed_base]):
        seed = int(seed)
        set_seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch"],
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=pin_memory,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg["batch"],
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=pin_memory,
        )

        student_cfg = cfg["student"]
        model = StudentNet(
            student_cfg["branches"]["V"],
            student_cfg["branches"]["X"],
            student_cfg["branches"]["L"],
            student_cfg["prior"],
            student_cfg["moe"],
        ).to(device)

        trainer = DistillationTrainer(model, train_cfg)
        best_r2, best_epoch, metrics = trainer.train(
            train_loader,
            val_loader,
            target_mean,
            target_std,
            y_val,
            train_cfg["epochs"],
            train_cfg["patience"],
        )
        logger.info(
            "Seed %d | best mean R2=%.4f @ epoch %d | final mean R2=%.4f RMSE=%.4f",
            seed,
            best_r2,
            best_epoch,
            metrics["mean"]["R2"],
            metrics["mean"]["RMSE"],
        )
        for task in TASKS:
            logger.debug(
                "Seed %d | %s | R2=%.4f RMSE=%.4f",
                seed,
                task,
                metrics[task]["R2"],
                metrics[task]["RMSE"],
            )
        ckpt_path = artifacts_dir / f"student_seed{seed}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "target_mean": target_mean,
                "target_std": target_std,
                "scaler_vnir": scalers.vnir,
                "scaler_xrf": scalers.xrf,
                "metrics": metrics,
                "teacher_metrics": teacher_metrics,
                "config": cfg,
            },
            ckpt_path,
        )
        logger.info("Saved checkpoint to %s", ckpt_path)
        seed_results.append({"seed": seed, "metrics": metrics})

    mean_r2 = np.array([item["metrics"]["mean"]["R2"] for item in seed_results])
    mean_rmse = np.array([item["metrics"]["mean"]["RMSE"] for item in seed_results])
    logger.info(
        "Aggregate mean R2=%.4f ± %.4f | mean RMSE=%.4f ± %.4f",
        mean_r2.mean(),
        mean_r2.std(ddof=1) if len(mean_r2) > 1 else 0.0,
        mean_rmse.mean(),
        mean_rmse.std(ddof=1) if len(mean_rmse) > 1 else 0.0,
    )


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def build_model_from_config(cfg: Dict) -> StudentNet:
    student_cfg = cfg["student"]
    return StudentNet(
        student_cfg["branches"]["V"],
        student_cfg["branches"]["X"],
        student_cfg["branches"]["L"],
        student_cfg["prior"],
        student_cfg["moe"],
    )


def run_evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger = get_logger("evaluate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(Path(args.checkpoint), device)

    data_cfg = cfg["data"]
    Xv, Xx, Xl_by_task, Xl_full, targets, _ = load_modalities(data_cfg)
    val_idx = np.asarray(ckpt["val_idx"], dtype=int)

    scaler_v = ckpt["scaler_vnir"]
    scaler_x = ckpt["scaler_xrf"]
    Xv_val = scaler_v.transform(Xv[val_idx]).astype(np.float32)
    Xx_val = scaler_x.transform(Xx[val_idx]).astype(np.float32)
    Xl_val = Xl_full[val_idx]
    y_val = targets[val_idx]
    target_mean = ckpt["target_mean"]
    target_std = ckpt["target_std"]
    y_val_norm = ((y_val - target_mean) / target_std).astype(np.float32)

    dataset = SoilDataset(Xv_val, Xx_val, Xl_val, y_val_norm)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    metrics = evaluate_model(model, loader, target_mean, target_std, y_val)
    logger.info(
        "Checkpoint %s | mean R2=%.4f RMSE=%.4f",
        args.checkpoint,
        metrics["mean"]["R2"],
        metrics["mean"]["RMSE"],
    )
    for task in TASKS:
        logger.info("%s | R2=%.4f RMSE=%.4f", task, metrics[task]["R2"], metrics[task]["RMSE"])


def run_visualize(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    cfg = load_config(Path(args.config))
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger = get_logger("visualize")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(Path(args.checkpoint), device)

    data_cfg = cfg["data"]
    Xv, Xx, Xl_by_task, Xl_full, targets, _ = load_modalities(data_cfg)
    val_idx = np.asarray(ckpt["val_idx"], dtype=int)

    scaler_v = ckpt["scaler_vnir"]
    scaler_x = ckpt["scaler_xrf"]
    Xv_val = scaler_v.transform(Xv[val_idx]).astype(np.float32)
    Xx_val = scaler_x.transform(Xx[val_idx]).astype(np.float32)
    Xl_val = Xl_full[val_idx]
    y_val = targets[val_idx]
    target_mean = ckpt["target_mean"]
    target_std = ckpt["target_std"]
    y_val_norm = ((y_val - target_mean) / target_std).astype(np.float32)

    dataset = SoilDataset(Xv_val, Xx_val, Xl_val, y_val_norm)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            outputs = model(batch[0], batch[1], batch[2])
            preds.append(outputs.cpu().numpy())
    preds = np.vstack(preds)
    denorm = preds * target_std + target_mean

    out_dir = Path(args.output or "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, task in enumerate(TASKS):
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_scatter(y_val[:, idx], denorm[:, idx], f"{task} Scatter", ax=ax)
        fig.savefig(out_dir / f"{task}_scatter.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved scatter plots to %s", out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Soil multitask distillation CLI")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Run knowledge distillation training.")
    train_p.set_defaults(func=run_train)

    eval_p = sub.add_parser("evaluate", help="Evaluate a saved checkpoint on validation split.")
    eval_p.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    eval_p.set_defaults(func=run_evaluate)

    vis_p = sub.add_parser("visualize", help="Generate scatter plots for a checkpoint.")
    vis_p.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    vis_p.add_argument("--output", help="Directory for plots.")
    vis_p.set_defaults(func=run_visualize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
