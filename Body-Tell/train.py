"""Body-Tell Phase 1 training script.

Usage:
    # Single GPU
    python train.py --config configs/phase1_voxtell_body.yaml
    # Smoke test
    python train.py --config configs/phase1_voxtell_body.yaml --smoke --volume-size 72 64 128 --amp
    # Multi-GPU (DDP via torchrun)
    torchrun --nproc_per_node=2 train.py --config configs/phase1_voxtell_body.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from body_tell.data.dataset import HyperBodyPromptDataset, prompt_collate_fn
from body_tell.losses.prompt_losses import PromptSegmentationLoss
from body_tell.metrics.prompt_metrics import compute_prompt_metrics
from body_tell.models.voxtell_body_model import VoxTellBodyConfig, VoxTellBodyModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_cfg(d: dict[str, Any], parent: str = "") -> dict[str, Any]:
    """Flatten nested cfg dict to dotted keys for wandb.config UI filtering."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten_cfg(v, key))
        else:
            out[key] = v
    return out


def _init_wandb(args: argparse.Namespace, cfg: dict[str, Any]):
    """Init W&B on rank 0 only. Returns Run or None."""
    if not args.wandb or not is_main_process():
        return None
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError(
            "--wandb passed but wandb is not installed. "
            "Run `pip install wandb>=0.18.0`."
        ) from e

    mode = "offline" if args.wandb_offline else "online"
    name = args.wandb_run_name or f"phase1-{time.strftime('%Y%m%d-%H%M%S')}"
    tags = [t for t in (args.wandb_tags or "").split(",") if t]
    if args.smoke:
        tags.append("smoke")

    project = args.wandb_project or cfg.get("project", "Body-Tell")

    run = wandb.init(
        project=project,
        entity=args.wandb_entity,
        name=name,
        mode=mode,
        config=_flatten_cfg(cfg) | {
            "cli.lr": args.lr,
            "cli.batch_size": args.batch_size,
            "cli.epochs": args.epochs,
            "cli.amp": args.amp,
            "cli.smoke": args.smoke,
            "cli.volume_size": args.volume_size,
        },
        tags=tags or None,
        dir=args.wandb_dir,
        resume="allow" if args.wandb_resume_id else None,
        id=args.wandb_resume_id,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("best/*", step_metric="epoch")
    return run


def build_model(cfg: dict[str, Any]) -> VoxTellBodyModel:
    mc = cfg["model"]
    config = VoxTellBodyConfig(
        input_channels=mc["input_channels"],
        encoder_channels=tuple(mc["encoder_channels"]),
        text_embedding_dim=mc["text_embedding_dim"],
        query_dim=mc["query_dim"],
        text_projection_hidden_dim=mc["text_projection_hidden_dim"],
        transformer_num_heads=mc["transformer_num_heads"],
        transformer_layers=mc["transformer_layers"],
        transformer_feedforward_dim=mc["transformer_feedforward_dim"],
        decoder_layer=mc["decoder_layer"],
        num_maskformer_stages=mc["num_maskformer_stages"],
        num_heads=mc["num_heads"],
        deep_supervision=mc["deep_supervision"],
    )
    return VoxTellBodyModel(config)


def build_dataset(
    cfg: dict[str, Any],
    split: str,
    volume_size_override: tuple[int, ...] | None = None,
) -> HyperBodyPromptDataset:
    dc = cfg["data"]
    volume_size = volume_size_override or tuple(dc["volume_size"])
    return HyperBodyPromptDataset(
        root=dc["root"],
        split=split,
        volume_size=volume_size,
        num_positive=dc["num_positive"],
        num_negative=dc["num_negative"],
        min_voxels=dc.get("min_voxels", 1),
    )


def build_loss(cfg: dict[str, Any]) -> PromptSegmentationLoss:
    lc = cfg["loss"]
    return PromptSegmentationLoss(
        bce_weight=lc["bce_weight"],
        dice_weight=lc["dice_weight"],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: PromptSegmentationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_fg_dice = 0.0
    total_neg_fp = 0.0
    n_batches = 0

    for batch in tqdm(
        loader,
        desc=f"epoch {epoch}",
        disable=not is_main_process(),
    ):
        occupancy = batch["occupancy"].to(device)
        text_embeddings = batch["text_embeddings"].to(device)
        target_masks = batch["target_masks"].to(device)
        target_empty = batch["target_empty"].to(device)
        prompt_valid = batch.get("prompt_valid")
        if prompt_valid is not None:
            prompt_valid = prompt_valid.to(device)

        optimizer.zero_grad()

        with autocast("cuda", enabled=use_amp):
            logits = model(occupancy, text_embeddings)
            result = criterion(logits, target_masks, prompt_valid=prompt_valid)

        if scaler is not None:
            scaler.scale(result["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            result["loss"].backward()
            optimizer.step()

        with torch.no_grad():
            metrics = compute_prompt_metrics(
                logits.float(),
                target_masks,
                target_empty=target_empty,
                prompt_valid=prompt_valid,
            )

        total_loss += result["loss"].item()
        total_bce += result["bce_loss"].item()
        total_dice += result["dice_loss"].item()
        total_fg_dice += metrics["foreground_mean_dice"]
        total_neg_fp += metrics["negative_fp_rate"]
        n_batches += 1

    avg = {
        "loss": total_loss / n_batches,
        "bce_loss": total_bce / n_batches,
        "dice_loss": total_dice / n_batches,
        "foreground_mean_dice": total_fg_dice / n_batches,
        "negative_fp_rate": total_neg_fp / n_batches,
    }
    if is_main_process():
        log.info(
            "Epoch %03d | loss=%.4f (bce=%.4f dice=%.4f) | fg_dice=%.4f | neg_fp=%.4f",
            epoch,
            avg["loss"],
            avg["bce_loss"],
            avg["dice_loss"],
            avg["foreground_mean_dice"],
            avg["negative_fp_rate"],
        )
    return avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: PromptSegmentationLoss,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_fg_dice = 0.0
    total_neg_fp = 0.0
    n_batches = 0

    for batch in tqdm(
        loader,
        desc="val",
        disable=not is_main_process(),
    ):
        occupancy = batch["occupancy"].to(device)
        text_embeddings = batch["text_embeddings"].to(device)
        target_masks = batch["target_masks"].to(device)
        target_empty = batch["target_empty"].to(device)
        prompt_valid = batch.get("prompt_valid")
        if prompt_valid is not None:
            prompt_valid = prompt_valid.to(device)

        with autocast("cuda", enabled=use_amp):
            logits = model(occupancy, text_embeddings)
            result = criterion(logits, target_masks, prompt_valid=prompt_valid)

        metrics = compute_prompt_metrics(
            logits.float(),
            target_masks,
            target_empty=target_empty,
            prompt_valid=prompt_valid,
        )

        total_loss += result["loss"].item()
        total_fg_dice += metrics["foreground_mean_dice"]
        total_neg_fp += metrics["negative_fp_rate"]
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "foreground_mean_dice": 0.0, "negative_fp_rate": 0.0}

    avg = {
        "loss": total_loss / n_batches,
        "foreground_mean_dice": total_fg_dice / n_batches,
        "negative_fp_rate": total_neg_fp / n_batches,
    }
    if is_main_process():
        log.info(
            "  [val] loss=%.4f | fg_dice=%.4f | neg_fp=%.4f",
            avg["loss"],
            avg["foreground_mean_dice"],
            avg["negative_fp_rate"],
        )
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Body-Tell Phase 1 training")
    parser.add_argument(
        "--config", type=str, default="configs/phase1_voxtell_body.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-samples", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--volume-size", type=int, nargs=3, default=None, metavar=("D", "H", "W"),
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B (rank-0 only)")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="Defaults to cfg['project']",
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-tags", type=str, default=None, help="Comma-separated tags",
    )
    parser.add_argument("--wandb-dir", type=str, default="./wandb")
    parser.add_argument(
        "--wandb-resume-id", type=str, default=None, help="Resume crashed run by ID",
    )
    args = parser.parse_args()

    # DDP setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    if is_main_process():
        log.info("Device: %s (distributed=%s)", device, distributed)

    cfg = load_config(args.config)

    wb_run = _init_wandb(args, cfg)

    try:
        use_amp = args.amp and device.type == "cuda"
        scaler = GradScaler("cuda") if use_amp else None
        if use_amp and is_main_process():
            log.info("Mixed precision (AMP) enabled")

        epochs = args.epochs if args.epochs is not None else (200 if args.smoke else 50)
        batch_size = args.batch_size
        volume_size_override = tuple(args.volume_size) if args.volume_size else None

        if volume_size_override and is_main_process():
            log.info("Volume size override: %s", volume_size_override)

        train_dataset = build_dataset(cfg, split="train", volume_size_override=volume_size_override)
        if is_main_process():
            log.info("Train dataset: %d samples", len(train_dataset))

        if args.smoke:
            n_smoke = min(args.smoke_samples, len(train_dataset))
            train_dataset = Subset(train_dataset, list(range(n_smoke)))
            if is_main_process():
                log.info("Smoke mode: using %d samples for %d epochs", n_smoke, epochs)

        train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=prompt_collate_fn,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        val_dataset = build_dataset(cfg, split="val", volume_size_override=volume_size_override)
        val_loader = None
        if len(val_dataset) > 0 and not args.smoke:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=prompt_collate_fn,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            if is_main_process():
                log.info("Val dataset: %d samples", len(val_dataset))

        model = build_model(cfg).to(device)
        if distributed:
            model = DDP(model, device_ids=[local_rank])

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_main_process():
            log.info("Model parameters: %s (%.2fM)", f"{n_params:,}", n_params / 1e6)

        if wb_run is not None:
            wb_run.config.update(
                {
                    "n_params": n_params,
                    "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                    "use_amp_effective": use_amp,
                },
                allow_val_change=True,
            )

        criterion = build_loss(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

        ckpt_dir = Path(args.checkpoint_dir)
        if is_main_process():
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_dice = 0.0
        history: list[dict[str, float]] = []

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch,
                scaler=scaler, use_amp=use_amp,
            )
            history.append(train_metrics)

            if wb_run is not None:
                payload = {f"train/{k}": v for k, v in train_metrics.items()}
                payload["epoch"] = epoch
                payload["lr"] = optimizer.param_groups[0]["lr"]
                wb_run.log(payload)

            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
                if wb_run is not None:
                    val_payload = {f"val/{k}": v for k, v in val_metrics.items()}
                    val_payload["epoch"] = epoch
                    wb_run.log(val_payload)
                if is_main_process() and val_metrics["foreground_mean_dice"] > best_dice:
                    best_dice = val_metrics["foreground_mean_dice"]
                    state_dict = model.module.state_dict() if distributed else model.state_dict()
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": state_dict,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "metrics": val_metrics,
                            "config": cfg,
                        },
                        ckpt_dir / "best.pt",
                    )
                    log.info("  Saved best checkpoint (dice=%.4f)", best_dice)
                    if wb_run is not None:
                        wb_run.summary["best/foreground_mean_dice"] = best_dice
                        wb_run.summary["best/epoch"] = epoch
                        wb_run.summary["best/val_loss"] = val_metrics["loss"]
                        wb_run.summary["best/negative_fp_rate"] = val_metrics["negative_fp_rate"]

            if is_main_process() and args.smoke and epoch % 50 == 0:
                state_dict = model.module.state_dict() if distributed else model.state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics,
                        "config": cfg,
                    },
                    ckpt_dir / f"smoke_epoch{epoch:03d}.pt",
                )

        elapsed = time.time() - t0
        if is_main_process():
            log.info("Training complete in %.1fs (%d epochs)", elapsed, epochs)

            if args.smoke:
                first_loss = history[0]["loss"]
                last_loss = history[-1]["loss"]
                last_dice = history[-1]["foreground_mean_dice"]
                log.info(
                    "Smoke result: loss %.4f -> %.4f (%.1f%% reduction), final fg_dice=%.4f",
                    first_loss,
                    last_loss,
                    100 * (1 - last_loss / first_loss),
                    last_dice,
                )
                if last_loss >= first_loss:
                    log.warning("SMOKE FAILED: loss did not decrease!")
                elif last_dice < 0.1:
                    log.warning("SMOKE WARNING: Dice still very low after overfit attempt")
                else:
                    log.info("SMOKE PASSED: model shows learning signal")

            state_dict = model.module.state_dict() if distributed else model.state_dict()
            torch.save(
                {
                    "epoch": epochs,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": history[-1],
                    "config": cfg,
                    "history": history,
                },
                ckpt_dir / "last.pt",
            )
            log.info("Saved final checkpoint to %s/last.pt", ckpt_dir)
    finally:
        if wb_run is not None:
            wb_run.finish()
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
