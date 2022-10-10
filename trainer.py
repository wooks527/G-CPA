"""Trainer.

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from modules import Classifier, OccludedDataset
from utils.transforms import RandomMixup, RandomCutmix
from utils.general import init_seeds

import pytorch_lightning as pl
import argparse


def parse_args() -> argparse.Namespace:
    """Parse arguments.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        help="dataset directory",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        help="train dataset directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="model name",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="the number of classes",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=128,
        help="batch size",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="epoch",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="warm-up epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="weight decay",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="weight of model",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="resnet50",
        help="experiment name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    parser.add_argument(
        "--swa",
        type=int,
        default=1,
        help="apply swa",
    )
    parser.add_argument(
        "--mixup",
        action="store_true",
        help="apply Mixup",
    )
    parser.add_argument(
        "--cutmix",
        action="store_true",
        help="apply CutMix",
    )
    parser.add_argument(
        "--random_erase",
        action="store_true",
        help="apply RandomErase",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bilinear",
        help="interpolation method",
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default="0.0",
        help="gradient clipping",
    )
    parser.add_argument(
        "--occ_data",
        type=str,
        default=None,
        help="occlusion data directory path",
    )
    parser.add_argument(
        "--occ_ratio",
        type=float,
        default="0.0",
        help="occlusion ratio",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_seeds(seed=args.seed)
    interpolation = InterpolationMode(args.interpolation)
    ra_interpolation = interpolation
    if args.interpolation == "bilinear":
        ra_interpolation = InterpolationMode("nearest")

    train_transforms = [
        transforms.RandAugment(interpolation=ra_interpolation),
        # transforms.TrivialAugmentWide(interpolation=ra_interpolation),
        transforms.RandomResizedCrop(224, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if args.random_erase:
        train_transforms.append(transforms.RandomErasing(p=0.25))
    train_transforms = transforms.Compose(train_transforms)

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=interpolation),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if not args.occ_data:
        train_dataset = datasets.ImageFolder(
            f"{args.train_data}",
            transform=train_transforms,
        )
    else:
        train_dataset = OccludedDataset(
            f"{args.train_data}",
            transform=train_transforms,
            occ_data=args.occ_data,
            occ_ratio=args.occ_ratio,
        )
    val_dataset = datasets.ImageFolder(
        f"{args.data}/val/Images",
        transform=val_transforms,
    )

    mix_transforms = []
    if args.mixup:
        mixup = RandomMixup(args.num_classes, p=1.0, alpha=0.8)
        mix_transforms.append(mixup)
    if args.cutmix:
        cutmix = RandomCutmix(args.num_classes, p=1.0, alpha=1.0)
        mix_transforms.append(cutmix)
    if mix_transforms:
        mixupcutmix = transforms.RandomChoice(mix_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=12,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        num_workers=12,
    )

    model = Classifier(
        model_name=args.model,
        num_classes=args.num_classes,
        epoch=args.epoch,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        weights=args.weights,
        cutmix=args.cutmix,
    )

    tb_logger = TensorBoardLogger(save_dir=f"logs/{args.exp_name}")
    wandb_logger = WandbLogger(
        save_dir=f"logs/{args.exp_name}",
        project="ood-cv-cls",
        name=args.exp_name,
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=30,
        monitor="valid_acc",
        mode="max",
        dirpath=f"logs/{args.exp_name}/checkpoints",
        filename="{epoch:02d}-{valid_acc:.4f}",
    )
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        devices=2,
        strategy="ddp",
        accelerator="gpu",
        logger=[tb_logger, wandb_logger],
        callbacks=[checkpoint_callback, lr_callback],
        accumulate_grad_batches=args.swa,
        gradient_clip_val=args.gradient_clip,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
