"""Trainer.

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from modules import Classifier

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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_transforms = transforms.Compose(
        [
            transforms.RandAugment(),
            # transforms.RandomResizedCrop(224),
            transforms.RandomResizedCrop(640),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(672),
            transforms.CenterCrop(640),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = datasets.ImageFolder(
        f"{args.data}/train",
        # f"{args.data}/imagenet_c10",
        # f"{args.data}/train_with_imgc10",
        transform=train_transforms,
    )
    val_dataset = datasets.ImageFolder(
        f"{args.data}/val_nui",
        # f"{args.data}/val",
        transform=val_transforms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=32,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        num_workers=32,
    )

    model = Classifier(
        model_name=args.model,
        num_classes=args.num_classes,
        epoch=args.epoch,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        weights=args.weights,
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
        # gradient_clip_val=5.0,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
