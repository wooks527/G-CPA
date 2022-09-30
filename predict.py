"""Predict and create submission csv file..

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning.loggers import TensorBoardLogger
from modules import Classifier, ImageWithLabelDataset, ImageDataset

import pytorch_lightning as pl
import pandas as pd
import argparse
import torch
import os


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
        "--batch",
        type=int,
        default=128,
        help="batch size",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="last.ckpt",
        help="checkpoint file",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="experiment name",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="test dataset",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    try:
        test_dataset = datasets.ImageFolder(
            f"{args.data}",
            transform=test_transforms,
        )
    except Exception as e:
        print(e)
        try:
            test_dataset = ImageWithLabelDataset(
                f"{args.data}",
                transform=test_transforms,
            )
        except Exception as e:
            print(e)
            test_dataset = ImageDataset(
                f"{args.data}",
                transform=test_transforms,
            )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=12,
    )

    model = Classifier.load_from_checkpoint(args.ckpt)
    tb_logger = TensorBoardLogger(save_dir=f"results/{args.exp_name}")
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=tb_logger,
    )

    print(f"\n[{args.test_data}] Start Prediction:")
    results = trainer.predict(model, test_loader)
    if hasattr(test_dataset, "classes"):
        imgs = [os.path.basename(img) for img, _ in test_dataset.imgs]
        classes = test_dataset.classes
    else:
        imgs = [os.path.basename(img) for img in test_dataset.imgs]
        sample_df = pd.read_csv(f"csvs/{args.test_data}.csv")
        classes = sorted(set(sample_df["pred"]))

    idx_to_cls = {idx: cls_name for idx, cls_name in enumerate(classes)}
    preds_list = torch.cat([result["preds"] for result in results]).tolist()
    preds_list = [idx_to_cls[pred] for pred in preds_list]
    results_dict = {img: pred for img, pred in zip(imgs, preds_list)}

    df_preds = pd.read_csv(f"csvs/{args.test_data}.csv")
    for idx, row in df_preds.iterrows():
        df_preds.iloc[idx]["pred"] = results_dict[row["imgs"]]
    df_preds.to_csv(
        f"results/{args.exp_name}/{args.test_data}.csv",
        index=False,
    )
    print("Save prediction results.")
