from tqdm import tqdm
from glob import glob

import pandas as pd
import argparse
import os
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        help="csv path of prediction results",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        help="training data directory",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        help="test data directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="new data directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="threshold of prediction probability",
    )
    args = parser.parse_args()

    pred_df = pd.read_csv(args.csv_path)
    CLASSES = pred_df["pred"].unique().tolist()
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    for cls in CLASSES:
        os.makedirs(f"{args.out_dir}/{cls}")

    count = 0
    for idx, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
        if row["prob"] > args.threshold:
            img_path = f"{args.test_dir}/{row['imgs']}"
            target_path = f"{args.out_dir}/{row['pred']}/{row['imgs']}"
            shutil.copyfile(img_path, target_path)
            count += 1

    print(f"Extracted: {count}/{len(pred_df)}")

    train_img_paths = glob(f"{args.train_dir}/**/*")
    for train_img_path in tqdm(train_img_paths):
        cls = train_img_path.split("/")[-2]
        img_fname = os.path.basename(train_img_path)
        target_path = f"{args.out_dir}/{cls}/{img_fname}"
        shutil.copyfile(train_img_path, target_path)

    print("Original training images were copied.")
