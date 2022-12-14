"""Modules for model and dataset.

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from typing import Callable, cast, Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset
from torchvision.models import resnet50, vit_b_16, swin_b, swin_v2_b
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision.models.swin_transformer import (
    Swin_B_Weights,
    Swin_V2_B_Weights,
)
from torchvision.datasets import ImageFolder
from models.CoAtNet import CoAtNet
from models.YOLOv7 import YOLOv7Backbone
from utils.scheduler import CosineAnnealingWithWarmUpLR
from glob import glob
from PIL import Image

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torchmetrics
import os
import cv2
import random


class Classifier(pl.LightningModule):
    """Clasiifer module."""

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 10,
        epoch: int = 30,
        warmup_epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        weights: str = None,
        cutmix: bool = False,
    ) -> None:
        """Initialize instance.

        Args:
            model_name: classification model name
            num_classes: the number of classes
            epoch: the number of epochs
            warmup_epochs: the number of epochs for warm-up
            lr: learning rate
            weight_decay: weight decay
            weights: weight of model
            cutmix: whether apply cutmix or not
        """
        super().__init__()
        if model_name == "resnet50":
            model = resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "vit":
            model = vit_b_16(
                weights=ViT_B_16_Weights,
                # weights=None, # TO-DO: Parameterize weights param
                num_classes=num_classes,
            )
        elif model_name == "swin_b":
            model = swin_b(
                weights=Swin_B_Weights.IMAGENET1K_V1,
                # weights=None, # TO-DO: Parameterize weights param
                num_classes=num_classes,
            )
        elif model_name == "swin_v2_b":
            model = swin_v2_b(
                weights=Swin_V2_B_Weights.IMAGENET1K_V1,
                # weights=None, # TO-DO: Parameterize weights param
                # num_classes=num_classes,
            )
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)
        elif "coatnet" in model_name:
            if model_name == "coatnet0":
                num_blocks = [2, 2, 3, 5, 2]  # L
                channels = [64, 96, 192, 384, 768]  # D
            elif model_name == "coatnet1":
                num_blocks = [2, 2, 6, 14, 2]  # L
                channels = [64, 96, 192, 384, 768]  # D
            elif model_name == "coatnet2":
                num_blocks = [2, 2, 6, 14, 2]  # L
                channels = [128, 128, 256, 512, 1026]  # D
            elif model_name == "coatnet3":
                num_blocks = [2, 2, 6, 14, 2]  # L
                channels = [192, 192, 384, 768, 1536]  # D
            elif model_name == "coatnet4":
                num_blocks = [2, 2, 12, 28, 2]  # L
                channels = [192, 192, 384, 768, 1536]  # D
            else:
                assert False, "Not Supported Model..."

            block_types = ["C", "C", "T", "T"]
            img_size = 224
            in_channels = 3

            model = CoAtNet(
                (img_size, img_size),
                in_channels,
                num_blocks=num_blocks,
                channels=channels,
                block_types=block_types,
                num_classes=num_classes,
            )
        elif "yolov7_backbone" == model_name:
            model = YOLOv7Backbone(num_classes=num_classes, weights=weights)
        else:
            assert False, "Not supported model"

        self.model_name = model_name
        self.cls_nums = num_classes
        self.epoch = epoch
        self.warump_epochs = warmup_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.cutmix = cutmix
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.pred_acc = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Compute and return the training loss and some additional metrics.

        Args:
            batch: batch data
            batch_idx: batch index

        Returns:
            dictionary contains loss, preds, and lables
        """
        inputs, labels = batch
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = self.criterion(outputs, labels)
        labels = labels.max(dim=1)[1] if self.cutmix else labels
        return {"loss": loss, "preds": preds, "labels": labels}

    def training_epoch_end(self, outputs: List) -> None:
        """Called at the end of the training epoch with the outputs.

        Args:
            outputs: all the outputs returned by training_step
        """
        losses = []
        for output in outputs:
            losses.append(self.all_gather(output)["loss"])
            self.train_acc.update(output["preds"], output["labels"])
        loss = torch.mean(torch.stack(losses))

        if self.trainer.is_global_zero:
            self.log("train_loss", loss, rank_zero_only=True)
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Operates on a single batch of data from the validation set.

        Args:
            batch: batch data
            batch_idx: batch index

        Returns:
            dictionary contains loss, preds, and lables
        """
        inputs, labels = batch
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = self.criterion(outputs, labels)
        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs: List) -> None:
        """Called at the end of the validation epoch with the outputs.

        Args:
            outputs: all the outputs returned by validation_step
        """
        losses = []
        for output in outputs:
            losses.append(self.all_gather(output)["loss"])
            self.valid_acc.update(output["preds"], output["labels"])
        loss = torch.mean(torch.stack(losses))

        if self.trainer.is_global_zero:
            self.log("valid_loss", loss, rank_zero_only=True)
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = torch.sigmoid(self.model(inputs))
        probs, preds = torch.max(outputs, 1)
        while "swin" in self.model_name and preds.max() >= self.cls_nums:
            max_pred = preds.max()
            idxs = (preds == max_pred).nonzero().flatten()
            for idx in idxs:
                outputs[idx][max_pred] = 0
                preds[idx] = torch.argmax(outputs[idx])

        return {"preds": preds, "labels": labels, "probs": probs}

    def on_predict_epoch_end(self, results: List) -> None:
        """Called at the end of the prediction epoch with the results.

        Args:
            results: all the outputs returned by training_step
        """
        preds, labels = [], []
        for result in results[0]:
            for dtype, values in result.items():
                if dtype == "preds":
                    preds.append(values)
                elif dtype == "labels":
                    labels.append(values)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        pred_acc = self.pred_acc(preds, labels)
        print(f"Prediction Accuracy: {pred_acc:.4f}")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers
           to use in your optimization.

        Returns:
            optimizer: optimizer to update the model.
        """
        # optimizer = optim.SGD(
        #   self.model.parameters(),
        #   lr=0.001,
        #   momentum=0.9,
        # )
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if not self.warump_epochs:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epoch,
            )
        else:
            scheduler = CosineAnnealingWithWarmUpLR(
                optimizer,
                T_max=self.epoch,
                warmup_epochs=self.warump_epochs,
            )
        return [optimizer], [scheduler]


class ImageWithLabelDataset(ImageFolder):
    """Custom image dataset to handle empty floders."""

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to
        e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory,
                             corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name
                                           to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed.
                Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None
                        or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset()
            # would use the class_to_idx logic of the find_classes() function,
            # instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory,
            class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
        )


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    dirs = os.scandir(directory)
    classes = sorted([entry.name for entry in dirs if entry.is_dir()])
    if not classes:
        error_msg = f"Couldn't find any class folder in {directory}."
        raise FileNotFoundError(error_msg)

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def has_file_allowed_extension(
    filename: str,
    extensions: Union[str, Tuple[str, ...]],
) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(
        extensions if isinstance(extensions, str) else tuple(extensions)
    )


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional
          and will use the logic of the ``find_classes`` function by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be "
            + "None or not None at the same time"
        )

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    # available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    # if target_class not in available_classes:
                    #     available_classes.add(target_class)

    # empty_classes = set(class_to_idx.keys()) - available_classes
    # if empty_classes:
    #     msg = (
    #         f"Found no valid file for the classes
    #           {', '.join(sorted(empty_classes))}. "
    #     )
    #     if extensions is not None:
    #         msg += f"Supported extensions are:
    #                  {extensions if isinstance(extensions, str)
    #                  else ', '.join(extensions)}"
    #     raise FileNotFoundError(msg)

    return instances


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.imgs = glob(f"{img_dir}/*")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx])
        if self.transform:
            image = self.transform(image)
        return image, 0


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class OccludedDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        occ_data: str = None,
        occ_ratio: float = 0.0,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        self.occ_data = occ_data
        self.occ_ratio = occ_ratio
        if self.occ_data:
            self.occ_masks = sorted(glob(f"{self.occ_data}/**/*.npz"))
            self.occ_imgs = []
            for p in self.occ_masks:
                self.occ_imgs.append(p.replace(".npz", ".JPEG"))

    def get_mixed_img(self, sample, img_path, mask_path, p):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path)
        mask = list(mask.values())[0]
        occ_obj = img * mask

        gray_img = cv2.cvtColor(occ_obj.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(
            gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
        )
        x, y, w, h = cv2.boundingRect(contours[0])
        rect = occ_obj[y : y + h, x : x + w]

        new_img = np.array(sample)
        if new_img.shape[0] > new_img.shape[1]:
            x, w = new_img.shape[1] // 4, new_img.shape[1] // 2
            y, h = new_img.shape[0] // 4, new_img.shape[0] // 2
        else:
            x, w = new_img.shape[1] // 4, new_img.shape[1] // 2
            y, h = new_img.shape[0] // 4, new_img.shape[0] // 2

        rect_resized = cv2.resize(rect, (w, h))
        for r_idx, rows in enumerate(new_img[y : y + h, x : x + w, :]):
            for c_idx, cols in enumerate(rows):
                for d_idx, d in enumerate(cols):
                    if rect_resized[r_idx, c_idx, d_idx] == 0:
                        rect_resized[r_idx, c_idx, d_idx] = new_img[
                            y + r_idx, x + c_idx, d_idx
                        ]

        new_img[y : (y + h), x : (x + w), :] = rect_resized
        new_img = Image.fromarray(new_img)
        # new_img.save(f"./preprocess/old/test/{os.path.basename(p)}")
        return new_img

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target)
                   where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.occ_data and random.random() <= self.occ_ratio:
            occ_idx = random.randint(0, len(self.occ_imgs) - 1)
            img_path = self.occ_imgs[occ_idx]
            mask_path = self.occ_masks[occ_idx]
            sample = self.get_mixed_img(sample, img_path, mask_path, path)
        # else:
        #     sample.save(f"./preprocess/old/test/{os.path.basename(path)}")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
