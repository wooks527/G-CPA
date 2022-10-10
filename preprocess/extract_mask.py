from glob import glob
from tqdm import tqdm
from skimage.measure import label

import argparse
import cv2
import numpy as np
import torch
import timm
import os
import time

from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
)

from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Use NVIDIA GPU acceleration",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU deivcie number",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./examples/both.png",
        help="Input image path",
    )
    parser.add_argument(
        "--aug_smooth",
        action="store_true",
        help="Apply test time augmentation to smooth the CAM",
    )
    parser.add_argument(
        "--eigen_smooth",
        action="store_true",
        help="Reduce noise by taking the first principle componenet"
        "of cam_weights*activations",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="scorecam",
        help="Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam",
    )

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == "__main__":
    """python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
    }

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
    model.eval()

    if args.use_cuda:
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)  # change allocation of current GPU
        model = model.cuda()

    target_layers = [model.layers[-1].blocks[-1].norm2]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](
            model=model,
            target_layers=target_layers,
            use_cuda=args.use_cuda,
            reshape_transform=reshape_transform,
            ablation_layer=AblationLayerVit(),
        )
    else:
        cam = methods[args.method](
            model=model,
            target_layers=target_layers,
            use_cuda=args.use_cuda,
            reshape_transform=reshape_transform,
        )

    t0 = time.time()
    for img_path in tqdm(glob(f"{args.data}/**/*.JPEG")):
        print(img_path)
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img_resized = cv2.resize(rgb_img, (224, 224))
        rgb_img_norm = np.float32(rgb_img_resized) / 255
        input_tensor = preprocess_image(
            rgb_img_norm, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 1024

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=None,
            eigen_smooth=args.eigen_smooth,
            aug_smooth=args.aug_smooth,
        )

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam.reshape(224, 224, 1)
        grayscale_cam = cv2.resize(grayscale_cam, rgb_img.shape[:2][::-1])[
            :, :, np.newaxis
        ]

        threshold = 0.2
        num_pixels = grayscale_cam.shape[0] * grayscale_cam.shape[1]
        num_masks = np.count_nonzero(grayscale_cam >= threshold)
        mask_ratio = num_masks / num_pixels
        if mask_ratio > 0.5 or mask_ratio < 0.1:
            continue

        grayscale_cam[grayscale_cam >= threshold] = 1
        grayscale_cam[grayscale_cam < threshold] = 0
        cls = os.path.dirname(img_path).split("/")[-1]
        img_fname = os.path.basename(img_path).split(".")[0]

        masks = label(grayscale_cam)
        num_clusters = len(np.unique(masks))
        if num_clusters == 2:
            np.savez_compressed(
                f"{args.data}/{cls}/{img_fname}.npz",
                grayscale_cam,
            )

    print(f"Extracted Time: {time.time() - t0:.2f}")
