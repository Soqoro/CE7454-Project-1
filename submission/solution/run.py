# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2025-02-18 19:09:59
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-02-19 08:13:52
# @Email:  root@haozhexie.com

import argparse
import cv2
import torch

from PIL import Image


def main(input, output, weights):
    # Load the input image
    img = cv2.imread(input)

    # TODO: Initialize the neural network model
    # Example:
    # from models import YourSegModel
    # model = YourSegModel()
    model = None

    # Load the checkpoint
    ckpt = torch.load(weights)
    # NOTE: Make sure that the weights are saved in the "state_dict" key
    # DO NOT CHANGE THIS VALUE, i.e., ckpt["state_dict"]
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Inference with the model (Update as needed)
    # Normalize the image.
    # NOTE: Make sure it is aligned with the training data
    # Example: img = (img / 255.0 - 0.5) * 2.0
    prediction = model(img)

    # Convert PyTorch Tensor to numpy array
    mask = prediction.cpu().numpy()
    # Save the prediction
    Image.fromarray(mask.convert("P")).save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    args = parser.parse_args()
    main(args.input, args.output, args.weights)
