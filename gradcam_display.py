# @misc{jacobgilpytorchcam,
#   title={PyTorch library for CAM methods},
#   author={Jacob Gildenblat and contributors},
#   year={2021},
#   publisher={GitHub},
#   howpublished={\url{https://github.com/jacobgil/pytorch-grad-cam}},
# }

import argparse
from PIL import Image
import requests, io, os
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
from train_sa import new_clsnet
from imagecorruptions import corrupt
from pytorch_grad_cam import GradCAM
# \
# HiResCAM, \
# ScoreCAM, \
# GradCAMPlusPlus, \
# AblationCAM, \
# XGradCAM, \
# EigenCAM, \
# EigenGradCAM, \
# LayerCAM, \
# FullGrad, \
# GradCAMElementWise

from pytorch_grad_cam.utils.image import show_cam_on_image


def parse():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
    parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                        help='model architecture (default: resnet50)')
    parser.add_argument('--read_path', default='./pic',
                        help='picture path to show')
    parser.add_argument('--save_path', default='./save',
                        help='picture path to save')
    parser.add_argument('--target_class', default=None, type=int)

    parser.add_argument('--mean', type=float, nargs='+', default=[0.485, 0.456, 0.406], metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=[0.229, 0.224, 0.225], metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default=3, type=int,
                        help='1: lanczos 2: bilinear 3: bicubic')
    parser.add_argument('--input-size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--crop-pct', default=0.875, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number classes in dataset')
    # parser.add_argument('--method', type=str, default='gradcam',
    #                     choices=['gradcam', 'hirescam', 'gradcam++',
    #                              'scorecam', 'xgradcam',
    #                              'ablationcam', 'eigencam',
    #                              'eigengradcam', 'layercam', 'fullgrad'])
    parser.add_argument('--method', type=str, default='clean')

    args = parser.parse_args()
    return args

def main(args, filename, read_path, save_path):
    input_image = os.path.join(read_path, filename)
    model = resnet50(pretrained=True)
    model = new_clsnet(model)
    state_dict = torch.load("./checkpoints/SA/best.pth")
    model.load_state_dict(state_dict["model"])
    model.cuda()
    model.eval()
    target_layers = [model.resnet_layer]

    preprocess2tensor = [
        transforms.Resize(int(args.input_size / args.crop_pct), interpolation=args.interpolation),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    methods = {"gradcam": GradCAM,
               # "hirescam": HiResCAM,
               # "scorecam": ScoreCAM,
               # "gradcam++": GradCAMPlusPlus,
               # "ablationcam": AblationCAM,
               # "xgradcam": XGradCAM,
               # "eigencam": EigenCAM,
               # "eigengradcam": EigenGradCAM,
               # "layercam": LayerCAM,
               # "fullgrad": FullGrad,
               # "gradcamelementwise": GradCAMElementWise
               }

    targets = args.target_class

    cam_algorithm = methods["gradcam"]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=True) as cam:

        cam.batch_size = 1
        raw_image = Image.open(input_image).convert('RGB')
        img = preprocess2tensor[1](preprocess2tensor[0](raw_image))
        # img.save(save_path + '/%s_org.jpg' % filename)
        img = np.array(img)
        distort_img = corrupt(img, corruption_name="defocus_blur", severity=3)
        img = Image.fromarray(distort_img)
        # img.save(save_path + '/%s_%s.jpg' % (filename.split(".")[0], "blur"))

        input_tensor = preprocess2tensor[3](preprocess2tensor[2](img)).unsqueeze(0)

        np_img = np.array(img) / 255.

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(np_img, grayscale_cam, use_rgb=True, )

        Image.fromarray(cam_image).save(save_path + '/%s_%s.jpg' % (filename.split(".")[0], args.method))


if __name__ == "__main__":
    args = parse()
    img_files = os.listdir(args.read_path)
    for img in img_files:
        main(args, img, args.read_path, args.save_path)
        print(img, " ok!")
