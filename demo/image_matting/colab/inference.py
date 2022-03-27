import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.models.modnet import MODNet

background_color = (0, 255, 255)


def np2image(img):
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def generate_image(img, mask):
    background = np.zeros_like(img)
    background[:, :] = background_color
    alpha = mask[:, :, np.newaxis]
    foreground = alpha * img.astype(float)
    background = (1.0 - alpha) * background
    result = foreground + background
    result = np2image(result)
    return result


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default="./images/a.jpeg", help='path of input images')
    parser.add_argument('--output-path', type=str, default="./", help='path of output images')
    parser.add_argument('--ckpt-path', type=str, default="./pretrained/modnet_photographic_portrait_matting.ckpt",
                        help='path of pre-trained MODNet')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()
    if not os.path.exists(args.ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()

    # define hyper-parameters
    ref_size = 384

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(args.ckpt_path)
    else:
        weights = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

    # inference images
    input_path = args.input_path
    im_name = os.path.basename(input_path)

    print('Process image: {0}'.format(im_name))
    # read image
    im = cv2.imread(input_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    orig_im = im
    # unify image channels to 3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    print("开始测速>>>")
    # inference
    start_time = time.time()
    for i in range(100):
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)
        print(i)
    print("100次推理pytorch耗时", (time.time() - start_time) / 100)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    result = generate_image(orig_im, matte)
    matte_name = im_name.split('.')[0] + '.png'
    cv2.imwrite(os.path.join(args.output_path, matte_name), result)
