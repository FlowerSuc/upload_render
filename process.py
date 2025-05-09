from io import BytesIO

import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from random import random, choice
from PIL import Image
import torchvision.transforms.functional as TF
from scipy.ndimage.filters import gaussian_filter

rz_dict = {
    'bilinear': TF.InterpolationMode.BILINEAR,
    'bicubic': TF.InterpolationMode.BICUBIC,
    'lanczos': TF.InterpolationMode.LANCZOS,
    'nearest': TF.InterpolationMode.NEAREST
}


def data_augment(img, opt):
    # 数据增强：高斯模糊，下采样，JPEG compression
    if random() < opt.down_prob:
        img = custom_resize(img, opt)
    img = np.array(img)
    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        img = gaussian_blur(img, sig)
    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def gaussian_blur(img, sigma):
    if len(img.shape) == 3:
        img_blur = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_blur[:, :, i] = gaussian_filter(img[:, :, i], sigma=sigma)
    else:
        img_blur = gaussian_filter(img, sigma=sigma)
    return img_blur


def sample_discrete(s):
    # 离散区间采样
    if len(s) == 1:
        return s[0]
    return choice(s)


def sample_continuous(s):
    # 连续区间采样
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2")


def custom_resize(img, opt):
    # 下采样
    r = sample_continuous(opt.down_ratio)
    interp = sample_discrete(opt.rz_interp)
    h, w = img.height, img.width

    down_size = (max(1, int(w * r)), max(1, int(h * r)))
    img = TF.resize(img, down_size, interpolation=rz_dict[interp])
    img = TF.resize(img, (w, h), interpolation=rz_dict[interp])

    return img


def ED(img):
    # 纹理多样性计算(texture diversity)
    # (C, H, W)
    r1, r2 = img[:, 0:-1, :], img[:, 1::, :]  # 垂直方向
    r3, r4 = img[:, :, 0:-1], img[:, :, 1::]  # 水平方向
    r5, r6 = img[:, 0:-1, 0:-1], img[:, 1::, 1::]  # 对角线方向↖
    r7, r8 = img[:, 0:-1, 1::], img[:, 1::, 0:-1]  # 对角线方向↘

    s1 = torch.sum(torch.abs(r1 - r2)).item()
    s2 = torch.sum(torch.abs(r3 - r4)).item()
    s3 = torch.sum(torch.abs(r5 - r6)).item()
    s4 = torch.sum(torch.abs(r7 - r8)).item()

    return s1 + s2 + s3 + s4


def processing_RPTC(img, opt):
    # 分块 打乱 重组 并且按照ED给出的纹理多样性给出rich, poor texture
    rate = opt.patch_num
    num_block = int(pow(2, rate))
    patchsize = int(opt.loadSize / num_block)  # 32x32
    randomcrop = transforms.RandomCrop(patchsize)

    minsize = min(img.size)
    if minsize < patchsize:
        img = transforms.Resize((patchsize, patchsize))(img)

    img = transforms.ToTensor()(img)
    # img = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)

    img_template = torch.zeros(3, opt.loadSize, opt.loadSize)  # 空白画布 往里面补充patch
    img_crops = []  # 存切割下来的图片以及其对应的纹理复杂度
    # 选取192块 取top33% bottom33%
    for i in range(num_block * num_block * 3):
        cropped_img = randomcrop(img)
        texture_rich = ED(cropped_img)
        img_crops.append([cropped_img, texture_rich])

    img_crops = sorted(img_crops, key=lambda x: x[1])  # 按照纹理复杂度从低到高排序

    count = 0
    # 取前33% 作为img_poor
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:, ii * patchsize:(ii + 1) * patchsize, jj * patchsize:(jj + 1) * patchsize] = \
                img_crops[count][0]
            count += 1
    img_poor = img_template.clone().unsqueeze(0)  # (1, 3, H, W)

    # 取后33% 作为img_rich
    count = -1
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:, ii * patchsize:(ii + 1) * patchsize, jj * patchsize:(jj + 1) * patchsize] = \
                img_crops[count][0]
            count -= 1
    img_rich = img_template.clone().unsqueeze(0)  # (1, 3, H, W)

    img = torch.cat((img_poor, img_rich), 0)  # 0维度拼接->(2, 3, H, W)

    return img
