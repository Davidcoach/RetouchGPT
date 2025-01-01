import os
import json
import random
import glob
from torch.utils.data import Dataset
import cv2
from PIL import Image, ImageDraw
import numpy as np
import math
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import pandas as pd
import torchvision
import json
from random import choice

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
skin_concerns_data = {
    "dark circle": {
        "what_are_they": "Dark circles are a common skin condition where the skin underneath the eyes appears darker.",
        "causes": "They can be caused by thinning skin, visible blood vessels, pigmentation, genetics, fatigue, aging, and lifestyle.",
        "image_retouching": "Use healing brush or clone stamp tool to blend the dark area. Lighten with brightness or exposure adjustment layer.",
        "daily_prevention": "Get enough sleep, maintain a balanced diet, stay hydrated, reduce alcohol and caffeine, use moisturizer and sunscreen."
    },
    "forehead wrinkles": {
        "what_are_they": "Lines or creases on the forehead, often due to aging or repetitive facial expressions.",
        "causes": "Loss of skin elasticity, repetitive forehead muscle contractions, and sun exposure.",
        "image_retouching": "Use smoothing or healing tools to soften wrinkles. Lighten with dodge tool and apply selective blur for a natural look.",
        "daily_prevention": "Use sunscreen, moisturize, anti-aging products with retinol, wear sunglasses, manage stress."
    },
    "acne": {
        "what_are_they": "A skin condition with pimples, blackheads, inflamed skin, commonly on the face, neck, chest, and back.",
        "causes": "Clogged hair follicles from oil and dead skin cells, hormonal changes, medications, diet, stress.",
        "image_retouching": "Spot healing brush for pimples, patch tool for larger areas, adjust texture and hue.",
        "daily_prevention": "Wash face with gentle cleanser, avoid heavy makeup, balanced diet, stress management."
    },
    "freckle": {
        "what_are_they": "Small, flat, brown marks on the skin, more pronounced with sun exposure.",
        "causes": "Genetics and sun exposure increase melanin in skin cells, resulting in freckles.",
        "image_retouching": "Healing brush to blend, color correction for toning down.",
        "daily_prevention": "Sunscreen, protective clothing, vitamin C serum."
    },
    "fragmented hair": {
        "what_are_they": "Damaged, split, or broken hair from heat styling, chemical processing, or harsh treatments.",
        "causes": "Excessive heat styling, over-washing, chemical treatments, environmental factors.",
        "image_retouching": "Clone stamp or healing brush to replicate healthy hair, adjust sharpness and smoothness.",
        "daily_prevention": "Minimize heat styling, heat protectant products, condition regularly, trim ends, protect from sun."
    },
    "crow's feet": {
        "what_are_they": "Lines at the outer corners of the eyes, noticeable when smiling or squinting.",
        "causes": "Repetitive facial movements, aging, loss of collagen and elastin, sun exposure.",
        "image_retouching": "Smoothing or healing brush used subtly. Reduce visibility instead of removing completely for natural result.",
        "daily_prevention": "Sunglasses, sunscreen, eye cream with retinol or hyaluronic acid, sleep, hydration."
    },
    "spots": {
        "what_are_they": "Variety of skin issues like blemishes, age spots, or sunspots, typically darker than surrounding skin.",
        "causes": "Excess melanin production triggered by sun exposure, hormonal changes, aging, inflammation.",
        "image_retouching": "Spot healing brush for small areas, patch tool for larger spots, color correction.",
        "daily_prevention": "Sunscreen, protective clothing, skin care with vitamin C or retinol."
    },
    "fine lines": {
        "what_are_they": "Subtle creases in the skin, due to aging, found around the eyes and mouth.",
        "causes": "Natural aging, loss of skin elasticity and collagen, sun damage, dehydration, smoking.",
        "image_retouching": "Soft healing brush to gently smooth out, maintain natural texture.",
        "daily_prevention": "Consistent skin care routine with collagen-boosting products, sunscreen, hydration."
    },
    "pimple": {
        "what_are_they": "Small pustule or papule, lesions or inflammations from clogged or infected pores.",
        "causes": "Overproduction of oil, mixed with dead skin cells to block pores, bacteria growth leading to inflammation.",
        "image_retouching": "Spot healing brush for quick clearing, clone stamp for controlled edit.",
        "daily_prevention": "Clean skin, non-comedogenic products, balanced diet, stress management."
    }}

skin_label = ["dark circle", "forehead wrinkles", "acne", "freckle", "fragmented hair", "crow's feet", "spots",
              "fine lines", "pimple"]


class FaceDataset(Dataset):
    def __init__(self, path, resolution=512, data_type="train", data_percentage=1, return_mask=False):
        self.resolution = resolution
        self.data_type = data_type
        self.return_mask = return_mask
        self.imgs = glob.glob(os.path.join(path, data_type, 'source', '*.*'))
        self.imgs_r = glob.glob(os.path.join(path, data_type, 'target', '*.*'))
        if self.return_mask:
            self.masks = glob.glob(os.path.join(path, data_type, 'mask', '*.*'))
        self.imgs = sorted(self.imgs, key=lambda x: os.path.basename(x))
        self.imgs_r = sorted(self.imgs_r, key=lambda x: os.path.basename(x))
        if self.return_mask:
            self.masks = sorted(self.masks, key=lambda x: os.path.basename(x))
        self.parquet = \
        pd.read_parquet(r'/share/home/HCI/dingchun/RetouchGPT/train-00000-of-00001.parquet', engine='pyarrow')["text"]
        assert len(self.imgs) == len(self.imgs_r), "Can not match the FFHQ and FFHQR!"
        for i, j in zip(self.imgs, self.imgs_r):
            assert os.path.basename(i) == os.path.basename(j)
        self.length = len(self.imgs)
        if data_type == 'train':
            self.length = int(self.length * data_percentage)
            self.imgs = self.imgs[:self.length]
            self.imgs_r = self.imgs_r[:self.length]
            if self.return_mask:
                self.masks = self.masks[:self.length]
        print(f"Data number: {len(self.imgs)}")
        self.positions = ["top left", "top", "top right", "left", "center", "right", "bottom left", "bottom",
                          "bottom right"]
        self.toTensor = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor()
        ])

        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __len__(self):
        return self.length

    def sigmoid(self, x, fac=50):
        return 1 / (1 + torch.exp(-fac * x))

    def find_max(self, lst):
        max_value = max(lst)
        index = lst.index(max_value)
        return index

    def find_class(self, prompt):
        keywords = ["man", "woman", "boy", "girl"]
        key = "human"
        for keyword in keywords:
            if keyword in prompt:
                key = keyword
        return key

    def crop(self, diff):
        img_list = []
        for i in range(3):
            for j in range(3):
                sub_img = torchvision.transforms.functional.crop(diff, 75 * i, 75 * j, 75, 75)
                img_list.append(sub_img)
        value_list = [torch.sum(i) for i in img_list]
        index = self.find_max(value_list)
        position = self.positions[index]
        return position

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img_r = Image.open(self.imgs_r[index]).convert("RGB")
        if self.return_mask:
            mask = Image.open(self.masks[index]).convert("L")

        img, img_r = self.toTensor(img), self.toTensor(img_r)
        img, img_r = self.normalize(img), self.normalize(img_r)
        if self.return_mask:
            mask = self.toTensor(mask)

        if self.data_type == 'train':
            flip = random.random()
            if flip < 0.5:
                img, img_r = TF.hflip(img), TF.hflip(img_r)
                if self.return_mask:
                    mask = TF.hflip(mask)
        diff = torch.abs(img - img_r).mean(dim=0, keepdim=True)
        if self.return_mask:
            diff = diff * mask
        diff_normed = (diff - diff.amin(dim=[1, 2], keepdim=True)) / (
                    diff.amax(dim=[1, 2], keepdim=True) - diff.amin(dim=[1, 2], keepdim=True))
        diff_normed = self.sigmoid(diff_normed)
        diff_normed[diff_normed > 0.7], diff_normed[diff_normed <= 0.7] = 1, 0
        position = self.crop(diff_normed)
        # save_image(diff, "diff.png")
        # name = img_r_path[-7:-1]
        # save_image(diff_normed, f"diff_normed_{name}.png")
        # save_image(mask, f"mask_{name}.png")
        if self.data_type == "test":
            index += 62999
        prompt = self.parquet[index]

        conversation_normal = []
        conversation_normal.append(
            {"from": "human", "value": "Is there any imperfections in the image?"})
        conversation_normal.append({"from": "gpt", "value": "No, there is no imperfections in the image."})

        conversation_abnormal = []
        conversation_abnormal.append(
            {"from": "human", "value": "Is there any imperfections in the image?"})

        abnormal_describe = f"Yes, there are imperfections in the image, they are at the {position} of the image."
        conversation_abnormal.append({"from": "gpt", "value": abnormal_describe})
        blemish = choice(
            ["dark circle", "forehead wrinkles", "acne", "freckle", "fragmented hair", "crow's feet", "spots",
             "fine lines", "pimple"])
        instruction_templates = [
            f"Please retouch the {position} area to remove {blemish}.",
            f"The {position} part of the face has {blemish}; kindly fix it.",
            f"Can you smooth out the {blemish} in the {position} section?",
            f"There's {blemish} in the {position} area, please correct it.",
            f"Improve the {position} region by removing the {blemish}."
        ]
        for i in range(5):
            conversation_abnormal.append(
                {"from": "human", "value": instruction_templates[i]})
            conversation_abnormal.append(
                {"from": "gpt", "value": "Sure, I will retouch the image to remove the imperfections."})
        if self.return_mask:
            return img, img_r, mask, [conversation_abnormal], [conversation_normal]
        else:
            return img, img_r, [conversation_abnormal], [conversation_normal]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class Testset(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.root = args.input_path
        ### input A (label maps)
        self.source_paths = sorted(make_dataset(self.root))
        self.size = args.size
        self.dataset_size = len(self.source_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.source_paths[index], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2, 0, 1).flip(0)
        img = img.to(torch.float32)
        img = (img / 255 - 0.5) / 0.5
        return os.path.basename(self.source_paths[index])[:-4], img

    def __len__(self):
        return len(self.source_paths)