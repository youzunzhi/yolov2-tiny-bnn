import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset


class Yolov2Dataset(Dataset):
    def __init__(self, options, training=False, multiscale=False):
        self.data_cfg = self.parse_data_cfg(options.data_cfg)
        if training:
            path = self.data_cfg["train"]
        else:
            path = self.data_cfg["valid"]
        with open(path, "r") as file:
            self.img_files = file.readlines()
        if self.data_cfg['names'].find('voc') != -1:
            self.label_files = [
                path.replace("JPEGImages", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                for path in self.img_files
            ]
        else:  # suppose it's COCO
            self.label_files = [
                path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                for path in self.img_files
            ]

        self.img_size = 416
        self.normalized_labels = True
        self.augment = training
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        #  ----- Image -----
        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        # img = transforms.ToTensor()(Image.open(img_path))
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ----- Label -----
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        # import pydevd_pycharm
        # pydevd_pycharm.settrace('172.26.3.54', port=12344, stdoutToServer=True, stderrToServer=True)
        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horizontal_flip(img, targets)

        return img, targets

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets

    def __len__(self):
        return len(self.img_files)

    def parse_data_cfg(self, path):
        """Parses the data configuration file"""
        options = dict()
        with open(path, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            key, value = line.split('=')
            options[key.strip()] = value.strip()
        return options


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def horizontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
