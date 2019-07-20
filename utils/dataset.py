import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Yolov2Dataset(Dataset):
    def __init__(self, options, training, multiscale=True):
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

        self.batch_size = options.batch_size
        self.batch_count = 0
        self.img_size = 13 * 32
        self.training = training
        self.multiscale = multiscale
        if multiscale:
            self.multiscale_interval = 10
            self.min_scale = 10 * 32
            self.max_scale = 19 * 32
        self.get_item_choice = 1 # 0 for erik, 1 for marvis


    def __getitem__(self, index):
        get_item_ways = [self.get_item_erik, self.get_item_marvis]
        return get_item_ways[self.get_item_choice](index)

    def get_item_marvis(self, index):
        """
        marvis' way to get item: resize directly(by PIL.Image) in collate_fn
        """
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img)

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes

        return img, targets, img_path

    def get_item_erik(self, index):
        """
        erik's way to get item: pad to square, then resize it in collate_fn
        """
        #  ----- Image -----
        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        # img = transforms.ToTensor()(Image.open(img_path))
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, img_height, img_width = img.shape
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ----- Label -----
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        targets = None
        assert os.path.exists(label_path), "label_path not exist:" + label_path
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = img_width * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = img_height * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = img_width * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = img_height * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= img_width / padded_w
            boxes[:, 4] *= img_height / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        return img, targets, img_path

    def collate_fn(self, batch):
        imgs, targets, img_paths = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.training and self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_scale, self.max_scale + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([self.resize(img, img_path, self.img_size) for (img, img_path) in zip(imgs, img_paths)])

        self.batch_count += 1
        return imgs, targets, img_paths

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

    def get_dataloader(self):
        return DataLoader(
                self,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=self.collate_fn
            )

    def resize(self, img, img_path, size):
        if self.get_item_choice == 0:
            img = F.interpolate(img.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        else:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            img = transforms.ToTensor()(img)
        return img


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



