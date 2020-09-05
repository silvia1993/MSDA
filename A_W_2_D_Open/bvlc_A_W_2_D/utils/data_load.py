import torch
import torch.utils.data as data

from PIL import Image
import os
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = data[0]
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images


class OfficeImage(data.Dataset):
    def __init__(self, label, split="train", transform=None):
        imgs = make_dataset(label)
        self.label = label
        self.split = split
        self.imgs = imgs
        self.transform = transform
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
 
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert("RGB")
        
        img = img.resize((256, 256), Image.BILINEAR)

        if self.split == "train":
            w, h = img.size
            tw, th = (227, 227)
            x1 = np.random.randint(0, w - tw)
            y1 = np.random.randint(0, h - th)
            img = img.crop((x1, y1, x1 + tw, y1 + th))
        if self.split == "test":
            img = img.crop((15, 15, 242, 242))

        img = np.array(img, dtype=np.float32)
        img = img[:, :, ::-1]
        img = img - self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
            
        return img, target

    def __len__(self):
        return len(self.imgs)
