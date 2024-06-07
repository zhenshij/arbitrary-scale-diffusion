import os
import json
import random
from PIL import Image
from abc import abstractmethod

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms


class DatasetBase(Dataset):
    def __init__(self,
                 data_root,
                 txt_file=None,
                 size=None,
                 lr_size=None,
                 interpolation="bicubic",
                 first_k=None
                 ):
        self.data_root = data_root
        if txt_file is not None:
            with open(txt_file, "r") as f:
                self.image_paths = f.read().splitlines()
        else:
            self.image_paths = sorted(os.listdir(data_root))
        if first_k is not None:
            self.image_paths = self.image_paths[:first_k]
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.lr_size = lr_size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image_path = os.path.join(self.data_root, self.image_paths[i])
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)

        data = {}
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image_hr = np.array(image).astype(np.uint8)
        data["image_hr"] = (image_hr / 127.5 - 1.0).astype(np.float32)

        if self.lr_size is not None:
            lr_image = image.resize((self.lr_size, self.lr_size), resample=self.interpolation)
            lr_image = np.array(lr_image).astype(np.uint8)
            data["image_lr"] = (lr_image / 127.5 - 1.0).astype(np.float32)

        return data


class FFHQ(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="../datasets/ffhq_1024", **kwargs)


class FFHQTrain(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/ffhq_train.txt", data_root="../datasets/ffhq_1024", **kwargs)


class FFHQValidation(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/ffhq_val.txt", data_root="../datasets/ffhq_1024", **kwargs)


class CelebAHQ(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="../datasets/celeba_hq_256", **kwargs)


class CelebAHQTrain(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/celebahq_train.txt", data_root="../datasets/celeba_hq_256", **kwargs)


class CelebAHQValidation(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/celebahq_val.txt", data_root="../datasets/celeba_hq_256", **kwargs)


class LSUNBedroomTrain(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/bedrooms_train.txt", data_root="../datasets/lsun_bedroom/train", **kwargs)


class LSUNBedroomValidation(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/bedrooms_val.txt", data_root="../datasets/lsun_bedroom/train", **kwargs)


class LSUNChurchTrain(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/church_outdoor_train.txt", data_root="../datasets/lsun_church/train", **kwargs)


class LSUNChurchValidation(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/church_outdoor_val.txt", data_root="../datasets/lsun_church/train", **kwargs)


class LSUNTowerTrain(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/tower_train.txt", data_root="../datasets/lsun_tower/train", **kwargs)


class LSUNTowerValidation(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/tower_val.txt", data_root="../datasets/lsun_tower/train", **kwargs)

class DIV2KBase(DatasetBase):
    def __init__(self, data_root, repeat=1, augment=False, **kwargs):
        super().__init__(data_root=data_root, **kwargs)
        self.repeat = repeat
        self.augment = augment

    def __getitem__(self, i):
        image_path = os.path.join(self.data_root, self.image_paths[i % self._length])
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        x0 = random.randint(0, img.shape[0] - self.size)
        y0 = random.randint(0, img.shape[1] - self.size)
        img = img[x0: x0 + self.size, y0: y0 + self.size]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = np.flip(x, axis=0)
                if vflip:
                    x = np.flip(x, axis=1)
                if dflip:
                    x = np.transpose(x, (1, 0, 2))
                return x

            img = augment(img)

        image = Image.fromarray(img)

        data = {}

        image_hr = np.array(image).astype(np.uint8)
        data["image_hr"] = (image_hr / 127.5 - 1.0).astype(np.float32)

        if self.lr_size is not None:
            lr_image = image.resize((self.lr_size, self.lr_size), resample=self.interpolation)
            lr_image = np.array(lr_image).astype(np.uint8)
            data["image_lr"] = (lr_image / 127.5 - 1.0).astype(np.float32)

        return data

    def __len__(self):
        return self._length * self.repeat

class DIV2KTrain(DIV2KBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="../datasets/DIV2K_train_HR", repeat=20, augment=True, **kwargs)

class DIV2KValidation(DIV2KBase):
    def __init__(self, **kwargs):
        # super().__init__(data_root="../datasets/DIV2K_valid_HR", repeat=160, **kwargs)
        super().__init__(data_root="../datasets/DIV2K_valid_HR", **kwargs)

class DF2KTrain(DIV2KBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="../datasets/DF2K_train_HR", repeat=4, augment=True, **kwargs)
