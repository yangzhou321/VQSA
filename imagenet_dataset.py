"""
get data loaders
"""
from __future__ import print_function

import os
import random
import socket
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from imagecorruptions import corrupt
import PIL.Image as Image

AlexNet_Error = [88.60, 89.40, 92.30, 82.00, 82.60, 78.60, 79.80, 86.70, 82.70, 81.90, 56.50, 85.30, 64.60, 71.80,
                 60.70, 84.50, 78.70, 71.80, 65.80]


class ImageNet_DataSet(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, crp_idx=None, sev_lvl=None, is_pair=False):
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.crp_idx = crp_idx
        self.sev_lvl = sev_lvl
        if self.crp_idx is None and self.sev_lvl is None:
            self.is_train = True
        else:
            self.is_train = False
        self.is_pair = is_pair
        if self.is_pair:
            self.crp_sev_sample_dict = {i: list(range(1, 6)) for i in range(0, 15)}
        else:
            self.crp_sev_sample_dict = {i: list(range(1, 6)) for i in range(-1, 15)}

    def get_CrpNameAndSevLevel(self):
        # if len(self.crp_sev_sample_dict.keys()) == 0:
        #     self.crp_sev_sample_dict = {i: list(range(1, 6)) for i in range(15)}
        id = random.randint(0, len(self.crp_sev_sample_dict.keys()) - 1)
        key = list(self.crp_sev_sample_dict.keys())[id]
        crp_func = key
        id = random.randint(0, len(self.crp_sev_sample_dict[key]) - 1)
        sev = self.crp_sev_sample_dict[key][id]
        self.crp_sev_sample_dict[key].pop(id)
        if len(self.crp_sev_sample_dict[key]) == 0:
            self.crp_sev_sample_dict.pop(key)
        if len(self.crp_sev_sample_dict.keys()) == 0:
            if self.is_pair:
                self.crp_sev_sample_dict = {i: list(range(1, 6)) for i in range(0, 15)}
            else:
                self.crp_sev_sample_dict = {i: list(range(1, 6)) for i in range(-1, 15)}
            # print("_______________________________________________________________________________")
        # print(self.crp_sev_sample_dict)
        # print(crp_func, sev)
        return crp_func, sev

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is None:
            raise ValueError("Datasets transform is need!")

        img = img.convert("RGB")
        img = self.transform[1](self.transform[0](img))

        if self.is_train:
            crp_idx, sev_lvl = self.get_CrpNameAndSevLevel()
        else:
            crp_idx, sev_lvl = self.crp_idx, self.sev_lvl
        if crp_idx == -1:
            img = self.transform[3](self.transform[2](img))
            return img, target, index  # clean
        img = np.array(img)
        distort_img = corrupt(img, corruption_number=crp_idx, severity=sev_lvl)

        img = Image.fromarray(img)
        img = self.transform[3](self.transform[2](img))
        distort_img = Image.fromarray(distort_img)
        distort_img = self.transform[3](self.transform[2](distort_img))
        if self.is_pair:
            img = torch.cat([img, distort_img], dim=0)
            return img, target, index  # train
        else:
            return distort_img, target, index  # test corrupt


class ImgNet_C_val_Dst(Dataset):
    def __init__(self, clean_img_path, img_path, ann_path, crp_name, severity):
        super(ImgNet_C_val_Dst, self).__init__()
        self.clean_img_path = clean_img_path
        self.img_path = img_path
        self.ann_path = ann_path
        self.crp_name = crp_name
        self.severity = severity
        self.image_ann_list = []
        with open(ann_path, 'r') as f:
            for i in f.readlines():
                self.image_ann_list.append(i)
        f.close()
        # print(self.image_ann_list[0])
        self.tsfrm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img_filename = self.image_ann_list[index].split(" ")[0]
        ann = int(self.image_ann_list[index].split(" ")[1])
        if self.crp_name == "clean":
            img = Image.open(os.path.join(self.clean_img_path, img_filename))
        else:
            img = Image.open(os.path.join(self.img_path, self.crp_name, self.severity, img_filename))
        img = img.convert("RGB")
        # img = cv2.imread(os.path.join(self.img_path, img_filename))
        # print(img)
        img = self.tsfrm(img)
        return img, ann, self.crp_name

    def __len__(self):
        return len(self.image_ann_list)


if __name__ == "__main__":
    root_path = "/home/yangzhou/datasets/imagenet/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_folder = os.path.join(root_path, 'train')
    test_folder = os.path.join(root_path, 'val')
    dataset = ImageNet_DataSet(train_folder, train_transform, is_pair=True)
    train_dataloader = DataLoader(dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=16,
                                  pin_memory=True)
    for i, (img, target) in enumerate(train_dataloader):
        print(img.size())
        raise ValueError("Break! ")
