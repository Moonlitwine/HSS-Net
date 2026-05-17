import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import PIL
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data
import random

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.dataset = self.config.data.dataset
        self.dataset_dir = self.config.data.dataset_dir
        self.train_filelist = self.config.data.train_filelist
        self.val_filelist = self.config.data.val_filelist
        self.batch_size = self.config.sampling.batch_size
        self.patch_size = self.config.backbone.image_size
        self.data_augment = self.config.data_augment.augment
        self.flip_prob = self.config.data_augment.flip_prob
        self.scale_prob = self.config.data_augment.scale_prob
        self.around_padding = self.config.data_augment.around_padding

    def get_loaders(self, is_train=True):
        if is_train:
            assert (os.path.exists(os.path.join(self.dataset_dir, 'train')))
            assert (os.path.exists(os.path.join(self.dataset_dir, 'val')))
            train_dataset = NewDataset(dataset=self.dataset,
                                       dir=os.path.join(self.dataset_dir, 'train'),
                                       transforms=self.transforms,
                                       patch_size=self.patch_size,
                                       filelist=self.train_filelist,
                                       data_augment=self.data_augment,
                                       flip_prob=self.flip_prob,
                                       scale_prob=self.scale_prob,
                                       around_padding=self.around_padding,
                                       is_train=True
                                       )
            val_dataset = NewDataset(dataset=self.dataset,
                                     dir=os.path.join(self.dataset_dir, 'val'),
                                     transforms=self.transforms
                                     )

            print(f"train dataset: {train_dataset.__len__()}")
            print(f"val dataset: {val_dataset.__len__()}")

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                       shuffle=True, num_workers=self.config.data.num_workers,
                                                       pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                     shuffle=False, num_workers=self.config.data.num_workers,
                                                     pin_memory=True)

            return train_loader, val_loader
        else:
            assert (os.path.exists(os.path.join(self.dataset_dir, 'sampling')))
            test_dataset = NewDataset(dataset=self.dataset,
                                      dir=os.path.join(self.dataset_dir, 'sampling'),
                                      transforms=self.transforms
                                      )

            print(f"sampling dataset: {test_dataset.__len__()}")

            deraining_loader = torch.utils.data.DataLoader(test_dataset,
                                                           batch_size=self.config.sampling.batch_size,
                                                           shuffle=False, num_workers=self.config.data.num_workers,
                                                           pin_memory=True)

            return deraining_loader

class NewDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, dir, transforms, patch_size=128, filelist=None, data_augment=False,
                 flip_prob=0.0, scale_prob=0.0, around_padding=False, is_train=False):
        super().__init__()

        self.dataset = dataset
        self.is_train = is_train

        image_dir = dir
        input_names, gt_names = [], []
        aug_mode = []
        img_ids = []

        image_inputs = os.path.join(image_dir, 'input')
        image_gts = os.path.join(image_dir, 'gt')

        if filelist == "" or filelist is None:
            if dataset == "Allweather":
                for f in listdir(image_inputs):
                    if isfile(os.path.join(image_inputs, f)):
                        input_names.append(f)
                        gt_names.append(f)
                        img_ids.append(f)
            elif dataset == "Outdoor_Rain":
                for f in sorted(listdir(image_inputs)):
                    if isfile(os.path.join(image_inputs, f)):
                        filename = os.path.splitext(f)
                        ext = filename[1]
                        filename = filename[0]
                        input_names.append(f)
                        gt_names.append(filename[:7] + ext)
                        img_ids.append(f)
            elif dataset == "RainDrop":
                for f in sorted(listdir(image_inputs)):
                    if isfile(os.path.join(image_inputs, f)):
                        filename = os.path.splitext(f)
                        ext = filename[1]
                        filename = filename[0]
                        input_names.append(f)
                        gt_names.append(filename[:-5] + "_clean" + ext)
                        img_ids.append(f)
            else:
                for f in sorted(listdir(image_inputs)):
                    if isfile(os.path.join(image_inputs, f)):
                        input_names.append(f)
                        gt_names.append(f)
                        img_ids.append(f)
        else:
            if dataset == "RealSetting":
                print(f"image_list:{filelist}\n")
                with open(filelist) as f:
                    contents = f.readlines()
                    for i in contents:
                        img_name = i.strip('\n').split(' ')
                        input_names.append(img_name[0])
                        gt_names.append(img_name[1])
                        aug_mode.append(int(img_name[2]) % 4)
                        img_ids.append(img_name[0])
            else:
                raise NotImplementedError

        self.dir = dir
        self.image_inputs = image_inputs
        self.image_gts = image_gts
        self.input_names = input_names
        self.gt_names = gt_names
        self.aug_mode = aug_mode
        self.img_ids = img_ids
        self.transforms = transforms
        self.patch_size = patch_size
        self.data_augment = data_augment
        self.flip_prob = flip_prob
        self.scale_prob = scale_prob
        self.around_padding = around_padding

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = self.img_ids[index]
        input_img = PIL.Image.open(
            os.path.join(self.image_inputs, input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(
                os.path.join(self.image_gts, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.image_gts, gt_name)).convert(
                'RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        if self.is_train:
            if self.data_augment:
                scale_prob = random.random()
                if scale_prob < self.scale_prob:
                    input_img = input_img.resize((input_img.width // 2, input_img.height // 2), resample=Image.BILINEAR)
                    gt_img = gt_img.resize((gt_img.width // 2, gt_img.height // 2), resample=Image.BILINEAR)

            input_img = np.array(input_img)
            gt_img = np.array(gt_img)

            h, w, _ = gt_img.shape

            if self.around_padding:
                h_pad = self.patch_size - h % self.patch_size
                w_pad = self.patch_size - w % self.patch_size
            else:
                h_pad = max(0, self.patch_size - h)
                w_pad = max(0, self.patch_size - w)
            if h_pad != 0 or w_pad != 0:
                input_img = np.pad(input_img, ((0, h_pad), (0, w_pad), (0, 0)), mode="reflect")
                gt_img = np.pad(gt_img, ((0, h_pad), (0, w_pad), (0, 0)), mode="reflect")

            h, w, _ = gt_img.shape
            if h != self.patch_size or w != self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)

                input_img = input_img[top:top + self.patch_size, left:left + self.patch_size]
                gt_img = gt_img[top:top + self.patch_size, left:left + self.patch_size]

            if not self.aug_mode:
                augmode = random.randint(0, 3)
                input_img = self.data_augmentation(input_img, mode=augmode).copy()
                gt_img = self.data_augmentation(gt_img, mode=augmode).copy()
            elif self.aug_mode[index] != 0:
                input_img = self.data_augmentation(input_img, mode=self.aug_mode[index]).copy()
                gt_img = self.data_augmentation(gt_img, mode=self.aug_mode[index]).copy()

            if self.data_augment:
                flip_prob = random.random()
                if flip_prob < self.flip_prob:
                    input_img = np.fliplr(input_img).copy()
                    gt_img = np.fliplr(gt_img).copy()

        input_img = self.transforms(input_img)
        gt_img = self.transforms(gt_img)

        output = [input_img, gt_img]

        return torch.cat(output, dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

    def data_augmentation(self, image, mode):
        if mode == 0:
            return image
        elif mode == 1:
            return np.fliplr(image)
        elif mode == 2:
            return np.rot90(image, k=2)
        elif mode == 3:
            return np.rot90(np.fliplr(image), k=2)
        elif mode == 4:
            return np.rot90(image)
        elif mode == 5:
            return np.rot90(np.fliplr(image))
        elif mode == 6:
            return np.rot90(image, k=3)
        elif mode == 7:
            return np.rot90(np.fliplr(image), k=3)
        else:
            return image