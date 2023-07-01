import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# several data augmentation strategies
def cv_random_flip(img, label, thermal):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        thermal = thermal.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, thermal


def randomCrop(image, label, thermal):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), thermal.crop(random_region)


def randomRotation(image, label, thermal):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        thermal = thermal.rotate(random_angle, mode)
    return image, label, thermal


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, Thermal_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.thermals = [Thermal_root + f for f in os.listdir(Thermal_root) if f.endswith('.bmp')
                         or f.endswith('.png') or f.endswith('.jpg')]  # different from rgbd
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.thermals = sorted(self.thermals)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.551, 0.619, 0.532], [0.241, 0.236, 0.244])])  # different from rgbd
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.thermals_transform = transforms.Compose(  # different from rgbd
            [transforms.Resize((self.trainsize, self.trainsize)),
             transforms.ToTensor(),
             transforms.Normalize([0.341, 0.360, 0.753], [0.208, 0.269, 0.241])])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        thermal = self.rgb_loader(self.thermals[index])
        image, gt, thermal = cv_random_flip(image, gt, thermal)
        image, gt, thermal = randomCrop(image, gt, thermal)
        image, gt, thermal = randomRotation(image, gt, thermal)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        # gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        thermal = self.thermals_transform(thermal)

        return image, gt, thermal

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.thermals)
        images = []
        gts = []
        thermals = []
        for img_path, gt_path, thermal_path in zip(self.images, self.gts, self.thermals):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            thermal = Image.open(thermal_path)
            if img.size == gt.size and gt.size == thermal.size:
                images.append(img_path)
                gts.append(gt_path)
                thermals.append(thermal_path)
        self.images = images
        self.gts = gts
        self.thermals = thermals

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, thermal):
        assert img.size == gt.size and gt.size == thermal.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), thermal.resize((w, h),
                                                                                                        Image.NEAREST)
        else:
            return img, gt, thermal

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, thermal_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):
    dataset = SalObjDataset(image_root, gt_root, thermal_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader
