import os

import torch
import torchvision.transforms as transforms
from PIL import Image



class valid_dataset:
    def __init__(self, image_root, testsize):
        rgb_root = image_root + 'RGB/'
        depth_root = image_root + 'depth/'
        gt_root = image_root + 'GT/'
        # print(rgb_root)
        # print(depth_root)
        # print(gt_root)
        self.testsize = testsize
        self.images = [rgb_root + f for f in os.listdir(rgb_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        # print(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        depth = torch.cat((depth, depth, depth), 1)
        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]
        # image_for_post=self.rgb_loader(self.images[self.index])
        # image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, depth, gt, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# if __name__ == '__main__':
#
#     to_pil = transforms.ToPILImage()
#     dutrgbd = '../datasets/RGBD/NJUD_NLPR_DUT_2985/validation/DUT-RGBD/'
#     njud = '../datasets/RGBD/NJUD_NLPR_DUT_2985/validation/NJUD/'
#     nlpr = '../datasets/RGBD/NJUD_NLPR_DUT_2985/validation/NLPR/'
#     stere = '../datasets/RGBD/NJUD_NLPR_DUT_2985/validation/STERE/'
#     sip = '../datasets/RGBD/NJUD_NLPR_DUT_2985/validation/SIP/'
#     rgbd135 = '../datasets/RGBD/NJUD_NLPR_DUT_2985/validation/RGBD135/'
#     ssd = '../datasets/RGBD/NJUD_NLPR_DUT_2985/validation/SSD/'
#     lfsd = '../datasets/RGBD/NJUD_NLPR_DUT_2985/validation/LFSD/'
#
#     to_test = {'DUT-RGBD': dutrgbd, 'NJUD': njud, 'NLPR': nlpr, 'STERE': stere, 'SIP': sip, 'RGBD135': rgbd135,
#                'SSD': ssd,
#                'LFSD': lfsd}
#     model = CMP2T()
#     model.load_state_dict(torch.load('../saved_models/CMP2T/CMP2TV6_B3_l1e5_200_gen.pth'))
#     model.cuda()
#     model.eval()
#     with torch.no_grad():
#         for name, root in to_test.items():
#             test_loader = valid_dataset(root, 384)
#             mae = cal_mae()
#             for i in tqdm(range(test_loader.size)):
#                 image, depth, gt, HH, WW, image_name = test_loader.load_data()
#                 image = image.cuda()
#                 depth = depth.cuda()
#                 # depth = torch.cat((depth, depth, depth), 1)
#                 side_out4, side_out3, side_out2, side_out1 = model(image,depth)
#                 sal_map = side_out1
#                 sal_map = sal_map.sigmoid()
#                 sal_map = to_pil(sal_map.data.squeeze(0).cpu())
#                 sal_map = sal_map.resize((HH, WW), Image.BILINEAR)
#                 # print(gt.shape,sal_map.shape)
#                 if sal_map.size != gt.size:
#                     x, y = gt.size
#                     sal = sal_map.resize((x, y))
#                 gt = np.asarray(gt, np.float32)
#                 gt /= (gt.max() + 1e-8)
#                 gt[gt > 0.5] = 1
#                 gt[gt != 1] = 0
#                 res = sal_map
#                 res = np.array(res)
#                 if res.max() == res.min():
#                     res = res / 255
#                 else:
#                     res = (res - res.min()) / (res.max() - res.min())
#                 # 二值化会提升mae和meanf,em
#                 # res[res > 0.5] = 1
#                 # res[res != 1] = 0
#                 mae.update(res, gt)
#             MAE = mae.show()
#             print(name+': ',MAE)
#
