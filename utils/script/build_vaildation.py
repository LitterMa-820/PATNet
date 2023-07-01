import os
import random
import shutil

if __name__ == '__main__':
    from eval.score_config import *

    validation_path = '../datasets/RGBD/NJUD_NLPR_DUT_2985/validation/'
    test_datasets = {'DUT-RGBD': dutrgbd, 'NJU2K': njud, 'NJUD': njud, 'NLPR': nlpr, 'STERE': stere, 'SIP': sip,
                     'DES': rgbd135, 'RGBD135': rgbd135, 'SSD': ssd, 'LFSD': lfsd}
    for name, root in test_datasets.items():
        print(name,root)
        destination_rgb = validation_path + name+'/RGB/'
        destination_depth = validation_path + name+'/depth/'
        destination_GT = validation_path + name+'/GT/'
        os.makedirs(destination_rgb)
        os.makedirs(destination_depth)
        os.makedirs(destination_GT)
        images = [f.split('.')[0] for f in os.listdir(root+'RGB/') if
                  f.endswith('.jpg') or f.endswith('png') or f.endswith('bmp')]
        random.shuffle(images)
        path_rgb_jpg = [root + 'RGB/' + f + '.jpg' for f in images]
        path_rgb_png = [root + 'RGB/' + f + '.png' for f in images]
        path_depth_jpg = [root + 'depth/' + f + '.jpg' for f in images]
        path_depth_png = [root + 'depth/' + f + '.png' for f in images]
        path_gt_jpg = [root + 'GT/' + f + '.jpg' for f in images]
        path_gt_png = [root + 'GT/' + f + '.png' for f in images]

        # print(images)
        for i in range(20):
            if os.path.exists(path_rgb_jpg[i]):
                rgb_path = path_rgb_jpg[i]
            else:
                rgb_path = path_rgb_png[i]
            shutil.copy(rgb_path, destination_rgb)

            if os.path.exists(path_depth_jpg[i]):
                depth_path = path_depth_jpg[i]
            else:
                depth_path = path_depth_png[i]
            shutil.copy(depth_path, destination_depth)

            if os.path.exists(path_gt_jpg[i]):
                gt_path = path_gt_jpg[i]
            else:
                gt_path = path_gt_png[i]
            shutil.copy(gt_path, destination_GT)

