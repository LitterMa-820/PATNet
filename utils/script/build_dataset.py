import os.path
import shutil


def moveImage(image_root, destination):
    images = [image_root + f for f in os.listdir(image_root) if
              f.endswith('.jpg') or f.endswith('png') or f.endswith('bmp')]
    destinations = [destination + f for f in os.listdir(image_root) if
                    f.endswith('.jpg') or f.endswith('png') or f.endswith('bmp')]
    if not os.path.exists(destination):
        os.makedirs(destination)
    for image, des in zip(images, destinations):
        shutil.copy(image, des)

def image_collate():
    images1 =[ f for f in os.listdir('D:/计算机/数据集/MySODDataset/Training_dataset/NJUD_NLPR_DUT_2985/RGB') if f.endswith('.jpg') or f.endswith('png') or f.endswith('bmp')]
    images2 =[ f for f in os.listdir('D:/code/My_SOD_CODE/BaseLine/datasets/RGBD_SOD_Datasets/Training_dataset/NJUD_NLPR_DUT_downstream_task_SOD/RGB') if f.endswith('.jpg') or f.endswith('png') or f.endswith('bmp')]
    set1 = set(images1)
    set2 = set(images2)
    print(set1 ^ set2)
if __name__ == '__main__':
    # image_root = 'D:/计算机/数据集/MySODDataset/Training_dataset/NJUD_NLPR_2185/RGB/'
    # destination = 'D:/计算机/数据集/MySODDataset/Training_dataset/NJUD_NLPR_DUT_2985/RGB/'
    # moveImage(image_root,destination)
    image_collate()
