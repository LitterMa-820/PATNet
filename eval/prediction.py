import argparse

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import sys

import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from score_config import *

sys.path.append('../models')

from utils.misc import check_mkdir
import ttach as tta
results_save_path += 'predictions/'
torch.manual_seed(118)
# torch.cuda.set_device(0)
ckpt_path = '../saved_models/FTN/'
parser = argparse.ArgumentParser()
parser.add_argument('--ss', type=str, help='snapshot')
parser.add_argument('--modal', default='rgbd', type=str, help='rgbd or rgbt')
parser.add_argument('--in_size', default=(384, 384), type=tuple, help='input size')
opt = parser.parse_args()
args = {
    'snapshot': opt.ss,
    'crf_refine': False,
    'save_results': True
}
# test modality setting
if opt.modal == 'rgbd':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    to_test = {'DUT-RGBD': dutrgbd, 'NJUD': njud, 'NLPR': nlpr, 'STERE': stere, 'SIP': sip, 'RGBD135': rgbd135,
               'SSD': ssd,
               'LFSD': lfsd}
    data_modal = 'depth'
    input_size = (384, 384)
elif opt.modal == 'rgbt':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.341, 0.360, 0.753], [0.208, 0.269, 0.241])

    ])
    to_test = {'VT-821': vt821, 'VT-1000': vt1000, 'VT-5000': vt5000}
    data_modal = 'T'
    input_size = (352, 352)
else:
    print('input --modal=rgbd or rgbt')
    exit()
modal_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

transforms = tta.Compose(
    [
        # tta.HorizontalFlip(),
        # tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
        tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)


def main():
    # t0 = time.time()
    print('--------options--------')
    print('snapshot: ', opt.ss)
    print('running modality ', opt.modal)
    print('input size ', opt.in_size)
    net = getNet()
    print('load snapshot \'%s\' for testing' % args['snapshot'])
    print('testing for ' + opt.modal)
    net.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot'] + '.pth')))
    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            print(name, root)
            root1 = os.path.join(root, data_modal)
            img_list = [os.path.splitext(f) for f in os.listdir(root1)]
            # print(img_list)
            for idx, img_name in enumerate(img_list):
                print('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                rgb_png_path = os.path.join(root, 'RGB', img_name[0] + '.png')
                rgb_jpg_path = os.path.join(root, 'RGB', img_name[0] + '.jpg')
                modal_jpg_path = os.path.join(root, data_modal, img_name[0] + '.jpg')
                modal_png_path = os.path.join(root, data_modal, img_name[0] + '.png')
                if os.path.exists(rgb_png_path):
                    img = Image.open(rgb_png_path).convert('RGB')
                else:
                    img = Image.open(rgb_jpg_path).convert('RGB')
                if os.path.exists(modal_jpg_path):
                    modality = Image.open(data_modal)
                else:
                    modality = Image.open(modal_png_path)

                if data_modal == 'depth':
                    modality = modality.convert('L')
                else:
                    modality = modality.convert('RGB')

                w_, h_ = img.size
                img_resize = img.resize(input_size, Image.BILINEAR)  # Foldconv cat是320
                modality_resize = modality.resize(input_size, Image.BILINEAR)  # Foldconv cat是320
                img_var = Variable(img_transform(img_resize).unsqueeze(0)).cuda()
                modality_var = Variable(modal_transform(modality_resize).unsqueeze(0)).cuda()
                n, c, h, w = img_var.size()
                if data_modal == 'depth':
                    modality = torch.cat((modality_var, modality_var, modality_var), 1)
                else:
                    modality = modality_var
                mask = []
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_var)
                    m_trans = transformer.augment_image(modality)

                    ##
                    out = infer(net, rgb_trans, m_trans)

                    deaug_mask = transformer.deaugment_mask(out)
                    mask.append(deaug_mask)

                prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction = prediction.sigmoid()
                prediction = to_pil(prediction.data.squeeze(0).cpu())
                prediction = prediction.resize((w_, h_), Image.BILINEAR)
                # if args['crf_refine']:
                #     prediction = crf_refine(np.array(img), np.array(prediction))
                if args['save_results']:
                    check_mkdir(os.path.join(results_save_path, args['snapshot'], name))
                    prediction.save(os.path.join(results_save_path, args['snapshot'], name, img_name[0] + '.png'))


if __name__ == '__main__':
    main()
