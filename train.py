import argparse
import os
import random

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
import tensorboardX
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda import amp
from data_loader.rgb_d_loader import get_loader
from models.p3Net import p3Net
from utils.train_utils import adjust_lr, AvgMeter, save_checkpoint
from validation import validation
from torchvision.utils import make_grid
from config import *

os.environ['TORCH_HOME'] = '../saved_models'
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=3, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--save_opt', action='store_true', help='save optimizer? default is false')
random.seed(118)
np.random.seed(118)
torch.manual_seed(118)
torch.cuda.manual_seed(118)
torch.cuda.manual_seed_all(118)
opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = p3Net('./saved_models/p2t_base.pth')
generator.cuda()

generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)
save_path = './saved_models/p3Net'
model_name = 'p3Net'
min_mae = 1
validation_step = 0
train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

size_rates = [1]  # multi-scale training
use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def get_loss(side_out1, side_out2, side_out3, side_out4, target1, target2, target3, target4):
    loss1 = structure_loss(side_out1, target1)
    loss2 = structure_loss(side_out2, target2)
    loss3 = structure_loss(side_out3, target3)
    loss4 = structure_loss(side_out4, target4)
    # sml = get_saliency_smoothness(torch.sigmoid(side_out1), mask1)
    # 1 ，0.9，0.8，0.7 0.8
    return loss1, loss2, loss3, loss4


if not os.path.exists(save_path):
    os.makedirs(save_path)

sw = tensorboardX.SummaryWriter('./results/tensorboard_log/' + model_name + '_log')
global_step = 0
for epoch in range(1, opt.epoch + 1):
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        generator_optimizer.zero_grad()
        images, gts, depths = pack
        images = Variable(images)
        gts = Variable(gts)
        depths = Variable(depths)
        images = images.cuda()
        gts = gts.cuda()
        depths = depths.cuda()
        b, c, h, w = gts.size()
        target_1 = F.interpolate(gts, size=h // 8, mode='nearest')
        target_2 = F.interpolate(gts, size=h // 16, mode='nearest')
        target_3 = F.interpolate(gts, size=h // 32, mode='nearest')

        with amp.autocast(enabled=use_fp16):
            depth_3 = torch.cat((depths, depths, depths), 1)
            side_out4, side_out3, side_out2, side_out1 = generator(images, depth_3)  # hed
            loss1, loss2, loss3, loss4 = get_loss(side_out1, side_out2, side_out3, side_out4, gts, target_1, target_2, target_3)
            loss = loss1 + loss2 + loss3 + loss4
        generator_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(generator_optimizer)
        scaler.update()
        loss_record.update(loss.data, opt.batchsize)
        global_step += 1
        sw.add_scalar('lr', generator_optimizer.param_groups[0]['lr'], global_step=global_step)
        sw.add_scalars('loss',
                       {'loss_sum': loss.item(), 'loss1': loss1.item(), 'loss2': loss2.item(), 'loss3': loss3.item(),
                        'loss4': loss4.item()},
                       global_step=global_step)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))
        if i % 100 == 0 or i == total_step or i == 1:
            res = side_out1[0].clone()
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = torch.tensor(res).unsqueeze(dim=0)
            show_gt = gts[0].clone().cpu().data
            show_gt = torch.cat((show_gt, show_gt, show_gt), dim=0)
            show_res = torch.cat((res, res, res), dim=0)
            grid_image = make_grid(
                [images[0].clone().cpu().data, show_gt, show_res], 3,
                normalize=True)
            sw.add_image('res', grid_image, i)
    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    if epoch % 10 == 0 or epoch % opt.epoch == 0:
        mae, validation_step = validation(generator, opt.trainsize, sw, validation_step)
        print('this validation mae is', mae, 'epoch is', epoch)
        sw.add_scalars('mae', {'mae': mae}, global_step=epoch)
        current_mae_avg = mae
        if current_mae_avg < min_mae and epoch > 150:
            min_mae = current_mae_avg
            print('the lower mae and saving models...')
            path = save_path + model_name + '_%d' % epoch + '_gen.pth'
            torch.save(generator.state_dict(), path)
    if epoch % opt.epoch == 0 or epoch % opt.epoch == 200 or epoch % opt.epoch == 250:
        path = save_path
        path = path + model_name + '_%d' % epoch + '_gen.pth'
        if opt.save_opt:
            save_checkpoint(path, generator, epoch, generator_optimizer)
        else:
            torch.save(generator.state_dict(), path)
