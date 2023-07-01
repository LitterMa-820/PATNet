import datetime
import os

import numpy as np
import sys
import argparse

sys.path.append('..')
from eval.load_test_data import test_dataset
from eval.saliency_metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, cal_acc
from score_config import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--mn', required=True, type=str, help='models name')
parser.add_argument('--modal', default='rgbd', type=str, help='rgbd or rgbt')
parser.add_argument('--p', required=True, type=str, help='salient map path')
opt = parser.parse_args()

if opt.modal == 'rgbd':
    test_datasets = {'DUT-RGBD': dutrgbd, 'NJU2K': njud, 'NJUD': njud, 'NLPR': nlpr, 'STERE': stere, 'SIP': sip,
                     'DES': rgbd135, 'RGBD135': rgbd135, 'SSD': ssd, 'LFSD': lfsd}
elif opt.modal == 'rgbt':
    test_datasets = {'VT-821': vt821, 'VT-1000': vt1000, 'VT-5000': vt5000}
else:
    print('please input rgbd or rgbt')
    exit()

RGBD_SOD_Models = {opt.p: os.path.join(results_path+'predictions/', opt.p)}
results_save_path = results_save_path+'result_latex/'
if not os.path.exists(results_save_path):
    os.makedirs(results_save_path)

results_save_path = results_save_path+ opt.p + '-' + str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')) + '.txt'
# table head
open(results_save_path, 'w').write(opt.p + '\n' + '\\begin{tabular}{cccccc}\n\\toprule' + '\n' + '\midrule\n')
open(results_save_path, 'a').write('datasets/metric & mae & maxF & wFm & Sm & Em \\\\\n')
mae_list, max_f_list, sm_list, em_list, wfm_list = [], [], [], [], []

nums_ever_dataset = []
for method_name, method_map_root in RGBD_SOD_Models.items():
    print('test method:', method_name, method_map_root)
    for name, root in test_datasets.items():
        print(name)
        sal_root = method_map_root + '/' + name
        print(sal_root)
        gt_root = root + 'GT'
        print(gt_root)
        if os.path.exists(sal_root):
            print('\033[32m file exist! \033[0m')
            test_loader = test_dataset(sal_root, gt_root)
            size = test_loader.size
            nums_ever_dataset.append(size)
            mae, fm, sm, em, wfm, m_dice, m_iou, ber, acc = cal_mae(), cal_fm(
                test_loader.size), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc()
            for i in tqdm(range(test_loader.size)):
                # print ('predicting for %d / %d' % ( i + 1, test_loader.size))
                sal, gt = test_loader.load_data()
                if sal.size != gt.size:
                    x, y = gt.size
                    sal = sal.resize((x, y))
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                gt[gt > 0.5] = 1
                gt[gt != 1] = 0
                res = sal
                res = np.array(res)
                if res.max() == res.min():
                    res = res / 255
                else:
                    res = (res - res.min()) / (res.max() - res.min())
                # 二值化会提升mae和meanf,em
                # res[res > 0.5] = 1
                # res[res != 1] = 0
                mae.update(res, gt)
                sm.update(res, gt)
                fm.update(res, gt)
                em.update(res, gt)
                wfm.update(res, gt)
            MAE = mae.show()
            mae_list.append(MAE)
            # maxf, meanf, _, _ = fm.show()
            maxf, _, _, _ = fm.show()
            max_f_list.append(maxf)
            # avg_mean_f.append(meanf)
            sm = sm.show()
            sm_list.append(sm)
            em = em.show()
            em_list.append(em)
            wfm = wfm.show()
            wfm_list.append(wfm)
            log = 'method_name: {} dataset: {} MAE: {:.4f} maxF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f} '.format(
                method_name, name, MAE, maxf, wfm, sm, em)
            print('\n' + log)
            table_content = name + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' \\\\'
            table_content = table_content.format(MAE, maxf, wfm, sm, em)
            open(results_save_path, 'a').write(table_content + '\n')
        else:
            print('\033[31m file is not exist! \033[0m')

    mae, max_f, wfm, sm, em = np.array(mae_list), np.array(max_f_list), np.array(wfm_list), np.array(
        sm_list), np.array(em_list)
    avg_mae, avg_max_f, avg_wfm, avg_sm, avg_em = np.mean(mae), np.mean(max_f), np.mean(wfm), np.mean(
        sm), np.mean(em)

    nums_ever_dataset = np.array(nums_ever_dataset)
    nums_sample = nums_ever_dataset.sum()
    w_avg_mae, w_avg_max_f, w_avg_wfm, w_avg_sm, w_avg_em = nums_ever_dataset * mae, nums_ever_dataset * max_f, nums_ever_dataset * wfm, nums_ever_dataset * sm, nums_ever_dataset * em
    # print('--------------------testing-------------------------')
    # print('nums_ever_dataset:', nums_ever_dataset)
    # print('nums_sample:', nums_sample)
    # print(w_avg_mae, '\n', w_avg_max_f, '\n', w_avg_wfm, '\n,', w_avg_sm, '\n,', w_avg_em)
    w_avg_mae, w_avg_max_f, w_avg_wfm, w_avg_sm, w_avg_em = np.sum(w_avg_mae/nums_sample), np.sum(w_avg_max_f/nums_sample), np.sum(w_avg_wfm/nums_sample), np.sum(w_avg_sm/nums_sample), np.sum(w_avg_em/nums_sample)

    avg_log = 'method_name: {} on all dataset avg_MAE: {:.4f} avg_maxF: {:.4f} avg_wfm: {:.4f} avg_Sm: {:.4f} avg_Em: {:.4f}'.format(
        method_name, avg_mae, avg_max_f, avg_wfm, avg_sm, avg_em)

    w_avg_log = 'method_name: {} on all dataset w_avg_MAE: {:.4f} avg_maxF: {:.4f} avg_wfm: {:.4f} avg_Sm: {:.4f} avg_Em: {:.4f}'.format(
        method_name, w_avg_mae, w_avg_max_f, w_avg_wfm, w_avg_sm, w_avg_em)
    print(avg_log)
    print(w_avg_log)

    table_avg = 'average' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' \\\\'
    table_w_avg = 'w_average' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' & ' + '{:.4f}' + ' \\\\'
    table_avg = table_avg.format(avg_mae, avg_max_f, avg_wfm, avg_sm, avg_em)
    table_w_avg = table_w_avg.format(w_avg_mae, w_avg_max_f, w_avg_wfm, w_avg_sm, w_avg_em)
    open(results_save_path, 'a').write(table_avg + '\n')
    open(results_save_path, 'a').write(table_w_avg + '\n')
    open(results_save_path, 'a').write('\\bottomrule\n\end{tabular}' + '\n')
