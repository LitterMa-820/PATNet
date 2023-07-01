# coding: utf-8
import os
import sys

sys.path.append('..')
from models.ablationstudy import p3NetAblation1

# dutrgbd_root_test = '../datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/DUT-RGBD/'
# njud_root_test = '../datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/NJUD/'
# nlpr_root_test = '../datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/NLPR/'
# stere_root_test = '../datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/STERE/'
# sip_root_test = '../datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/SIP/'
# rgbd135_root_test = '../datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/RGBD135/'
# ssd_root_test = '../datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/SSD/'
# lfsd_root_test = '../datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/LFSD/'
#
# vt821_root_test = '../datasets/RGBT_SOD_Datasets/test/VT821/'
# vt1000_root_test = '../datasets/RGBT_SOD_Datasets/test/VT1000/'
# vt5000_root_test = '../datasets/RGBT_SOD_Datasets/test/VT5000/'

dutrgbd_root_test = '/home/amax/mjh_datasets/datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/DUT-RGBD/'
njud_root_test = '/home/amax/mjh_datasets/datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/NJUD/'
nlpr_root_test = '/home/amax/mjh_datasets/datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/NLPR/'
stere_root_test = '/home/amax/mjh_datasets/datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/STERE/'
sip_root_test = '/home/amax/mjh_datasets/datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/SIP/'
rgbd135_root_test = '/home/amax/mjh_datasets/datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/RGBD135/'
ssd_root_test = '/home/amax/mjh_datasets/datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/SSD/'
lfsd_root_test = '/home/amax/mjh_datasets/datasets/RGBD/NJUD_NLPR_DUT_2985/Testing_dataset/LFSD/'

vt821_root_test = '/home/amax/mjh_datasets/RGBT_SOD_Datasets/test/VT821/'
vt1000_root_test = '/home/amax/mjh_datasets/RGBT_SOD_Datasets/test/VT1000/'
vt5000_root_test = '/home/amax/mjh_datasets/RGBT_SOD_Datasets/test/VT5000/'

dutrgbd = os.path.join(dutrgbd_root_test)
njud = os.path.join(njud_root_test)
nlpr = os.path.join(nlpr_root_test)
stere = os.path.join(stere_root_test)
sip = os.path.join(sip_root_test)
rgbd135 = os.path.join(rgbd135_root_test)
ssd = os.path.join(ssd_root_test)
lfsd = os.path.join(lfsd_root_test)

vt821 = os.path.join(vt821_root_test)
vt1000 = os.path.join(vt1000_root_test)
vt5000 = os.path.join(vt5000_root_test)

results_path = '../results/'
results_save_path = os.path.join('..', 'results/')

RGBD_SOD_RGBT_Models = {'P2TBaseLineRGBT': os.path.join(results_path, 'P2TBaselineRGBT_100_gen/')}


def getNet():
    return p3NetAblation1(None).cuda()


def infer(net, rgb_trans, m_trans):
    side_out4, side_out3, side_out2, side_out1 = net(rgb_trans, m_trans)
    return side_out1
