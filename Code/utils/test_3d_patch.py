import pathlib

import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label

def getLargestCC(segmentation):
    labels = label(segmentation)
    #assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        largestCC = segmentation
    return largestCC


def var_all_case_LA_mean(model1, model2, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,diff = False,has_feature=False,dataset = 'LA', text_embed=None,test_mode = False):
    if dataset == 'LA':
        with open('./Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/LA/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for
                      item
                      in image_list]
    elif dataset == 'BraTS':
        with open('./Datasets/brats/val.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"E:\4_Dataset\BraTS\2019-SSL\\" + item.replace('\n', '') + ".h5" for item in image_list]
    elif dataset =='KiTS19':
        with open('./Datasets/kits/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/KiTS19/KiTS19-SSL/" + item.replace('\n', '') + ".h5" for item in image_list]


    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if dataset=='KiTS19':
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction, score_map = test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size,
                                                      num_classes=num_classes,diff = diff,has_feature=has_feature,
                                                      text_embed=text_embed)
        if np.sum(prediction) == 0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice


def test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size, num_classes=1,diff = False,has_feature=False, text_embed=None):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    bcp = False

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                if not diff:
                    if text_embed is not None:
                        y1 = model1(test_patch, text_embed, bcp)[0]
                    else:
                        y1 = model1(test_patch)[0]
                    y1 = F.softmax(y1, dim=1)
                    if text_embed is not None:
                        y2 = model2(test_patch, text_embed, bcp)[0]
                    else:
                        y2 = model2(test_patch)[0]
                    y2 = F.softmax(y2, dim=1)
                else:
                    if text_embed is not None:
                        y1, _ = model1(test_patch, text_embed, bcp)
                    else:
                        y1, _ = model1(test_patch)
                    y1 = F.softmax(y1, dim=1)
                    y1 = (y1 > 0.5).float()
                    if has_feature:
                        if text_embed is not None:
                            y2, _ = model2(test_patch, y1, text_embed, bcp)
                        else:
                            y2, _ = model2(test_patch, y1)
                    else:
                        if text_embed is not None:
                            y2 = model2(test_patch, y1, text_embed, bcp)
                        else:
                            y2 = model2(test_patch, y1)
                    y2 = F.softmax(y2, dim=1)
                y1 = y1.cpu().data.numpy()
                y2 = y2.cpu().data.numpy()



                y = (y1[0, 1, :, :, :] + y2[0, 1, :, :, :]) / 2

                # with torch.no_grad():
                #     y1, _ = model(test_patch)
                #     y = F.softmax(y1, dim=1)

                # y = y.cpu().data.numpy()

                # y = y[0,1,:,:,:]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map

def var_all_case_LA(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,dataset='LA', text_embed=None):
    if dataset == 'LA':
        with open('./Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/LA/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for
                      item in image_list]
    elif dataset == 'BraTS':
        with open('./Datasets/brats/val.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"E:\4_Dataset\BraTS\2019-SSL" + item.replace('\n', '') + ".h5" for item in
                      image_list]
    elif dataset =='KiTS19':
        with open('./Datasets/kits/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/KiTS19/KiTS19-SSL/" + item.replace('\n', '') + ".h5" for item in image_list]

    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if dataset=='KiTS19':
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes, text_embed=text_embed)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def var_all_case_LA_cognition_difference(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,dataset='LA'):
    if dataset == 'LA':
        with open('../../Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/LA/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for
                      item in image_list]
    elif dataset == 'BraTS':
        with open('./Datasets/brats/val.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"E:\4_Dataset\BraTS\2019-SSL" + item.replace('\n', '') + ".h5" for item in
                      image_list]
    elif dataset =='KiTS19':
        with open('./Datasets/kits/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/KiTS19/KiTS19-SSL/" + item.replace('\n', '') + ".h5" for item in image_list]

    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if dataset=='KiTS19':
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def var_all_case_LA_cognition_difference_Dual(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,dataset='LA'):
    if dataset == 'LA':
        with open('../../Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/LA/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for
                      item in image_list]
    elif dataset == 'BraTS':
        with open('./Datasets/brats/val.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"E:\4_Dataset\BraTS\2019-SSL" + item.replace('\n', '') + ".h5" for item in
                      image_list]
    elif dataset =='KiTS19':
        with open('./Datasets/kits/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/KiTS19/KiTS19-SSL/" + item.replace('\n', '') + ".h5" for item in image_list]

    loader = tqdm(image_list)
    total_dice1 = 0.0
    total_dice2 = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if dataset=='KiTS19':
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction1, score_map1,prediction2, score_map2 = test_single_case_dual(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction1)==0:
            dice1 = 0
        else:
            dice1 = metric.binary.dc(prediction1, label)
        if np.sum(prediction2) == 0:
            dice2 = 0
        else:
            dice2 = metric.binary.dc(prediction2, label)
        total_dice1 += dice1
        total_dice2 += dice2
    avg_dice1 = total_dice1 / len(image_list)
    avg_dice2 = total_dice2 / len(image_list)
    print('average metric1 is {}'.format(avg_dice1))
    print('average metric2 is {}'.format(avg_dice2))
    return avg_dice1,avg_dice2


def test_single_case_dual(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map1 = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    score_map2 = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1, y2,_f1,_f2 = model(test_patch)
                    y1_out = F.softmax(y1, dim=1)
                    y2_out = F.softmax(y2, dim=1)


                y1_out = y1_out.cpu().data.numpy()
                y1_out = y1_out[0,1,:,:,:]
                y2_out = y2_out.cpu().data.numpy()
                y2_out = y2_out[0,1,:,:,:]
                score_map1[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map1[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y1_out
                score_map2[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map2[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y2_out
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map1 = score_map1/np.expand_dims(cnt,axis=0)
    label_map1 = (score_map1[0]>0.5).astype(int)
    score_map2 = score_map2 / np.expand_dims(cnt, axis=0)
    label_map2 = (score_map2[0] > 0.5).astype(int)
    if add_pad:
        label_map1 = label_map1[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map1 = score_map1[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        label_map2 = label_map2[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map2 = score_map2[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map1, score_map1,label_map2,score_map2

# region DiffUV val and test
def var_all_case_DiffUV(model_v,model_dfu, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                        model_choose = 0,has_feature=False,dataset = 'LA',in_prob = False, multi_step = False):

    if dataset=='LA':
        with open('./Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/LA/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item
                      in image_list]
    elif dataset=='BraTS':
        with open('./Datasets/brats/val.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"E:\4_Dataset\BraTS\2019-SSL\\" + item.replace('\n', '') + ".h5" for item in image_list]
    elif dataset =='KiTS19':
        with open('./Datasets/kits/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/KiTS19/KiTS19-SSL/" + item.replace('\n', '') + ".h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if dataset=='KiTS19':
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction, score_map = test_single_case_diffuv(model_v,model_dfu, image, stride_xy, stride_z, patch_size,
                                                        num_classes=num_classes,
                                                        model_choose = model_choose,
                                                        has_feature=has_feature,
                                                        in_prob= in_prob,
                                                        multi_step=multi_step)
        if np.sum(prediction) == 0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_single_case_diffuv(model_v,model_dfu, image, stride_xy, stride_z, patch_size,
                            num_classes=1,model_choose=0,has_feature = False,in_prob = False, multi_step = False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    if model_choose ==0:
                        y1, _ = model_v(test_patch)
                    elif model_choose==1:
                        y1_mid, _ = model_v(test_patch)
                        y1_mid = F.softmax(y1_mid, dim=1)
                        if not in_prob:
                            y1_mid = (y1_mid>0.5).float()
                        if has_feature:
                            y1, _ = model_dfu(test_patch, y1_mid, multi_step)
                        else:
                            y1 = model_dfu(test_patch, y1_mid, multi_step)
                    if y1.shape[1]==1:
                        y = F.sigmoid(y1)
                    else:
                        y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def test_all_case_DiffUV(model_v,model_dfu, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, model_choose = 0,has_feature=False,image_list= None,is_KiTS19 = False,save_path = None,in_prob = False, multi_step = False):
    if image_list==None:
        with open('./Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/LA/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for
                      item
                      in image_list]
    loader = tqdm(image_list)
    total_metric = 0.0
    for i,image_path in enumerate(loader):
        h5f = h5py.File(image_path, 'r')
        name = pathlib.Path(image_path).parent.name
        image = h5f['image'][:]
        label = h5f['label'][:]
        if is_KiTS19:
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction, score_map = test_single_case_diffuv(model_v,model_dfu, image, stride_xy, stride_z, patch_size, num_classes=num_classes,model_choose = model_choose,has_feature=has_feature,in_prob = in_prob,multi_step=multi_step)
        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        print(single_metric)
        if save_path != None:
            log_and_save_images(image=torch.from_numpy(image).unsqueeze(0).unsqueeze(0).cuda(),label=torch.from_numpy(label).unsqueeze(0).cuda(),outputs_pred=torch.from_numpy(prediction).unsqueeze(0).cuda(),
                                name=[name],model_name=str(model_choose),iter_num=i,writer=None,save_image_path=save_path,
                                dice_sample=single_metric[0])
        total_metric += np.asarray(single_metric)
    avg_metrics = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metrics))
    return avg_metrics

def get_test_pic(model_v,model_dfu, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,has_feature=False,image_list= None,is_KiTS19 = False,save_path = None):
    if image_list==None:
        with open('./Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/LA/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for
                      item
                      in image_list]
    loader = tqdm(image_list)
    total_metric = 0.0
    for i,image_path in enumerate(loader):
        h5f = h5py.File(image_path, 'r')
        name = pathlib.Path(image_path).parent.name
        image = h5f['image'][:]
        label = h5f['label'][:]
        if is_KiTS19:
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction_v, score_map_v = test_single_case_diffuv(model_v,model_dfu, image, stride_xy, stride_z, patch_size, num_classes=num_classes,model_choose = 0,has_feature=has_feature)
        prediction_dfu, score_map_dfu = test_single_case_diffuv(model_v,model_dfu, image, stride_xy, stride_z, patch_size, num_classes=num_classes,model_choose = 1,has_feature=True)
        if np.sum(prediction_v) == 0:
            single_metric_v = (0, 0, 0, 0)
            single_metric_dfu = (0, 0, 0, 0)
        else:
            single_metric_v = calculate_metric_percase(prediction_v, label[:])
            single_metric_dfu = calculate_metric_percase(prediction_dfu, label[:])
        if save_path != None:
            log_and_save_images_dual(image=torch.from_numpy(image).unsqueeze(0).unsqueeze(0).cuda(),label=torch.from_numpy(label).unsqueeze(0).cuda(),
                                     outputs_pred_h=torch.from_numpy(prediction_dfu).unsqueeze(0).cuda(),
                                     outputs_pred_l=torch.from_numpy(prediction_v).unsqueeze(0).cuda(),
                                     name=[name],iter_num=i,writer=None,save_image_path=save_path,
                                     dice_sample=single_metric_v[0])
        total_metric += np.asarray(single_metric_v)
    avg_metrics = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metrics))
    return avg_metrics

def test_all_case_LA_mean(model1, model2, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,diff = False,has_feature=False , image_list = None, is_KiTS19= False, text_embed=None):
    if image_list ==None:
        with open('./Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/LA/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for
                      item in image_list]

    loader = tqdm(image_list)
    total_metric = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if is_KiTS19:
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction, score_map = test_single_case_mean(model1, model2, image, stride_xy, stride_z, patch_size,
                                                      num_classes=num_classes,diff = diff,has_feature=has_feature,
                                                      text_embed=text_embed)
        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        total_metric += np.asarray(single_metric)
    avg_metrics = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metrics))
    return avg_metrics

# endregion
def test_all_case_average(model1, model2, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None, metric_detail=0, nms=0,is_KiTS19 = False,
                    text_embed=None):
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        if is_KiTS19:
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction, score_map = test_single_case_mean(model1, model2, image,
                                                      stride_xy, stride_z, patch_size, num_classes=num_classes,
                                                      text_embed=text_embed)
        if nms:
            prediction = getLargestCC(prediction)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])

        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
            ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)

        # if save_result:
        #     nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
        #              test_save_path + "%02d_pred.nii.gz" % ith)
        #     nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)),
        #              test_save_path + "%02d_scores.nii.gz" % ith)
        #     nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
        #     nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    # with open(test_save_path + '../performance.txt', 'w') as f:
    #     f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric
def test_all_case(model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0,is_KiTS19 = False):
    
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        if is_KiTS19:
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1, text_embed=None):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    bcp = False

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    if text_embed is not None:
                        y1, _ = model(test_patch, text_embed, bcp)
                    else:
                        y1, _ = model(test_patch)
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def var_all_case_LA_plus(model_l, model_r, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4):
   
    with open('/data/byh_data/SSNet_data/LA/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = ["/data/byh_data/SSNet_data/LA/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case_plus(model_l, model_r, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric

def test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1_l, _ = model_l(test_patch)
                    y1_r, _ = model_r(test_patch)
                    y1 = (y1_l + y1_r) / 2
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd

# region for Pic 1 G
def get_bias_pixel(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,dataset='LA'):
    if dataset == 'LA':
        with open('./Datasets/la/data_split/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/LA/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for
                      item in image_list]
    elif dataset == 'BraTS':
        with open('./Datasets/brats/val.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"E:\4_Dataset\BraTS\2019-SSL" + item.replace('\n', '') + ".h5" for item in
                      image_list]
    elif dataset =='KiTS19':
        with open('./Datasets/kits/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [r"/data1/mengqingxu/Dataset/KiTS19/KiTS19-SSL/" + item.replace('\n', '') + ".h5" for item in image_list]

    loader = tqdm(image_list)
    total_bias_pxiels = 0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if dataset=='KiTS19':
            # preprocess
            image = image.swapaxes(0, 2)  # 192*192*64
            label = label.swapaxes(0, 2)
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = (image - np.mean(image)) / np.std(image)
            label = (label > 0).astype(np.int8)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        bias_pixels = np.sum(prediction!=label)
        total_bias_pxiels += bias_pixels

    print('total bias pxiels is {}'.format(total_bias_pxiels))
    return total_bias_pxiels
#endregion