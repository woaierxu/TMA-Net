import pathlib
import os
import sys
import random
import argparse
import logging
import shutil

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from skimage.measure import label
from torch.utils.data import DataLoader
from tqdm import tqdm

from Code.networks.TMANet.embed_utils import get_embeddings
from utils import losses, test_3d_patch
from dataloaders.LADataset import LAHeart
from utils.LA_utils import to_cuda
from utils.BCP_utils import context_mask, update_ema_variables, DICE
from pancreas.losses import mix_loss, mix_mse_loss

from networks.Vnet import VNet
from networks.TMANet.TMANet import TMA_Net
from networks.ResVNet import ResVNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default=r'/data1/mengqingxu/Dataset/LA/data/', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='Baseline', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=6000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=1e-3, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1345, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
parser.add_argument(
    '--prompt_variant',
    type=str,
    default='LA',
    choices=[
        'LA',
    ],
    help='BiomedCLIP prompt ablation variant.',
)
args = parser.parse_args()


def create_TMAnet(ema=False):
    net = TMA_Net(
        patch_size = (112,112,80),
        n_channels=1,
        n_classes=2,
        normalization='instancenorm',
        has_dropout=False,
        text_fuse_level=1
    )
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model



def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    batch_array = np.asarray(batch_list)
    return torch.from_numpy(batch_array).to(device=segmentation.device)


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr

patch_size = (112, 112, 80)
num_classes = 2


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)

    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = (l1 != l2)
    return diff_mask


def pre_train(args, snapshot_path):
    model_bcp = create_TMAnet()
    model_nobcp = create_TMAnet()

    c_batch_size = args.batch_size//2
    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_lab{args.labelnum}', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_lab{args.labelnum}', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)



    optimizer = optim.Adam(model_bcp.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model_nobcp.parameters(), lr=1e-3)

    DICE = losses.mask_DiceLoss(nclass=2)

    model_bcp.train()
    model_nobcp.train()
    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    best_dice2 = 0
    max_epoch = args.pre_max_iteration // len(lab_loader_a) + 1
    iterator = tqdm(range(1, max_epoch), ncols=70)
    text_embed = get_embeddings(dataset_name=args.prompt_variant)
    for epoch_num in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)
            # bcp model input is image after bcp
            outputs,_ = model_bcp(volume_batch,text_embed,bcp=True,bcp_mask=loss_mask)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2
            # nobcp model input is two labeled images combine
            # so bcp = false is train more times in pretrain process
            volume_batch = torch.concat([img_a,img_b],dim=0)
            label_batch = torch.concat([lab_a,lab_b],dim=0)
            outputs2,feature = model_nobcp(volume_batch,text_embed,bcp=False)
            loss_ce2 = F.cross_entropy(outputs2, label_batch)
            loss_dice2 = DICE(outputs2, label_batch)
            loss2 = (loss_ce2 + loss_dice2) / 2

            iter_num += 1
            if iter_num > args.pre_max_iteration:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))

        if iter_num > args.pre_max_iteration:
            break

        if epoch_num % 5 == 0:
            model_bcp.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model_bcp, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=54, stride_z=40,text_embed=text_embed)
            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_best_path = os.path.join(snapshot_path, 'best_model.pth')
                save_net_opt(model_bcp, optimizer, save_best_path, epoch_num)
                with open(os.path.join(snapshot_path, 'best_model_iter.txt'), 'a') as f:
                    f.write(f'iter: {iter_num}, dice: {best_dice}\n')
                logging.info("save best model to {}, iter: {}, dice: {}".format(save_best_path, iter_num, best_dice))

            model_bcp.train()

            model_nobcp.eval()
            dice_sample2 = test_3d_patch.var_all_case_LA(model_nobcp, num_classes=num_classes, patch_size=patch_size,
                                                         stride_xy=54, stride_z=40,text_embed=text_embed)
            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_best_path = os.path.join(snapshot_path, 'best_model_resnet.pth')
                save_net_opt(model_nobcp, optimizer2, save_best_path, epoch_num)
                with open(os.path.join(snapshot_path, 'best_model_resnet_iter.txt'), 'a') as f:
                    f.write(f'iter: {iter_num}, dice: {best_dice2}\n')
                logging.info("save best resnet model to {}, iter: {}, dice: {}".format(save_best_path, iter_num, best_dice2))
            model_nobcp.train()



def self_train(args, pre_snapshot_path, self_snapshot_path):
    model_bcp = create_TMAnet()
    model_nobcp = create_TMAnet()
    ema_model1 = create_TMAnet(ema=True).cuda()



    c_batch_size = args.batch_size//2
    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_lab{args.labelnum}', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_lab{args.labelnum}', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_unlab{args.labelnum}', logging=logging)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split=f'train_unlab{args.labelnum}', reverse=True, logging=logging)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)



    optimizer = optim.Adam(model_bcp.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model_nobcp.parameters(), lr=1e-3)


    pretrained_model = os.path.join(pre_snapshot_path, 'best_model.pth')
    pretrained_model2 = os.path.join(pre_snapshot_path, 'best_model_resnet.pth')

    load_net_opt(model_bcp, optimizer, pretrained_model)
    load_net_opt(model_nobcp, optimizer2, pretrained_model2)

    load_net_opt(ema_model1, optimizer, pretrained_model)


    model_bcp.train()
    model_nobcp.train()
    ema_model1.train()

    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    best_dice2 = 0
    mean_best_dice = 0
    # max_epoch = 276
    max_epoch = args.self_max_iteration // len(lab_loader_a) + 1
    iterator = tqdm(range(1, max_epoch), ncols=70)
    text_embed = get_embeddings(dataset_name=args.prompt_variant)
    for i_epoch,epoch in enumerate(iterator):
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(
                zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda(
                [img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])

            with torch.no_grad():

                unoutput_a_1,_ = ema_model1(unimg_a,text_embed,bcp=False)
                unoutput_b_1,_ = ema_model1(unimg_b,text_embed,bcp=False)


                plab_a = get_cut_mask(unoutput_a_1, nms=1)
                plab_b = get_cut_mask(unoutput_b_1, nms=1)

                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            # bcp model's input is the image after bcp
            mixl_img = unimg_a * img_mask + img_b * (1 - img_mask)
            mixu_img = img_a * img_mask + unimg_b * (1 - img_mask)
            if i_epoch % 2 == 0:
                purel_img = img_a
                purel_label = lab_a
                pureu_img = unimg_a
                pureu_label = plab_a.long()
            else:
                purel_img = img_b
                purel_label = lab_b
                pureu_img = unimg_b
                pureu_label = plab_b.long()

            if random.randint(1, 5) == 1:
                outputs_l, _ = model_bcp(purel_img, text_embed, bcp=False)
                outputs_u, _ = model_bcp(pureu_img, text_embed, bcp=False)
                loss_l = (F.cross_entropy(outputs_l, purel_label) + DICE(outputs_l, purel_label)) / 2
                loss_u = (F.cross_entropy(outputs_u, pureu_label) + DICE(outputs_u, pureu_label)) / 2
            else:
                outputs_l, _ = model_bcp(mixl_img, text_embed, bcp=True, bcp_mask=loss_mask)
                outputs_u, _ = model_bcp(mixu_img, text_embed, bcp=True, bcp_mask=loss_mask)
                loss_l = mix_loss(outputs_l, plab_a.long(), lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
                loss_u = mix_loss(outputs_u, lab_a, plab_b.long(), loss_mask, u_weight=args.u_weight)

            outputs_l_2, _ = model_nobcp(purel_img, text_embed, bcp=False)
            outputs_u_2, _ = model_nobcp(pureu_img, text_embed, bcp=False)
            loss_l_2 = (F.cross_entropy(outputs_l_2, purel_label) + DICE(outputs_l_2, purel_label)) / 2
            loss_u_2 = (F.cross_entropy(outputs_u_2, pureu_label) + DICE(outputs_u_2, pureu_label)) / 2

            loss = (loss_l + loss_u)

            loss_2 = (loss_l_2 + loss_u_2)


            iter_num += 1
            if iter_num > args.self_max_iteration:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            logging.info('epoch %d iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f \
               net1_mse_loss_lab: %.4f, net1_mse_loss_unlab: %.4f, \
               ' % (epoch, iter_num, loss, loss_l, loss_u,0, 0))

            update_ema_variables(model_bcp, ema_model1, 0.99)

        if iter_num > args.self_max_iteration:
            break

        if epoch % 5 == 0:
            model_bcp.eval()
            model_nobcp.eval()
            mean_dice_sample = test_3d_patch.var_all_case_LA_mean(model_bcp, model_nobcp, num_classes=num_classes,
                                                                  patch_size=patch_size, stride_xy=54, stride_z=40,
                                                                  text_embed=text_embed)
            with open(os.path.join(self_snapshot_path, 'validation_metrics.txt'), 'a') as f:
                f.write(f'iter: {iter_num}, mean_dice: {round(mean_dice_sample, 4)}\n')

            # if dice_sample > best_dice:
            #     best_dice = round(dice_sample, 4)
            #     save_best_path = os.path.join(self_snapshot_path, 'best_model.pth')
            #     torch.save(model1.state_dict(), save_best_path)
            #     with open(os.path.join(self_snapshot_path, 'best_model_iter.txt'), 'a') as f:
            #         f.write(f'iter: {iter_num}, dice: {best_dice}\n')
            #     logging.info("save best model to {}, iter: {}, dice: {}".format(save_best_path, iter_num, best_dice))
            #     logging.info("cur dice %.4f, max dice %.4f" % (dice_sample, best_dice))
            #
            # if dice_sample2 > best_dice2:
            #     best_dice2 = round(dice_sample2, 4)
            #     save_best_path = os.path.join(self_snapshot_path, 'best_model_res.pth')
            #     torch.save(model2.state_dict(), save_best_path)
            #     with open(os.path.join(self_snapshot_path, 'best_model_res_iter.txt'), 'a') as f:
            #         f.write(f'iter: {iter_num}, dice: {best_dice2}\n')
            #     logging.info("save best res model to {}, iter: {}, dice: {}".format(save_best_path, iter_num, best_dice2))
            #     logging.info("resnet cur dice %.4f, max dice %.4f" % (dice_sample2, best_dice2))

            if mean_dice_sample > mean_best_dice:
                mean_best_dice = round(mean_dice_sample, 4)
                save_best_path1 = os.path.join(self_snapshot_path, 'best_model_v.pth')
                save_best_path2 = os.path.join(self_snapshot_path, 'best_model_r.pth')

                torch.save(model_bcp.state_dict(), save_best_path1)
                torch.save(model_nobcp.state_dict(), save_best_path2)

                with open(os.path.join(self_snapshot_path, 'best_model_mean_iter.txt'), 'a') as f:
                    f.write(f'iter: {iter_num}, mean_dice: {mean_best_dice}\n')
                logging.info("mean save best model to {}, iter: {}, dice: {}".format(save_best_path1, iter_num, mean_best_dice))
                logging.info("mean cur dice %.4f, max dice %.4f" % (mean_dice_sample, mean_best_dice))

            model_bcp.train()
            model_nobcp.train()


if __name__ == "__main__":
    import time
    import psutil
    def is_script_running(script_name):
        """使用psutil模块检查指定名称的Python脚本是否正在运行"""
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                # 检查进程名称是否包含python
                if 'python' in proc.info['name']:
                    # 获取进程的命令行参数
                    cmdline = proc.info['cmdline']
                    if cmdline and script_name in ' '.join(cmdline):
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False
    TARGET_SCRIPT = "LA_TMA_MBM_dual_lab10.py"
    while True:
        if is_script_running(TARGET_SCRIPT):
            print(f"脚本 {TARGET_SCRIPT} 正在运行，1分钟后再次检查...")
            time.sleep(60)  # 等待1分钟
        else:
            print(f"脚本 {TARGET_SCRIPT} 未运行，继续执行后续代码")
            break
    ## make logger file
    args.exp = pathlib.Path(__file__).name.replace(".py", "")
    if args.prompt_variant != "LA":
        args.exp = f"{args.exp}_{args.prompt_variant.lower().replace('-', '_')}"
    pre_snapshot_path = f"./model/LA/{args.exp}/pre_train"
    self_snapshot_path = f"./model/LA/{args.exp}/self_train"
    print("Starting TMA-Net training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copyfile(__file__, f"./model/LA/{args.exp}/{pathlib.Path(__file__).name}")
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    # -- Pre-Training
    if False:
        pre_snapshot_path = f'./model/LA/LA_TMA_MBM_dual_lab10/pre_train'
    else:
        logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        pre_train(args, pre_snapshot_path)
    logging.info("start self-train.")
    self_train(args, pre_snapshot_path, self_snapshot_path)
