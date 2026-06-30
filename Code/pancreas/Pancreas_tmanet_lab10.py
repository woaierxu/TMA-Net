import os
import pathlib
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tqdm_load

CODE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from Code.networks.TMANet.embed_utils import get_embeddings
from Code.networks.TMANet.TMANet import TMA_Net
from dataloaders import get_ema_model_and_dataloader
from losses import DiceLoss, mix_loss, softmax_mse_loss
from pancreas_utils import (
    config_log,
    cutmix_config_log,
    generate_mask,
    get_cut_mask,
    load_net,
    load_net_opt,
    mkdir,
    save_net,
    save_net_opt,
    seed_reproducer,
    to_cuda,
    update_ema_variables,
)
from test_util import test_calculate_metric, test_calculate_metric_mean

"""Global Variables"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed_test = 2020
seed_reproducer(seed=seed_test)

data_root, split_name = r"/data1/mengqingxu/Dataset/Pancreas/preprocess-SDCL/", "pancreas"

batch_size, lr = 2, 1e-3
pretraining_epochs, self_training_epochs = 101, 321
pretrain_save_step, st_save_step, pred_step = 10, 20, 5
alpha, consistency, consistency_rampup = 0.99, 0.1, 40
label_percent = 10
u_weight = 1.5
connect_mode = 2
try_second = 1
sec_t = 0.5
self_train_name = "self_train"
file_name = pathlib.Path(__file__).name.replace(".py", "")
result_dir = f"./model//{file_name}_per{label_percent}/"
mkdir(result_dir)

sub_batch = int(batch_size / 2)
consistency_criterion = softmax_mse_loss
CE = nn.CrossEntropyLoss()
CE_r = nn.CrossEntropyLoss(reduction="none")
DICE = DiceLoss(nclass=2)
cutmix_size = 64
patch_size = (96, 96, 96)
num_classes = 2

logger = None
text_embed = None


def create_TMAnet(ema=False):
    net = TMA_Net(
        patch_size=patch_size,
        n_channels=1,
        n_classes=num_classes,
        normalization="instancenorm",
        has_dropout=False,
        text_fuse_level=1,
    )
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)

    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = l1 != l2
    return diff_mask


def pretrain(net_bcp, net_nobcp, optimizer1, optimizer2, lab_loader_a, lab_loader_b, test_loader):
    """Pretrain BCP-aware and no-BCP TMA-fusion networks."""

    save_path = Path(result_dir) / "pretrain"
    save_path.mkdir(exist_ok=True)

    global logger
    logger, _ = cutmix_config_log(save_path, tensorboard=True)
    logger.info("TMA fusion pretrain, patch_size: {}, save path: {}".format(cutmix_size, str(save_path)))

    max_dice1 = 0
    max_dice2 = 0

    global text_embed
    text_embed = get_embeddings(dataset_name="Pancreas")

    for epoch in tqdm_load(range(1, pretraining_epochs + 1), ncols=70):
        if epoch % 5 == 0:
            net_bcp.eval()
            net_nobcp.eval()
            avg_metric1, _ = test_calculate_metric(net_bcp, test_loader.dataset, s_xy=32, s_z=32, text_embed=text_embed)
            avg_metric2, _ = test_calculate_metric(
                net_nobcp, test_loader.dataset, s_xy=32, s_z=32, text_embed=text_embed
            )

            logger.info("average metric is : {}".format(avg_metric1))
            logger.info("average metric is : {}".format(avg_metric2))
            val_dice1 = avg_metric1[0]
            val_dice2 = avg_metric2[0]

            if val_dice1 > max_dice1:
                save_net_opt(net_bcp, optimizer1, save_path / f"best_ema{label_percent}_pre_vnet.pth", epoch)
                max_dice1 = val_dice1
                with open(str(save_path / f"epoch_{epoch}_{label_percent}_ValDice1_{round(val_dice1, 4)}_self.txt"), "w") as f:
                    f.write(f"val_dice1:{round(val_dice1, 4)}")

            if val_dice2 > max_dice2:
                save_net_opt(net_nobcp, optimizer2, save_path / f"best_ema{label_percent}_pre_resnet.pth", epoch)
                max_dice2 = val_dice2
                with open(str(save_path / f"epoch_{epoch}_{label_percent}_ValDice2_{round(val_dice2, 4)}_self.txt"), "w") as f:
                    f.write(f"val_dice2:{round(val_dice2, 4)}")

            logger.info("\nEvaluation: val_dice: %.4f, val_maxdice: %.4f " % (val_dice1, max_dice1))
            logger.info("resnet Evaluation: val_dice: %.4f, val_maxdice: %.4f " % (val_dice2, max_dice2))

        net_bcp.train()
        net_nobcp.train()
        logger.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            img_mask, loss_mask = generate_mask(img_a, cutmix_size)

            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)
            outputs, _ = net_bcp(volume_batch, text_embed, bcp=True, bcp_mask=loss_mask)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            volume_batch2 = torch.concat([img_a, img_b], dim=0)
            label_batch2 = torch.concat([lab_a, lab_b], dim=0)
            outputs2, _ = net_nobcp(volume_batch2, text_embed, bcp=False)
            loss_ce2 = F.cross_entropy(outputs2, label_batch2)
            loss_dice2 = DICE(outputs2, label_batch2)
            loss2 = (loss_ce2 + loss_dice2) / 2

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            logger.info(
                "epoch %d step %d : loss: %.4f, loss_dice: %.4f, loss_ce: %.4f, "
                "res_loss: %.4f, res_dice: %.4f, res_ce: %.4f"
                % (epoch, step + 1, loss.item(), loss_dice.item(), loss_ce.item(), loss2.item(), loss_dice2.item(), loss_ce2.item())
            )

    return max_dice1


def ema_cutmix(
    net_bcp,
    net_nobcp,
    ema_net1,
    optimizer1,
    optimizer2,
    lab_loader_a,
    lab_loader_b,
    unlab_loader_a,
    unlab_loader_b,
    test_loader,
):
    save_path = Path(result_dir) / self_train_name
    save_path.mkdir(exist_ok=True)

    global logger
    logger, _ = config_log(save_path, tensorboard=True)
    logger.info("TMA fusion EMA training, save_path: {}".format(str(save_path)))

    pretrained_path = Path(result_dir) / "pretrain"
    load_net_opt(net_bcp, optimizer1, pretrained_path / f"best_ema{label_percent}_pre_vnet.pth")
    load_net_opt(net_nobcp, optimizer2, pretrained_path / f"best_ema{label_percent}_pre_resnet.pth")
    load_net_opt(ema_net1, optimizer1, pretrained_path / f"best_ema{label_percent}_pre_vnet.pth")
    logger.info("Loaded from {}".format(pretrained_path))

    max_dice1 = 0
    max_list1 = None
    max_dice3 = 0
    for i_epoch, epoch in enumerate(tqdm_load(range(1, self_training_epochs + 1))):
        logger.info("")

        if (epoch % 20 == 0) | ((epoch >= 160) & (epoch % 5 == 0)):
            net_bcp.eval()
            net_nobcp.eval()
            avg_metric3, _ = test_calculate_metric_mean(
                net_bcp, net_nobcp, test_loader.dataset, s_xy=32, s_z=32, text_embed=text_embed
            )

            logger.info("mean average metric is : {}".format(avg_metric3))
            val_dice3 = avg_metric3[0]

            if val_dice3 > max_dice3:
                save_net(net_bcp, str(save_path / f"best_ema_{label_percent}_self_v.pth"))
                save_net(net_nobcp, str(save_path / f"best_ema_{label_percent}_self_r.pth"))
                max_dice3 = val_dice3
                with open(str(save_path / f"epoch_{epoch}_{label_percent}_MeanDice_{round(val_dice3, 4)}_self.txt"), "w") as f:
                    f.write(f"val_dice3:{round(val_dice3, 4)}")

            logger.info("mean Evaluation: val_dice: %.4f, val_maxdice: %.4f " % (val_dice3, max_dice3))

        net_bcp.train()
        net_nobcp.train()
        ema_net1.train()
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, _), (unimg_b, _)) in enumerate(
            zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)
        ):
            img_a, lab_a, img_b, lab_b, unimg_a, unimg_b = to_cuda([img_a, lab_a, img_b, lab_b, unimg_a, unimg_b])

            with torch.no_grad():
                unoutput_a_1, _ = ema_net1(unimg_a, text_embed, bcp=False)
                unoutput_b_1, _ = ema_net1(unimg_b, text_embed, bcp=False)

                plab_a = get_cut_mask(unoutput_a_1, nms=True, connect_mode=connect_mode)
                plab_b = get_cut_mask(unoutput_b_1, nms=True, connect_mode=connect_mode)

                img_mask, loss_mask = generate_mask(img_a, cutmix_size)

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
                outputs_l, _ = net_bcp(purel_img, text_embed, bcp=False)
                outputs_u, _ = net_bcp(pureu_img, text_embed, bcp=False)
                loss_l = (F.cross_entropy(outputs_l, purel_label) + DICE(outputs_l, purel_label)) / 2
                loss_u = (F.cross_entropy(outputs_u, pureu_label) + DICE(outputs_u, pureu_label)) / 2
            else:
                outputs_l, _ = net_bcp(mixl_img, text_embed, bcp=True, bcp_mask=loss_mask)
                outputs_u, _ = net_bcp(mixu_img, text_embed, bcp=True, bcp_mask=loss_mask)
                loss_l = mix_loss(outputs_l, plab_a.long(), lab_b, loss_mask, u_weight=u_weight, unlab=True)
                loss_u = mix_loss(outputs_u, lab_a, plab_b.long(), loss_mask, u_weight=u_weight)

            outputs_l_2, _ = net_nobcp(purel_img, text_embed, bcp=False)
            outputs_u_2, _ = net_nobcp(pureu_img, text_embed, bcp=False)
            loss_l_2 = (F.cross_entropy(outputs_l_2, purel_label) + DICE(outputs_l_2, purel_label)) / 2
            loss_u_2 = (F.cross_entropy(outputs_u_2, pureu_label) + DICE(outputs_u_2, pureu_label)) / 2

            loss = loss_l + loss_u
            loss_2 = loss_l_2 + loss_u_2

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            update_ema_variables(net_bcp, ema_net1, alpha)

            logger.info(
                "epoch %d step %d : loss: %.4f, loss_l: %.4f, loss_u: %.4f, "
                "res_loss: %.4f, res_loss_l: %.4f, res_loss_u: %.4f"
                % (
                    epoch,
                    step + 1,
                    loss.item(),
                    loss_l.item(),
                    loss_u.item(),
                    loss_2.item(),
                    loss_l_2.item(),
                    loss_u_2.item(),
                )
            )

        if epoch == self_training_epochs:
            save_net(net_bcp, str(save_path / f"best_ema_{label_percent}_self_latest.pth"))
    return max_dice1, max_list1


def test_model(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    load_path = Path(result_dir) / self_train_name
    load_net(net1, load_path / f"best_ema_{label_percent}_self_v.pth")
    load_net(net2, load_path / f"best_ema_{label_percent}_self_r.pth")
    print("Successful Loaded")
    avg_metric, _ = test_calculate_metric(net1, test_loader.dataset, s_xy=32, s_z=32, text_embed=text_embed)
    avg_metric2, _ = test_calculate_metric(net2, test_loader.dataset, s_xy=32, s_z=32, text_embed=text_embed)
    avg_metric3, _ = test_calculate_metric_mean(net1, net2, test_loader.dataset, s_xy=32, s_z=32, text_embed=text_embed)
    print(avg_metric)
    print(avg_metric2)
    print(avg_metric3)


if __name__ == "__main__":
    import pynvml

    free_memory = 0
    threshold = 15
    nvml_was_init = False
    while free_memory < threshold:
        pynvml.nvmlInit()
        nvml_was_init = True
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = info.free / 1024**3
            print(f"GPU {i} Remain Memory: {free_memory:.2f} GB")
            if free_memory >= threshold:
                break
            time.sleep(60)
    if nvml_was_init:
        pynvml.nvmlShutdown()
    print("runing!")
    try:
        (_, _, _, _, _, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader) = (
            get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=label_percent)
        )
        net1 = create_TMAnet()
        net2 = create_TMAnet()
        ema_net1 = create_TMAnet(ema=True)
        optimizer1 = torch.optim.Adam(net1.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=lr)
        pretrain(net1, net2, optimizer1, optimizer2, lab_loader_a, lab_loader_b, test_loader)
        seed_reproducer(seed=seed_test)
        ema_cutmix(net1, net2, ema_net1, optimizer1, optimizer2, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader)
        test_model(net1, net2, test_loader)

    except Exception as e:
        if logger is not None:
            logger.exception("BUG FOUNDED ! ! !")
        raise e
