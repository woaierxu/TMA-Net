import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import pdb



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

def test_single_volume_mean(image, label, model, model2, classes, patch_size=[256, 256],in_pseudo=False):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        model2.eval()
        with torch.no_grad():
            output1 = model(input)
            output1_prob = torch.softmax(output1,dim=1)
            if in_pseudo:
                output1_prob =(output1_prob>0.5).float()
            output2 = model2(input,output1_prob)


            if len(output1)>1:
                output1 = output1[0]


            if len(output2)>1:
                output2 = output2[0]

            mean_prob = (torch.softmax(output1, dim=1) + torch.softmax(output2, dim=1)) / 2

            out = torch.argmax(mean_prob, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    if len(size) == 3 :
        tensor = tensor.unsqueeze(dim = 1)
        size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor.long(), 1)
    return one_hot

def test_single_volume(image, label, model1, model2, classes, patch_size=[256, 256], model_choose = 0,pre_train = False,in_pseudo = False):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        slice_label = label[ind, :, :]
        x, y = slice_label.shape[0], slice_label.shape[1]
        slice_label = zoom(slice_label, (patch_size[0] / x, patch_size[1] / y), order=0)
        slice_label  = torch.from_numpy(slice_label).unsqueeze(0).unsqueeze(0).float().cuda()
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        # model1.eval()
        # model2.eval()
        with torch.no_grad():
            if model_choose ==0 :
                output = model1(input)
                if len(output)>1:
                    output = output[0]
            elif model_choose == 1:
                if pre_train:
                    output = model2(input,to_one_hot(slice_label,nClasses=classes))
                    if len(output) > 1:
                        output = output[0]
                else:
                    out_mid = model1(input)
                    out_mid = torch.softmax(out_mid,dim=1)
                    if in_pseudo:
                        out_mid = (out_mid>0.5).float()
                    output = model2(input,out_mid)
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def test_single_volume_cross(image, label, model_l, model_r, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model_r.eval()
        model_l.eval()
        with torch.no_grad():
            output_l = model_l(input)
            output_r = model_r(input)
            output = (output_l + output_r) / 2
            if len(output)>1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


if __name__ == '__main__':
    torch.cuda.empty_cache()
