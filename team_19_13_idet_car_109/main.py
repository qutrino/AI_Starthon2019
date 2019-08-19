import os
import inspect
import math
import datetime

import numpy as np

import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


import torchvision.models as models
import argparse

from data_loader import feed_infer
from data_local_loader import data_loader, data_loader_with_split
from evaluation import evaluation_metrics, compute_iou_wh

from jenti_models.CornerSimple import model as jenti_normal
from jenti_models.CornerSqueezeSimple import model as jenti_squeeze
from jenti_models.CornerOffset import model as jenti_offset
from jenti_models.CornerSqueezeOffset import model as jenti_squeeze_offset
from jenti_models.py_utils.modules import _decode_one

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

if IS_ON_NSML:
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train', 'train_data')
    VAL_DATASET_PATH = None
else:
    TRAIN_DATASET_PATH = os.path.join('/home/data/NIPAKoreanCarLocalize/train/train_data')
    VAL_DATASET_PATH = os.path.join('/home/data/NIPAKoreanCarLocalize/test')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResNet(models.ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_resnet18():
    return ResNet(models.resnet.BasicBlock,
                      [2, 2, 2, 2],
                      num_classes=4)




def _infer(model, root_path, test_loader=None):
    model.eval()

    if test_loader is None:
        test_loader = data_loader(
            root=os.path.join(root_path, 'test_data'),
            phase='test')

    outputs = []
    s_t = time.time()
    for idx, (image, _, _, _, _, _, _, _, _) in enumerate(test_loader):
        image = image.cuda()
        with torch.no_grad():
            output, t, b = model(image, test=True)
        output = output.detach().cpu().numpy()
        outputs.append(output)

        if time.time() - s_t > 10:
            print('Infer batch {}/{}.'.format(idx + 1, len(test_loader)))

    outputs = np.concatenate(outputs, 0)
    return outputs


def local_eval(model, test_loader=None, test_label_file=None):
    prediction_file = 'pred_train.txt'
    feed_infer(prediction_file, lambda root_path: _infer(model, root_path, test_loader=test_loader))
    if not test_label_file:
        test_label_file = os.path.join(VAL_DATASET_PATH, 'test_label')
    metric_result = evaluation_metrics(
        prediction_file,
        test_label_file
    )
    print('Eval result: {:.4f} mIoU'.format(metric_result))
    return metric_result


def bind_nsml(model, optimizer, scheduler):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--gamma", type=float, default=1.0)
    args.add_argument("--train_split", type=float, default=0.9)
    args.add_argument("--lr", type=float, default=0.00025)
    args.add_argument("--weight_decay", type=float, default=0.0)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--print_iter", type=int, default=1)
    args.add_argument("--eval_split", type=str, default='val')
    args.add_argument('--save_every', type=int, default=1)
    args.add_argument('--my_tag', type=str, default='')
    #args.add_argument("--model", type=str, default='normal')

    # reserved for nsml
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    train_split = config.train_split
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    eval_split = config.eval_split
    mode = config.mode


    # if config.model == "normal":
    #     model = jenti_normal()
    # else:
    #     model = jenti_squeeze()


    #model = get_resnet18()

    model = jenti_offset()

    print(model)
    print("total parameter::", count_parameters(model))

    #exit()
    #loss_fn = nn.MSELoss()
    #init_weight(model)

    if cuda:
        model = model.cuda()
        #loss_fn = loss_fn.cuda()

    # optimizer = Adam(
    #     [param for param in model.parameters() if param.requires_grad],
    #     lr=base_lr, weight_decay=1e-4)
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    optimizer = Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=base_lr, weight_decay=config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)

    bind_nsml(model, optimizer, scheduler)
    if config.pause:
        nsml.paused(scope=locals())

    if mode == 'train':
        tr_loader, val_loader, val_label_file = data_loader_with_split(root=TRAIN_DATASET_PATH, train_split=train_split)

        num_batches = len(tr_loader)

        #local_eval(model, val_loader, val_label_file)

        #exit(0)

        for epoch in range(num_epochs):

            time_epoch = datetime.datetime.now()
            time_ = datetime.datetime.now()

            scheduler.step()
            print('Epoch:', epoch, 'LR:', scheduler.get_lr())
            model.train()
            for iter_, data in enumerate(tr_loader):
                x, label, tl_heat, br_heat, mask,  t_o, b_o, t_tag, b_tag = data


                if cuda:
                    x = x.cuda()
                    label = label.cuda()
                    tl_heat = tl_heat.cuda()
                    br_heat = br_heat.cuda()
                    mask = mask.cuda()
                    t_o = t_o.cuda()
                    b_o = b_o.cuda()
                    t_tag = t_tag.cuda()
                    b_tag = b_tag.cuda()

                pred = model(x)
                loss = model.loss(pred, [tl_heat, br_heat, mask, t_o, b_o, t_tag, b_tag])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (iter_ + 1) % print_iter == 0:

                    pred_xywh = _decode_one( pred[0][-1], pred[1][-1], pred[2][-1], pred[3][-1] )
                    #print(pred_xywh.shape, label.shape)
                    train_iou  = compute_iou_wh(pred_xywh, label).mean().item()

                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (num_batches / print_iter)
                    _epoch = epoch + ((iter_ + 1) / num_batches)
                    print('[{:.3f}/{:d}] loss({}) iou({}) '
                          'elapsed {} expected per epoch {}'.format(
                              _epoch, num_epochs, loss.item(), train_iou, elapsed, expected))

                    time_ = datetime.datetime.now()



            val_iou = local_eval(model, val_loader, val_label_file)
            elapsed = datetime.datetime.now() - time_epoch
            if (epoch+1) % config.save_every == 0:
                nsml.report(**{"summary":True, "step":_epoch, "scope":locals(), "train__loss:": loss.item(), "v_iou:": val_iou})
                nsml.save(str(epoch + 1))
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))

            #print('x size: {}, label: {}'.format(x.size(), label[-1]))
