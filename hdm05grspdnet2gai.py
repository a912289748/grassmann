from pathlib import Path

import torch
import torch.optim as optim
import numpy as np
import time

from mAtt.mAtt import GrweightAttentionManifold, SPDweightAttentionManifold
from myutil import AverageMeter
import math
from tqdm import tqdm
from model import BMS_Net
import argparse
from pprint import pprint
from myutil import group_list
from myutil import bcolors
from torch import nn
from BiMap import BiMap, FrMap, BiMapmul, FrMapmul
from ReEig import ReEigFunction
from LogEig import LogEigFunction
from SubConv import SubConvFunction
from SubLogEig import SubLogEigFunction
import spd.nn as nn_spd
from spd.optimizers import MixOptimizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="hdm05fromspdnetdata/hdm05gr.npz", help="path to dataset"
)
parser.add_argument("--ep", type=float, default=1e-4, help="epsilon for ReEig layer")
parser.add_argument("--batch_size", type=int, default=30, help="batch size")
parser.add_argument("--n_atom", type=int, default=28, help="number of dictionary atom")
parser.add_argument("--margin1", type=float, default=1, help="margin for triplet loss")
parser.add_argument("--margin2", type=float, default=1, help="margin for intra loss")
parser.add_argument(
    "--dims",
    type=str,
    default="400,200,100,50",
    help="dimensionality for extracting feature",
)
parser.add_argument("--n_class", type=int, default=7, help="number of class")
parser.add_argument(
    "--lambda1", type=float, default=1.2, help="trade-off coefficient for triplet loss"
)
parser.add_argument(
    "--lambda2", type=float, default=0.7, help="trade-off coefficient for intra loss"
)
parser.add_argument(
    "--save_folder",
    type=str,
    default="./hdm05fromspdnetmodel/hdm05msnetmodels/afew_msnetlogw",
    help="path to save model",
)
parser.add_argument(
    "--use_tensorboard", type=bool, default=True, help="whether to use tensorboard"
)
parser.add_argument(
    "--metric_method",
    type=str,
    default="log_w",
    help="method for feature metric, log, log_w or jbld",
)
parser.add_argument(
    "--log_dim", type=int, default=20, help="dimensionality for log metric"
)
parser.add_argument("--n_fc", type=int, default=1, help="number of fc layers")
parser.add_argument(
    "--n_fc_node", type=int, default=4096, help="number of nodes in fc layer"
)

args = parser.parse_args()
pprint(args)

best_correct = 0
best_epoch = 0

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)

dataset = np.load(args.dataset)
train_x = dataset["trX"]
train_y = dataset["trY"].astype(np.int64)
test_x = dataset["teX"]
test_y = dataset["teY"].astype(np.int64)


class SPDNet(nn.Module):
    """Docstring for SPDNet. """

    def __init__(self):
        super(SPDNet, self).__init__()
        channel = 8



        self.frmap1 = FrMapmul(93, 80,channel)
        self.qr1 = nn_spd.Qrs()
        self.pr1 = nn_spd.Projmap()
        self.avg1 = torch.nn.AvgPool2d(2)
        self.or1 = nn_spd.Orthmap()

        self.frmap2 = FrMapmul(40, 30,channel)
        self.qr2 = nn_spd.Qrs()
        self.pr2 = nn_spd.Projmap()
        self.avg2 = torch.nn.AvgPool2d(2)
        self.or2 = nn_spd.Orthmap()

        self.pr3 = nn_spd.Projmap()

        self.frmap3 = FrMapmul(50, 33,channel)

        self.linear = nn.Linear(1800, 130).double()

    def forward(self, x):
        x1 = (self.frmap1(x))

        x1 = self.qr1(x1)
        x1 = self.pr1(x1)
        x1 = self.avg1(x1)
        x1 = self.or1(x1)
        x1 = self.frmap2(x1)
        x1 = self.qr2(x1)
        x1 = self.pr2(x1)
        x1 = self.avg2(x1)
        x1 = self.or2(x1)
        x1 = self.pr3(x1)
        x1 = x1.view(x.size(0), -1)
        x1 = self.linear(x1)
        return x1


model = SPDNet()
# model.load_state_dict(torch.load(r'C:\Users\l\Desktop\备份代码\20230829代码\BoMS-master改msnet+layerbn+zazhi-graph\BoMS-master\hdm05fromspdnetmodel\hdm05msnetmodels\afew_msnetlogw_27.pkl')['state_dict'])
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=5e-4, momentum=0.0)
# opti = MixOptimizer(model.parameters(), lr=1e-2)

loss_function = torch.nn.CrossEntropyLoss()
n_trained_batch = 0


def train(epoch, optimizer):
    global n_trained_batch
    global train_x
    global train_y

    meter_acc = AverageMeter()
    meter_loss_total = AverageMeter()
    meter_loss_cls = AverageMeter()
    meter_loss_triplet = AverageMeter()
    meter_loss_intra = AverageMeter()

    model.train()
    batch_idx = 0
    index = np.random.permutation(len(train_x))
    train_x = train_x[index]
    train_y = train_y[index]
    for (data, target) in tqdm(
            group_list(train_x, train_y, args.batch_size),
            total=len(train_x) // args.batch_size,
    ):
        data = torch.from_numpy(data).double()
        target = torch.from_numpy(target)
        n_data = data.size(0)

        output = model(data)

        triplet_loss = 0
        intra_loss = 0
        classifier_loss = loss_function(output, target)

        total_loss = (
            classifier_loss
        )
        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        for layer in model.named_children():
            if layer[0].startswith("bimap"):
                q, r = torch.qr(layer[1].weight.data.permute(0,2,1))
                layer[1].weight.data = (
                        q @ (torch.sign(torch.sign(r) + 0.5))
                ).permute(0,2,1)


        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()

        total_loss = total_loss.item()
        classifier_loss = classifier_loss.item()

        meter_acc.update(correct / n_data, n_data)
        meter_loss_total.update(total_loss, n_data)
        meter_loss_cls.update(classifier_loss, n_data)
        meter_loss_triplet.update(triplet_loss, n_data)
        meter_loss_intra.update(intra_loss, n_data)

        if batch_idx % (len(train_x) // args.batch_size // 5) == 0:
            pstr = (
                f"Epoch:{epoch:2} Batch_idx:{batch_idx:2} "
                f"Loss:{bcolors.OKGREEN}{total_loss:.2f}{bcolors.ENDC}"
                f"({classifier_loss:.2f}/{triplet_loss:.2f}/{intra_loss:.2f})\t"
                f"Acc:{bcolors.OKGREEN}{correct / n_data * 100:.2f}{bcolors.ENDC}\n"
                f"Average: Loss:{meter_loss_total.avg:.2f}"
                f"({meter_loss_cls.avg:.2f}/{meter_loss_triplet.avg:.2f}/{meter_loss_intra.avg:.2f})"
                f"\tTime:{time.ctime()}"
                f"\tAcc:{bcolors.OKGREEN}{meter_acc.avg * 100:.2f}{bcolors.ENDC}"
            )
            tqdm.write(pstr)
        batch_idx += 1
        n_trained_batch += 1


def test(epoch):
    global best_correct
    global best_epoch
    global test_x
    global test_y

    meter_acc = AverageMeter()
    meter_loss_total = AverageMeter()
    meter_loss_cls = AverageMeter()
    meter_loss_triplet = AverageMeter()
    meter_loss_intra = AverageMeter()

    model.eval()
    correct = 0
    for data, target in tqdm(
            group_list(test_x, test_y, args.batch_size),
            total=len(test_x) // args.batch_size,
    ):
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        n_data = data.size(0)

        output = model(data)

        triplet_loss = 0
        intra_loss = 0
        classifier_loss = loss_function(output, target)

        total_loss = (
            classifier_loss
        )

        total_loss = total_loss.item()
        classifier_loss = classifier_loss.item()

        pred = output.data.max(1, keepdim=True)[1]
        current_correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()
        correct += current_correct

        meter_acc.update(current_correct / n_data, n_data)
        meter_loss_total.update(total_loss, n_data)
        meter_loss_cls.update(classifier_loss, n_data)
        meter_loss_triplet.update(triplet_loss, n_data)
        meter_loss_intra.update(intra_loss, n_data)

    if correct > best_correct:
        best_correct = correct
        best_epoch = epoch
        state = {
            "acc": correct / test_x.shape[0],
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "args": args,
        }
        torch.save(state, args.save_folder)  # 只保存效果最好的模型 覆盖写法

    states = {
        "acc": correct / test_x.shape[0],
        "epoch": epoch,
        "state_dict": model.state_dict(),

        "args": args,
    }
    torch.save(states, args.save_folder + '_' + str(epoch) + '.pkl')

    print(
        f"Epoch:{epoch:2} "
        f"Loss:{meter_loss_total.avg:.2f}"
        f"({meter_loss_cls.avg:.2f}/{meter_loss_triplet.avg:.2f}/{meter_loss_intra.avg:.2f})"
        f"\tAcc:{meter_acc.avg * 100:.2f}\tTime:{time.ctime()}"
    )
    with open(str(Path(args.save_folder).parent)+'/output.txt', 'a') as file:
        file.write(str(correct / test_x.shape[0]) + "\n")
    print(f"Best epoch:{best_epoch} Accuracy:{best_correct / len(test_x) * 100:.2f}\n")
    print("=" * 20)
    return correct


if __name__ == "__main__":
    for epoch in range(1, 100000):
        train(epoch, optimizer)
        # if epoch % 10 == 0:
        with torch.no_grad():
            correct = test(epoch)
