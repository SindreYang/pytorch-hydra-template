import os
import torch
from torch import nn
from tqdm import tqdm

from src.utils import get_logger
from torchmetrics import Accuracy, Dice


log = get_logger(__name__)


class MyTrain:
    """

    """

    def __init__(
            self,
            net: torch.nn.Module,  # 网络
            loss: torch.nn.Module,  # 损失torch.nn.CrossEntropyLoss()
            optimizer,
            device: str,
            num_classes: int,
    ):
        super().__init__()
        #
        self.device = device
        self.num_classes = num_classes
        # 初始化网络
        self.net = net.to(self.device)
        # 损失函数
        self.loss = loss
        # 优化器
        self.optim = optimizer(params=self.net.parameters())
        # 评价标准
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=2).to(self.device)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=2).to(self.device)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=2).to(self.device)



        # 为了记录到目前为止最好的验证准确性，方便保存最优模型
        self.val_acc_best = 0

    def step(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        preds = self.net(x).view(-1, self.num_classes)
        targets = y.view(-1, 1)[:, 0]
        loss = self.loss(preds, targets)
        return loss, preds, targets

    def training(self, dataset_loader):
        self.net.train()
        total_loss =0
        # 进度条
        with tqdm(dataset_loader,desc=f'训练 : ',colour="blue",leave=False) as t:
            for batch in t:
                loss, preds, targets = self.step(batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                acc=self.train_acc(preds, targets)
                total_loss+=loss.item()
                # 在每个步骤中更新进度条
                t.set_postfix(loss=format(loss, '.3f'), acc=format(acc, '.3f'))
                t.update(1)
        loss=loss/len(dataset_loader)
        acc = self.train_acc.compute()  # 使用自定义累积的所有批次的度量
        log.info(f"train_loss:<{loss}")
        log.info(f"train_acc:<{acc}>")
        self.train_acc.reset()  # 重置内部状态，以便度量为新数据做好准备
        return  acc ,loss

    def validation(self, dataset_loader):
        self.net.eval()
        total_loss = 0
        with torch.no_grad():
            with tqdm(dataset_loader, desc=f'验证 : ', colour="green", leave=False) as t:
                for batch in t:
                    loss, preds, targets = self.step(batch)
                    acc = self.val_acc(preds, targets)
                    total_loss += loss.item()
                    # 在每个步骤中更新进度条
                    t.set_postfix(loss=format(loss, '.3f'), acc=format(acc, '.3f'))
                    t.update(1)
        loss = loss / len(dataset_loader)
        acc = self.val_acc.compute()  # 使用自定义累积的所有批次的度量
        log.info(f"val_loss:<{loss}>")
        log.info(f"val_acc:<{acc}>")
        log.info("val_acc_best:<{self.val_acc_best}>")
        self.val_acc.reset()  # 重置内部状态，以便度量为新数据做好准备
        return acc, loss

    def test(self, dataset_loader, best_model):
        self.init_weights(best_model)
        self.net.eval()
        total_loss = 0
        with torch.no_grad():
            with tqdm(dataset_loader, desc=f'测试 : ') as t:
                for batch in t:
                    loss, preds, targets = self.step(batch)
                    acc = self.test_acc(preds, targets)
                    total_loss += loss.item()
                    # 在每个步骤中更新进度条
                    t.set_postfix(loss=format(loss, '.3f'), acc=format(acc, '.3f'))
                    t.update(1)
        loss = loss / len(dataset_loader)
        acc = self.val_acc.compute()  # 使用自定义累积的所有批次的度量
        log.info(f"test_loss:<{loss}>")
        log.info(f"test_acc:<{acc}>")
        self.test_acc.reset()

    def init_weights(self, pretrained=''):
        log.info('=> 正态分布的init权重')
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            log.info('=> 加载预训练模型 {}'.format(pretrained))
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                log.info('=> 加载 {} 预训练模型 {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)
