import os
import torch
from lightning_fabric.loggers import TensorBoardLogger
from torch import nn
from tqdm import tqdm

from src.utils import get_logger
from torchmetrics import Accuracy, Dice
from lightning.fabric import Fabric

log = get_logger(__name__)



class MyTrain:
    """
    所有的训练步骤
    """

    def __init__(
            self,
            net: torch.nn.Module,
            loss: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            TensorBoardLog: TensorBoardLogger,
            num_classes: int,
            accelerator: str = "auto",
            strategy: str = "auto",
            devices: int = 1,
            precision: str = "32",


    ):
        """

        :param net: torch.nn.Module类型的网络结构
        :param loss: 如torch.nn.CrossEntropyLoss()类型的损失
        :param optimizer: torch.optim.Optimizer类型的优化器
        :param TensorBoardLog: TensorBoardLogger类型的日志

        :param num_classes: 网络分类数
        :param accelerator: Fabric加速器类型:默认自动选择."cpu","cuda", "mps", "gpu", "tpu", "auto".
        :param strategy: Fabric加速器策略;默认为自动选择."single_device", "dp", "ddp", "ddp_spawn", "deepspeed", "ddp_sharded".
        :param devices: Fabric加速器设备数:默认为自动选择.list[int] 代表指定索引设备运行,.int 代表指定多个设备.-1 代表所有设备一起运行.
        :param precision: Fabric加速器精度： 默认以32位精度运行;
            "32": 32位精度（模型权重保留在torch.float32中).
            "16-mixed"：(16位混合精度（模型重量保留在torc.float32中).
            "bf16-mixed":(16位bfloat混合精度（模型重量保留在torc.float32中).
            ”16_true": 16位精度(模型权重转换为 torch.float16).
            "bf16-true":16位 bfloat 精度(模型权重转换到 torch.bfloat16).
            "64":64位(双)精度(模型权重转换为 torch.float64).
            "transformer-engine":通过 TransformerEngine (Hopper GPU 和更高版本)实现8位混合精度.
            详情： https://lightning.ai/docs/fabric/stable/fundamentals/precision.html.

        """
        super().__init__()
        # 初始化Fabric，用于切换多设备/分布式/混合精度
        self.fabric = Fabric(accelerator=accelerator, strategy=strategy, devices=devices, precision=precision)

        self.num_classes = num_classes
        # 初始化网络
        with self.fabric.init_module():
            self.net = net
        # 损失函数
        self.loss = loss
        # 优化器
        self.optim = optimizer(params=self.net.parameters())
        # 评价标准
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=2).to(self.fabric.device)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=2).to(self.fabric.device)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=2).to(self.fabric.device)

        # 为了记录到目前为止最好的验证准确性，方便保存最优模型
        self.val_acc_best = 0

        # tensorboard 记录器
        self.tb_log = TensorBoardLog
        self.tb_native_log = TensorBoardLog.experiment

    def step(self, batch):
        x, y = batch
        preds = self.net(x).view(-1, self.num_classes)
        targets = y.view(-1, 1)[:, 0]
        loss = self.loss(preds, targets)
        return loss, preds, targets

    def training(self, dataset_loader):
        # 开启Fabric，覆盖当前变量
        self.fabric.launch()
        self.net, self.optim = self.fabric.setup(self.net, self.optim)
        dataset_loader = self.fabric.setup_dataloaders(dataset_loader)
        # 启动训练
        self.net.train()
        total_loss = 0
        # 进度条
        with tqdm(dataset_loader, desc=f'训练 : ', colour="blue", leave=False) as t:
            for batch in t:
                # 开始迭代
                self.optim.zero_grad()
                loss, preds, targets = self.step(batch)
                self.fabric.backward(loss)
                self.optim.step()
                # 计算指标
                acc = self.train_acc(preds, targets)
                total_loss += loss.item()
                # 在每个步骤中更新进度条
                t.set_postfix(loss=format(loss, '.3f'), acc=format(acc, '.3f'))

        loss = loss / len(dataset_loader)
        acc = self.train_acc.compute()  # 使用自定义累积的所有批次的度量
        log.info(f"train_loss:<{loss}")
        log.info(f"train_acc:<{acc}>")
        self.train_acc.reset()  # 重置内部状态，以便度量为新数据做好准备
        return acc, loss

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

        loss = loss / len(dataset_loader)
        acc = self.val_acc.compute()  # 使用自定义累积的所有批次的度量
        log.info(f"val_loss:<{loss}>")
        log.info(f"val_acc:<{acc}>")
        log.info("val_acc_best:<{self.val_acc_best}>")
        self.val_acc.reset()  # 重置内部状态，以便度量为新数据做好准备
        return acc, loss

    def test(self, dataset_loader, best_model):
        self.init_weights()
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

        loss = loss / len(dataset_loader)
        acc = self.val_acc.compute()  # 使用自定义累积的所有批次的度量
        log.info(f"test_loss:<{loss}>")
        log.info(f"test_acc:<{acc}>")
        self.test_acc.reset()

    def save_model(self, save_path, new_loss):
        checkpoint = {
            "net": self.net.state_dict(),
            'optimizer': self.optim.state_dict(),
            "loss": new_loss
        }
        log.info(f"Save Model, Path:{save_path}，==>loss:{new_loss}\n")
        self.fabric.save(save_path,checkpoint)

    def load_model(self, load_path, strict=True):
        """
        加载模型
        :param load_path: 加载路径
        :param strict: 加载检查点通常是“严格”的，这意味着检查点中的参数名称必须与模型中的参数名称匹配。 但是，在加载检查点进行微调或迁移学习时，可能会发生只有部分参数与模型匹配的情况。 对于这种情况，您可以禁用严格加载以避免错误：
        """
        full_checkpoint = self.fabric.load(load_path, strict=strict)
        log.info('=> load pretrained  {} => loss :{}'.format(load_path, full_checkpoint["loss"]))
        self.net.load_state_dict(full_checkpoint["net"])
        self.optim.load_state_dict(full_checkpoint["optimizer"])

    def load_model_old(self, load_path):
        pre_model = torch.load(load_path)
        pretrained_dict = pre_model['net']
        self.optim.load_state_dict(pre_model['optimizer'])
        log.info('=> load pretrained  {} => loss :{}'.format(load_path, pre_model["loss"]))
        model_dict = self.net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        for k, _ in pretrained_dict.items():
            log.info('=> load {} in  {}'.format(k, load_path))
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

    def init_weights(self):
        log.info('=> init Conv2d and BatchNorm2d ')
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




