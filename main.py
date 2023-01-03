import os
import hydra
from omegaconf import DictConfig
from tqdm import trange
from src.utils import seed_torch, get_logger

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="experiment.yaml")
def main(config_global: DictConfig):
    """

    """
    # 初始化个性环境
    config = config_global.my_envs

    # 在pytorch，numpy和python中设置随机数生成器的种子
    seed_torch(config.train.get("seed"))
    log.info(f"随机数种子为 <{config.train.get('seed')}>")

    # 必要时将相对ckpt路径转换为绝对路径
    config.train.resume_from_checkpoint = os.path.join(
        hydra.utils.get_original_cwd(), config.train.get("resume_from_checkpoint")
    )
    log.info(f"预训练模型路径 <{config.train.resume_from_checkpoint}>")

    # 初始化数据加载器
    log.info(f"初始化化数据模块 <{config.datamodule._target_}>\n 路径: <{config.datamodule.data_dir}>")
    train_dataloader = hydra.utils.instantiate(config.datamodule, mode="train").train_dataloader()
    # hydra.utils.instantiate(config.datamodule, mode="train")[0] # 测试数据__getter__函数是否正确
    val_dataloader = hydra.utils.instantiate(config.datamodule, mode="val").val_dataloader()
    test_dataloader = hydra.utils.instantiate(config.datamodule, mode="test").test_dataloader()

    # 初始化模型加载器
    log.info(f"初始化训练器 <{config.model._target_}>")
    train = hydra.utils.instantiate(config.model)

    # 训练模型
    if config_global.get("train"):
        log.info("开始训练!")
        train.init_weights(pretrained=config.train.resume_from_checkpoint)
        epochs = config.train.epochs
        with trange(epochs,colour="red") as t:
            for epoch in t:
                t.set_description(f"总进度 Epoch {epoch}/{epochs} :")
                train_acc, train_loss = train.training(train_dataloader)
                val_acc, val_loss = train.validation(val_dataloader)
                t.set_postfix(train_loss=format(train_loss, '.3f'), train_acc=format(train_acc, '.3f'),
                              val_loss=format(train_loss, '.3f'), val_acc=format(val_acc, '.3f'))
                log.info(f"Epoch {epoch}/{epochs} : train_loss={train_loss} train_acc={train_acc} "
                         f"val_loss={val_loss}  val_acc={val_acc}")
                t.update(1)

    # 测试模型
    if config_global.get("test"):
        log.info("开始测试!")
        train.test(test_dataloader, config.train.ckpt_path)

    # 确保一切正常关闭
    log.info("Finalizing!")


if __name__ == "__main__":
    main()
