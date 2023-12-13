import os
import hydra
from lightning_fabric import fabric
from lightning_fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig
from tqdm import trange
from src.utils import seed_torch, get_logger

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="experiment.yaml")
def main(config_global: DictConfig):
    """

    """
    log.info(f"description: {config_global.description}")
    # 初始化个性环境
    config = config_global.my_envs

    # 在pytorch，numpy和python中设置随机数生成器的种子
    seed_torch(config.train.get("seed"))
    fabric.seed_everything(config.train.get("seed"))
    log.info(f"seed is : <{config.train.get('seed')}>")

    # 必要时将相对ckpt路径转换为绝对路径
    config.train.resume_from_checkpoint = os.path.join(
        hydra.utils.get_original_cwd(), config.train.get("resume_from_checkpoint")
    )
    log.info(f"pretrained model path: <{config.train.resume_from_checkpoint}>")

    # 初始化数据加载器
    log.info(f"Initialize the DataModule: <{config.datamodule._target_}>\t path: <{config.datamodule.data_dir}>")
    train_dataloader = hydra.utils.instantiate(config.datamodule, mode="train").train_dataloader()
    # hydra.utils.instantiate(config.datamodule, mode="train")[0] # 测试数据__getter__函数是否正确
    val_dataloader = hydra.utils.instantiate(config.datamodule, mode="val").val_dataloader()

    # 初始化流程加载器
    log.info(f"Initialize the Trainer <{config.pipeline._target_}>")
    tb_logger = TensorBoardLogger(root_dir="logs/runs/", name=config_global.name,version=config_global.version, flush_secs=10)
    log.info(
        f"From the command line, use <tensorboard --logdir={os.path.abspath(tb_logger.log_dir)} > to view the current TensorBoard record.")
    train = hydra.utils.instantiate(config.pipeline, TensorBoardLog=tb_logger)
    # 测试训练性能
    train.analytical_performance(hydra.utils.instantiate(config.datamodule, mode="test_performance").train_dataloader())
    # 训练模型
    log.info("start training!")
    train.load_model(load_path=config.train.resume_from_checkpoint)
    epochs = config.train.epochs
    check_train_loss = 1e4
    check_val_loss = 1e4
    with trange(epochs, colour="red") as t:
        for epoch in t:
            t.set_description(f"总进度 Epoch {epoch}/{epochs} :")
            train_acc, train_loss = train.training(train_dataloader)
            val_acc, val_loss = train.validation(val_dataloader)
            t.set_postfix(train_loss=format(train_loss, '.3f'), train_acc=format(train_acc, '.3f'),
                          val_loss=format(train_loss, '.3f'), val_acc=format(val_acc, '.3f'))
            # 记录到日志
            log.info(f"Epoch {epoch}/{epochs} : train_loss={train_loss} train_acc={train_acc} "
                     f"val_loss={val_loss}  val_acc={val_acc}")
            # 记录到tb
            tb_logger.log_metrics({"train_loss": train_loss, "train_acc": train_acc,
                                   "val_loss": val_loss, "val_acc": val_acc}, step=epoch)

            # 保存模型
            if train_loss < check_train_loss:
                train.save_model(save_path=config.train.get("resume_from_checkpoint"), new_loss=train_loss)
                check_train_loss = train_loss
            if val_loss < check_val_loss:
                train.save_model(save_path=config.train.get("ckpt_path"), new_loss=val_loss)
                check_val_loss = val_loss

    # 确保一切正常关闭
    log.info("Finalizing!")
    tb_logger.finalize("Finalizing")


if __name__ == "__main__":
    main()
