
<div align="center">

# Pytorch-Hydra-fabric 模板 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Conference](https://img.shields.io/badge/User-Sindre-orange)]()
[![Conference](https://img.shields.io/badge/Update-2023-blue)]()

</div>
 
## 项目描述
1. 需求
   1. 框架重复性过高，很多模块是重复性的。
   2. pytorch-lighting学习成本高，失去了原生pytorch感觉。
   3. 实验需要改变部分参数及记录相关日志，方便排查问题。
   4. 多GPU时编码重复，混合精度编码复杂；
   5. 版本及实验混乱，训练后还要依次测试。
   6. 在追求性能时，需要排查哪个算子慢；
2. 实现方式
   1. 使用hydra为配置核心。
   2. 将pytorch通用的组件实例化。
   3. 使用lighting-fabric实现多GPU/多节点/混合精度。
   4. 启动tensorboard统计及渲染信息；
   5. 用torch_tb_profiler进行性能监测(windows需要torch>2.1,否则会因为json内“/” 问题报错（也可以手动转成“//”))
   

3. 效果
<figure class="half" align="center">
<img src="img/img.png" width = "100%" height = "50%" >
<img src="img/img_0.png" width = "50%" height = "50%" >
<img src="img/img_1.png" width = "50%" height = "50%" >
<img src="img/img_2.png" width = "50%" height = "50%" >
<img src="img/img_3.png" width = "50%" height = "50%" >
<img src="img/img_4.png" width = "50%" height = "50%" >
<img src="img/img_5.png" width = "50%" height = "50%" >
<img src="img/img_6.png" width = "50%" height = "50%" >
<img src="img/img_7.png" width = "50%" height = "50%" >
</figure>


## 如何运行   
第一步，安装依赖  
```bash
# 克隆项目   
git clone https://github.com/SindreYang/pytorch-hydra-template

# 安装依赖   
cd pytorch-hydra-template
pip install -r requirements.txt
```

使用默认配置训练模型

```bash
python main.py
```

选择实验配置[configs/my_envs/](configs/my_envs/)训练模型 

```bash
python main.py my_envs=experiment_name.yaml
```

您可以像这样从命令行覆盖任何参数

```bash
python main.py my_envs.train.epochs=20
```


## 设置为包，方便导入
此项目设置为一个包，这意味着您现在可以轻松地将任何文件导入到任何其他文件中，如下所示：
```bash
pip install -e .
```


## 项目结构

```
├── configs                   <- Hydra 配置文件
│   ├── my_envs                  <- 实验配置
│   ├── hparams_search           <- 超参数搜索配置
│   └── experiment.yaml          <- 主要配置
│
├── datasets                   <- 训练数据
│
├── img                    <- readme图片资源
├── logs                   <- Hydra 和 PyTorch 记录器生成的日志
│
├── scripts                <- Shell 脚本
│
├── src                    <- 源代码
│   ├── datamodules              <- 数据模块
│   ├── models                   <- 模型
│   ├── pipeline                 <- 流程管理
│   └── utils                    <- 工具脚本
│
│
├── main.py                <- 训练and测试
│
├── requirements.txt          <- 用于安装 python 依赖项的文件
├── setup.py                  <- 打包成pip安装包
└── README.md
```

<br>

