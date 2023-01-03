# 基于隐函数的3D牙齿网格生成  

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2022-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2022-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## 项目描述
基于隐函数实现牙冠生成

## 如何运行   
第一步，安装依赖  
```bash
# 克隆项目   
git clone https://github.com/YourGithubName/deep-learning-project-template

# 安装依赖   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   

然后进入项目
 ```bash
# 项目路径
cd project

# 运行模块
python lit_classifier_main.py    
```

## 设置为包，方便导入
此项目设置为一个包，这意味着您现在可以轻松地将任何文件导入到任何其他文件中，如下所示：
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
 @article{xxxx
  title={IF-TG},
  author={sindre},
  journal={SC},
  year={2022}
}
```   
