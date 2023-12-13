import logging
import random
import numpy as np
import torch
import os


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果您使用的是多GPU。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pt_to_onnx(save_onnx_path,model, torch_inputs):
    """
    将pytorch模型转换为onnx样例，根据实际修改
    :param save_onnx_path: 保存onnx的路径
    :param model: pytorch模型
    :param torch_inputs: 输入样例
    :return:
    """
    import onnxruntime as rt
    # 保存onnx,动态维度
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'feature', 2: "pts"},
        'output': {0: 'batch_size', 1: "pts", 2: 'feature'},
    }

    torch.onnx.export(model, torch_inputs, save_onnx_path, opset_version=12, export_params=True, verbose=False,
                      input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

    # 推理
    sess = rt.InferenceSession(save_onnx_path,providers=['CPUExecutionProvider'])  # ['DmlExecutionProvider', 'GPUExecutionProvider'，'CPUExecutionProvider']
    X_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    numpy_input = torch_inputs.detach().cpu().numpy()
    rX = numpy_input.reshape([1, numpy_input.shape[0], numpy_input.shape[1]]).astype(np.float32)
    result = sess.run([output_name], {X_name: rX})[0][0]
    return result


def get_logger(name=__name__) -> logging.Logger:
    """python命令行记录器"""

    logger = logging.getLogger(name)
    return logger
