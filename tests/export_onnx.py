# tests/export_onnx.py
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
import onnx
import os

def export_resnet_onnx(save_path: str = "./models/resnet50_dynamic.onnx"):
    """导出动态 batch 的 ResNet50 ONNX 模型"""
    
    # 1. 加载 PyTorch 模型（和基线版本完全一致）
    print("1. 加载 PyTorch 模型...")
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()  # 必须切评估模式
    model.to("cpu")  # CPU 导出兼容性最好
    
    # 2. 构造示例输入（形状和真实推理一致）
    print("2. 构造示例输入...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 3. 定义动态轴（batch 维度可变，支持批量推理）
    dynamic_axes = {
        "images": {0: "batch_size"},    # 输入第0维是动态 batch
        "logits": {0: "batch_size"}     # 输出第0维对应动态 batch
    }
    
    # 4. 执行导出
    print("3. 开始导出 ONNX 模型...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with torch.no_grad():
        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=save_path,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,          # 算子集版本，兼容性好
            do_constant_folding=True,  # 开启常量折叠优化
            verbose=False
        )
    
    print(f"✅ ONNX 模型导出成功，保存路径: {save_path}")
    return save_path


def check_onnx_model(onnx_path: str):
    """检查 ONNX 模型结构是否合法"""
    print("\n4. 检查 ONNX 模型结构...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    
    # 打印模型信息
    print("✅ 模型结构合法")
    print(f"  输入节点: {[inp.name for inp in model.graph.input]}")
    print(f"  输出节点: {[out.name for out in model.graph.output]}")
    print(f"  算子集版本: {model.opset_import[0].version}")


if __name__ == "__main__":
    onnx_file = export_resnet_onnx()
    check_onnx_model(onnx_file)
