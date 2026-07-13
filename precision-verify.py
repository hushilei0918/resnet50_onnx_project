# tests/precision_verify.py
import torch
import numpy as np
import onnxruntime as ort
from torchvision import models
from torchvision.models import ResNet50_Weights

def verify_precision():
    """验证 PyTorch 和 ONNX 推理结果的一致性"""
    print("=== 开始精度对齐验证 ===")
    
    # 1. 加载 PyTorch 模型
    torch_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    torch_model.eval()
    torch_model.to("cpu")
    
    # 2. 加载 ONNX 模型
    ort_session = ort.InferenceSession(
        "./models/resnet50_dynamic.onnx",
        providers=["CPUExecutionProvider"]
    )
    input_name = ort_session.get_inputs()[0].name
    
    # 3. 生成相同的随机测试输入（固定种子，可复现）
    torch.manual_seed(42)
    test_input_torch = torch.randn(4, 3, 224, 224)  # batch=4 测试动态batch
    test_input_np = test_input_torch.numpy().astype(np.float32)
    
    # 4. PyTorch 推理
    with torch.no_grad():
        torch_output = torch_model(test_input_torch).numpy()
    
    # 5. ONNX Runtime 推理
    ort_output = ort_session.run(None, {input_name: test_input_np})[0]
    
    # 6. 计算误差
    mae = np.mean(np.abs(torch_output - ort_output))
    max_error = np.max(np.abs(torch_output - ort_output))
    
    print(f"测试 batch 大小: {test_input_torch.shape[0]}")
    print(f"PyTorch 输出形状: {torch_output.shape}")
    print(f"ONNX 输出形状: {ort_output.shape}")
    print(f"平均绝对误差 (MAE): {mae:.8f}")
    print(f"最大绝对误差: {max_error:.8f}")
    
    # 误差小于 1e-4 为合格
    if max_error < 1e-4:
        print("✅ 精度验证通过，误差属于浮点计算正常差异")
        return True
    else:
        print("⚠️ 误差过大，请检查导出配置和预处理逻辑")
        return False


if __name__ == "__main__":
    verify_precision()
